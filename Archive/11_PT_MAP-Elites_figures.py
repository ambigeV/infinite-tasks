# # General Config 

# ## Import 

from base_talos_01 import *
from plot_01 import *

import itertools
from itertools import product
import numpy as np
from tqdm import tqdm 
from time import time 
import multiprocessing as mp
import os
import pickle 
from scipy.stats.qmc import Sobol
from math import log, ceil  
from collections.abc import Iterable
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay
from scipy.spatial import distance 
from collections import defaultdict, deque
import sys
import math

# ## General

if os.uname()[1] == "multivac":
    n_proc = 60
elif os.uname()[1] == "evo256":
    n_proc = 100
else:
    n_proc = os.cpu_count()-2


# # MAP-Elites

def create_logdir(config):
    if config['use_logging']:
        now = datetime.datetime.now()
        logdir = "/home/pal/notebooks/data/360d_reflex/datasets/" + now.strftime("%Y/%m/%d/%Hh%Mm%Ss") + "/"
        os.makedirs(logdir, exist_ok=True)
        print(logdir)
        config["logdir"] = logdir


# ## Command Sampler 

# ### Variation Operators

def iso_dd(x, y, command_config):
    line_sigma, iso_sigma, dim = command_config['line_sigma'], command_config['iso_sigma'], command_config["dim"]
    candidate = x + np.random.normal(0, iso_sigma) + np.random.normal(0, line_sigma) * (y-x) 
    return np.clip(candidate, np.zeros(dim), np.ones(dim))


def sbx(x, y, command_config):
    '''
    SBX (cf Deb 2001, p 113) Simulated Binary Crossover

    A large value ef eta gives a higher probablitity for
    creating a `near-parent' solutions and a small value allows
    distant solutions to be selected as offspring.
    '''
    eta = 10 # command_config["eta"] 
    xl = 0
    xu = 1
    z = x.copy()
    r1 = np.random.random(size=len(x))
    r2 = np.random.random(size=len(x))

    
    for i in range(0, len(x)):
        if abs(x[i] - y[i]) > 1e-15:
            
            x1 = min(x[i], y[i])
            x2 = max(x[i], y[i])
            
            
            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            rand = r1[i]
            
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
                
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            c1 = min(max(c1, xl), xu)
            c2 = min(max(c2, xl), xu)

            if r2[i] <= 0.5:
                z[i] = c2
            else:
                z[i] = c1

    return z

variation_operators = {
    "iso_dd": iso_dd,
    "sbx": sbx,
}


# ### select parent

def select_parent(archive): 
    return archive.elites[np.random.randint(archive.n_cells)]


# ### use regressor

def regression(s, archive, config):
    """ linear regression  """
    _, idx = archive.tree.query(s, k=1)
    indexes = archive.centroid_neighbors[idx]  # find the direct neighbors using the precomputed delauney from the centroids 
    X = [archive.elites[i]["situation"] for i in indexes]
    Y = [archive.elites[i]["command"] for i in indexes]
    reg = LinearRegression().fit(X, Y)
    c = reg.predict(np.array([s]))[0] 
    dim = len(c)
    if config["regression_mutation_scaling"]:
        return np.clip(c + np.random.normal(0, config["linreg_sigma"]) * np.std(Y, axis=0), np.zeros(dim), np.ones(dim))
    else:
        return np.clip(c + np.random.normal(0, config["linreg_sigma"]), np.zeros(dim), np.ones(dim))


# ### Select task

def random_task(archive, config):
    # s not used 
    if config["continuous"]:
        selected_task = np.random.random(config["situation_config"]["dim"])
    else:
        idx = np.random.randint(0, archive.n_cells)
        selected_task = archive.centroids[idx]
    return selected_task


def closest2parent_tournament(s, archive, config):
    k = archive.k_closest2parent
    if config["continuous"]:
        tasks = np.random.random((k, config["situation_config"]["dim"]))
        _, indexes = archive.tree.query(tasks, k=1)
        situations = [archive.elites[i]["situation"] for i in indexes]
        distances = distance.cdist(situations, [s], "euclidean") 
        selected_task = tasks[np.argmin(distances)]
    else:
        indexes = np.random.randint(0, archive.n_cells, k)
        situations = [archive.elites[i]["situation"] for i in indexes]
        distances = distance.cdist(situations, [s], "euclidean") 
        selected_task = situations[np.argmin(distances)]
    return selected_task


# ### Sampler

def sampler(batch_size, starting_id, archive, config, verbose=0):
    command_config, situation_config = config["command_config"], config["situation_config"]
    samples, elite_id = [], starting_id
    for _ in range(batch_size):        
        if np.random.random() < config["proba_regression"]:
            s = random_task(archive, config)
            c = regression(s, archive, config)
            selected_operator = "regression"
        else:
            x, y = select_parent(archive), select_parent(archive)
            c = variation_operators[config["variation_operator"]](x["command"], y["command"], command_config)
            if config["use_closest2parent"]:
                s = closest2parent_tournament(x["situation"], archive, config)
                selected_operator = "sbx_closest2parent"
            else:
                s = random_task(archive, config)
                selected_operator = "sbx"

        samples.append((c, s, elite_id, command_config["bounds"], situation_config["bounds"], selected_operator, verbose, config["erase"]))
        elite_id += 1 
    return samples


# ## Command Evaluation

# +
def worker(job_queue, res_queue):
    while True:
        job = job_queue.get()
        if job == "Done":
            job_queue.close()
            res_queue.close()
            break 
        else:
            f, args = job
            res_queue.put(f(*args))
            
def make_box_jobs(samples, eval_func, verbose=0):
    def jobs():
        for sample in samples:
            yield (eval_func, sample)
    return jobs()


# -

def eval_command_archery_ME(c, s, elite_id, command_bounds, situation_bounds, kind, verbose=0, erase=True):
    evaluation = eval_archery(unwrap(c, command_bounds), unwrap(s, situation_bounds), verbose=verbose)
    evaluation["command"] = c
    evaluation["situation"] = s
    evaluation["id"] = elite_id
    evaluation["kind"] = kind
    return evaluation


def eval_command_arm_ME(c, s, elite_id, command_bounds, situation_bounds, kind, verbose=0, erase=True):
    r = arm(c, s)
    evaluation = {"reward": r}
    evaluation["command"] = c
    evaluation["situation"] = s
    evaluation["id"] = elite_id
    evaluation["kind"] = kind
    return evaluation


def eval_command_talos_door_opening_ME(c, s, elite_id, command_bounds, situation_bounds, kind, verbose=0, erase=True):
    evaluation = eval_talos_door_opening(unwrap_door_opening_command(c, command_bounds), unwrap_door_opening_situation(s, situation_bounds), verbose=verbose, erase=erase)
    evaluation["command"] = c
    evaluation["situation"] = s
    evaluation["kind"] = kind
    evaluation["id"] = elite_id
    evaluation["reward"] = evaluation["door_angle_after_pulling"]/(np.pi/2)
    return evaluation


# ## Bandit

# ### Vanilla Bandit

class Bandit():
    
    def __init__(self, values):
        self.successes = defaultdict(int)
        self.selected = defaultdict(int)
        self.log = []
        self.values = values
        
    def update(self, key, success):
        self.successes[key] += success
        self.selected[key] += 1 
        n = 0
        for n_value in self.selected.values():
            n += n_value

        if len(self.successes.keys()) < len(self.values):
            k = random.choice(self.values)
        else:
            ucb = []
            for k in self.values:
                n_a = self.selected[k]
                mean = self.successes[k] / n_a
                ucb += [mean +  math.sqrt(2 * math.log(n) / n_a)]
            k = self.values[np.argmax(ucb)]
            
        self.log.append(k)
        return k


# ### Window bandit

class BanditWindow():
    
    def __init__(self, values, T):
        self.successes_dict = defaultdict(int)
        self.successes_deque = deque()
        self.selected = deque()
        self.T = T
        self.log = []
        self.values = values
        
    def update(self, key, success):
        self.successes_dict[key] += success
        self.successes_deque.append(success)
        self.selected.append(key)
        if len(self.selected) > self.T:
            self.successes_dict[self.selected.popleft()] -= self.successes_deque.popleft()
            
        n = len(self.selected)
        if len(set(self.selected)) < len(self.values):
            k = random.choice(self.values)
        else:
            ucb = []
            for k in self.values:
                n_a = self.selected.count(k)
                mean = self.successes_dict[k] / n_a
                ucb += [mean +  math.sqrt(2 * math.log(n) / n_a)]
            k = self.values[np.argmax(ucb)]
            
        self.log.append(k)
        return k


# ## Archive

class Archive():
    
    def __init__(self, config):
        self.situation_config = config["situation_config"]
        self.command_config = config["command_config"]
        self.eval_command = config["eval_command"]
        self.rep = config["rep"]
        
        self.use_regression = False
        self.k_closest2parent = 1
        self.tournament_size = None if "fixed_tournament_size" not in config else config["fixed_tournament_size"]
        self.bandit_values = [1, 5, 10, 50, 100, 500]
        if config["use_bandit"]:
            self.bandit_closest2parent = BanditWindow(self.bandit_values, config["time_window"]) if config["time_window"] is not None else Bandit(self.bandit_values)
        else:
            self.bandit_closest2parent = None
        
        self.log_sampler = []
        self.samples = []
        self.tmp = []
        self.it = 0
        self.n_cells = config["n_cells"]
        self.create_centroids()
        self.tree = cKDTree(self.centroids, leafsize=2)  
        self.elites = [None for _ in range(self.n_cells)] 
        if config["verbose"]>1:
            print("Initialisation")
        if config["parallel"]:
            args = {}
            for i in range(self.n_cells):
                args[i] = (np.random.rand(self.command_config["dim"]), self.centroids[i], i, self.command_config["bounds"], self.situation_config["bounds"], "random")
            jobs = make_general_jobs(self.eval_command, args)
            res = general_master(jobs, n_processes=n_proc, verbose=config["verbose"])
            for key, evaluation in res.items():
                self.samples.append(evaluation)
                self.it += 1 
                self.elites[key] = evaluation
        else:
            for i in range(self.n_cells):
                evaluation = self.eval_command(np.random.rand(self.command_config["dim"]), self.centroids[i], i, self.command_config["bounds"], self.situation_config["bounds"], "random")
                evaluation["it"] = self.it
                self.samples.append(evaluation)
                self.it += 1 
                self.elites[i] = evaluation

    def create_centroids(self):
        if self.n_cells <= 10_000:
            self.centroids = cvt(self.n_cells, self.situation_config["dim"], rep=self.rep)
        else:
            self.centroids = np.random.random((self.n_cells, self.situation_config["dim"]))
        delauney = Delaunay(self.centroids)
        neighbors = [[i] for i in range(self.n_cells)]
        for neighborhood in delauney.simplices:
            for i in neighborhood:
                for j in neighborhood:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        self.centroid_neighbors = [list(set(n)) for n in neighbors]
    
    def add_evaluation(self, evaluation):
        _, index = self.tree.query(evaluation["situation"], k=1)
        evaluation["it"] = self.it 
        self.samples.append(evaluation)
        self.it += 1         
        if config["use_strict_elite"]:
            is_elite = evaluation["reward"] > self.elites[index]["reward"]
        else:
            is_elite = evaluation["reward"] >= self.elites[index]["reward"]
        if is_elite:
            self.elites[index] = evaluation
    
        if "closest2parent" in evaluation["kind"]:
            if self.bandit_closest2parent is not None:
                self.k_closest2parent = self.bandit_closest2parent.update(self.k_closest2parent, is_elite)
            elif self.tournament_size is not None:
                self.k_closest2parent = self.tournament_size
            else:
                self.k_closest2parent = np.random.choice(self.bandit_values)
        return is_elite


# ## Main

def MAP_Elites(config):
    with open(config["logdir"] + f"/config.pk", "wb") as f:
        pickle.dump(config, f)
        
    np.random.seed()
    
    archive = Archive(config)

    init_samples = sampler(n_proc, archive.it, archive, config, verbose=config['verbose']) 
    init_jobs = make_box_jobs(init_samples, config["eval_command"])
       
    if config["parallel"]:
        job_queue = mp.Queue()
        res_queue = mp.Queue()
        pool = mp.Pool(n_proc, worker, (job_queue, res_queue))
        for job in init_jobs:
            job_queue.put(job)
    else:
        job_queue = []
        for job in init_jobs:
            job_queue.append(job)
            
    if config['verbose'] > 0:
        t = tqdm(total=config['budget']-archive.it, ncols=150, smoothing=0.01, mininterval=1, maxinterval=1) 

    t0 = time()   
    for current_it in range(archive.it, config['budget']):
               
        # Collect last answer 
        if config["parallel"]:
            evaluation = res_queue.get()
        else:
            foo, args = job_queue.pop(0)
            evaluation = foo(*args)      
        
        # update loading bar
        if config['verbose'] > 0:
            t.update(1)

        # update archive
        new_elite = archive.add_evaluation(evaluation)
                
        if config["use_logging"]:
            if current_it % int(config['budget']//4) == 0:
                with open(config['logdir']+f"/archive.pk", "wb") as f:
                    pickle.dump(archive, f)
        
        # stop if reached budget
        if current_it == config['budget']-1:
            break 
                   
        # put new job 
        samples = sampler(1, current_it, archive, config, verbose=config['verbose']) 
        mono_job = make_box_jobs(samples, config["eval_command"])
        for job in mono_job:
            if config["parallel"]:
                job_queue.put(job)
            else:
                job_queue.append(job)
                
    if config['use_logging']:
        with open(config['logdir']+f"/archive.pk", "wb") as f:
            pickle.dump(archive, f)
            
    if config['verbose'] > 0:
        t.close()
    
    if config["parallel"]:
        pool.terminate()
        job_queue.close()
        res_queue.close()

    return {"archive": archive, "config": config}


# # Config

config = {
    # MAP-Elite
    "budget": 100_000,
    "verbose": 1,
    "erase": True,
    "parallel": False,
    "n_cells": 200,
    "linreg_sigma": 1.,
    "regression_mutation_scaling": True,
    "continuous": True,
    "proba_regression": 0.5,
    "use_closest2parent": True,
    "use_strict_elite": False,
    "use_bandit": True,
    "fixed_tournament_size": None,
    "time_window": None,
    "variation_operator": "sbx", 
    # Logging
    "use_logging": True,
    "logdir": "",
    "name": "",
}

env = "archery"
if env == "archery":
    config["eval_command"] = eval_command_archery_ME
    config["command_config"] = compute_archery_command(1, archery_action_bounds, line_sigma=0.2, iso_sigma=0.01)
    config["situation_config"] = compute_archery_situation(1, archery_state_bounds)
elif env == "arm":
    config["eval_command"] = eval_command_arm_ME
    config["command_config"] =  {"dim": 10, "bounds": None, "line_sigma": 0.2, "iso_sigma": 0.01}
    config["situation_config"] = {"dim": 2, "bounds": None}
elif env == "talos":
    config["eval_command"] = eval_command_talos_door_opening_ME
    config["command_config"] =  {"dim": 9, "bounds": talos_opening_door_command_bounds, "line_sigma": 0.2, "iso_sigma": 0.01}
    config["situation_config"] = {"dim": 3, "bounds": talos_opening_door_situation_bounds}
env_names = {'arm': '10-D Arm', "archery": "Archery", "talos": "Talos Door Opening"}
env_name = env_names[env]

# # Run

# + active=""
# # For n cells comparison
# parameters = {
#     "n_cells": [10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
# }
# params = grid_search(parameters)

# + active=""
# # For main comparison 
#
# parameters = None
# params = {
#     "MT-ME": {"n_cells": 5_000, "continuous": False, "proba_regression": 0, "use_closest2parent": True},
#     "PT-ME (sbx)": {"n_cells": 200, "continuous": True, "proba_regression": 0., "use_closest2parent": False},
#     "PT-ME (closest2parent sbx)": {"n_cells": 200, "continuous": True, "proba_regression": 0., "use_closest2parent": True},
#     "PT-ME (regression)": {"n_cells": 200, "continuous": True, "proba_regression": 1., "use_closest2parent": False},
#     "PT-ME (50% regression / 50% sbx)": {"n_cells": 200, "continuous": True,  "proba_regression": 0.5, "use_closest2parent": False},
#     "PT-ME (50% regression / 50% closest2parent sbx)": {"n_cells": 200, "continuous": True, "proba_regression": 0.5, "use_closest2parent": True},
# }

# + active=""
# # For bandit comparison
#
# parameters = None
# params = {
#     "No Tournament": {"time_window": None, "use_bandit": True, "use_closest2parent": False},
#     "Random Size": {"time_window": None, "use_bandit": False, "use_closest2parent": True},
#     "Window 100": {"time_window": 100, "use_bandit": True, "use_closest2parent": True},
#     "Window 1k": {"time_window": 1000, "use_bandit": True, "use_closest2parent": True},
#     "Window 10k": {"time_window": 10_000, "use_bandit": True, "use_closest2parent": True},
#     "No Widnow": {"time_window": None, "use_bandit": True, "use_closest2parent": True},
# }

# + active=""
# # For bandit comparison 2
#
# parameters = None
# params = {
#     "No Tournament": {"time_window": None, "use_bandit": True, "use_closest2parent": False},
#     "Random": {"time_window": None, "use_bandit": False, "use_closest2parent": True},
#     "Bandit": {"time_window": None, "use_bandit": True, "use_closest2parent": True},
#     "Fixed 5": {"time_window": None, "use_bandit": False, "use_closest2parent": True, "fixed_tournament_size": 5},
#     "Fixed 10": {"time_window": None, "use_bandit": False, "use_closest2parent": True, "fixed_tournament_size": 10},
#     "Fixed 50": {"time_window": None, "use_bandit": False, "use_closest2parent": True, "fixed_tournament_size": 50},
#     "Fixed 100": {"time_window": None, "use_bandit": False, "use_closest2parent": True, "fixed_tournament_size": 100},
#     "Fixed 500": {"time_window": None, "use_bandit": False, "use_closest2parent": True, "fixed_tournament_size": 500},
# }

# + active=""
# # For Strict or not strict (default)
# parameters = None
# params = {
#     "Strict": {"time_window": None, "use_bandit": True, "use_closest2parent": True, "use_strict_elite": True},
#     "Not Strict": {"time_window": None, "use_bandit": True, "use_closest2parent": True, "use_strict_elite": False},
# }
# -

# SBX vs ISO_DD
parameters = None
params = {
    "SBX": {"variation_operator": "sbx"},
    "ISO_DD": {"variation_operator": "iso_dd"},
}

# + active=""
# # Regression noise (default)
#
# parameters = {
#     "linreg_sigma": [10., 5., 2. , 1., 0.5, 0.2, 0.1, 0.01, 0.001, 0.],
# }
# params = grid_search(parameters)

# + active=""
# # SBX eta
#
# parameters = {
#     "eta": [1e5, 1e4, 1e3, 1e2,1e1,1e0,1e-1,1e-2,1e-3],
# }
# params = grid_search(parameters)

# + active=""
# # Mixe of everything (default)
#
# parameters = {
#     "linreg_sigma": [5., 1., 0.5, 0.1, 0.],
#     "n_cells": [10, 50, 100, 200, 500],
# }
# params = grid_search(parameters)

# + active=""
# # For Talos
# parameters = None
# params = {
#     "PT-ME": {"budget": 100_000, "n_cells": 200,},
# }

# + active=""
# # For Talos
#
# args = {}
# verbose = 1
# idx = 0
# config["logdir"] = "/home/pal/notebooks/data/360d_reflex/datasets/2023/11/12/08h48m38s/"
# while os.path.exists(config["logdir"]+f"/{idx}"):
#     idx += 1
# print(idx)
#
# for name, param in params.items():
#     tmp_config = deepcopy(config)
#     tmp_config["logdir"] = tmp_config["logdir"] + str(idx)
#     tmp_config["verbose"] = verbose 
#     tmp_config["name"] = name
#     tmp_config["parallel"] = env == "talos"
#     tmp_config["rep"] = idx+1
#     if not os.path.exists(tmp_config["logdir"]):
#         os.makedirs(tmp_config["logdir"])
#     for key, val in param.items():
#         if key == "eta":
#             tmp_config["command_config"]["eta"] = val
#         tmp_config[key] = val
#     args[idx] = (tmp_config,)
#     idx += 1
# assert (env != "talos" or len(args) == 1)
# jobs = make_general_jobs(MAP_Elites, args)
# Res = general_master(jobs, n_proc, 1-verbose)

# + active=""
# args = {}
# verbose = 1
# create_logdir(config)
# reps = 20
# idx = 0
# for name, param in params.items():
#     for i in range(reps):
#         tmp_config = deepcopy(config)
#         tmp_config["logdir"] = tmp_config["logdir"] + str(idx)
#         tmp_config["verbose"] = verbose 
#         tmp_config["name"] = name
#         tmp_config["parallel"] = env == "talos"
#         tmp_config["rep"] = i+1
#         if not os.path.exists(tmp_config["logdir"]):
#             os.makedirs(tmp_config["logdir"])
#         for key, val in param.items():
#             if key == "eta":
#                 tmp_config["command_config"]["eta"] = val
#             tmp_config[key] = val
#         args[idx] = (tmp_config,)
#         idx += 1
# assert (env != "talos" or len(args) == 1)
# jobs = make_general_jobs(MAP_Elites, args)
# Res = general_master(jobs, n_proc, 1-verbose)

# + active=""
# path = "/home/pal/notebooks/data/360d_reflex/datasets/2023/12/06/09h59m14s/"
# Res = {}
# for folder in tqdm(os.listdir(path)):
#     with open(path+folder+"/archive.pk", "rb") as f:
#         Res[int(folder)] = {"archive": pickle.load(f)}
#     with open(path+folder+"/config.pk", "rb") as f:
#         Res[int(folder)]["config"] = pickle.load(f)

# +
# main comparison
path = "/home/pal/notebooks/data/360d_reflex/datasets/2023/11/02/10h08m25s/"  # Archery x20 
path = "/home/pal/notebooks/data/360d_reflex/datasets/2023/11/02/10h38m58s/"  # Arm x20
# n_cells for PT-RME
path = "/home/pal/notebooks/data/360d_reflex/datasets/2023/11/06/09h28m30s/"  # Arm x20
path = "/home/pal/notebooks/data/360d_reflex/datasets/2023/11/06/13h15m31s/"  # Archery x20
# main comparison with better hyperparameters-tuning 
path = "/home/pal/notebooks/data/360d_reflex/datasets/2023/11/08/11h38m33s/"  # Arm x20
path = "/home/pal/notebooks/data/360d_reflex/datasets/2023/11/08/14h39m15s/"  # Archery x20

# Talos 
#path = "/home/pal/notebooks/data/360d_reflex/datasets/2023/11/12/08h48m38s/"
# -

env = "talos"
if env == "arm":
    path = "/home/pal/notebooks/data/PT-ME/datasets/PT-ME/arm/"  # Arm x20
elif env == "archery":
    path = "/home/pal/notebooks/data/PT-ME/datasets/PT-ME/archery/"  # Archery x20
elif env == "talos":
    path = "/home/pal/notebooks/data/PT-ME/datasets/PT-ME/talos/"
env_name = env_names[env]

Res = {}
for folder in tqdm(os.listdir(path)):
    with open(path+folder+"/archive.pk", "rb") as f:
        Res[int(folder)] = {"archive": pickle.load(f)}
    with open(path+folder+"/config.pk", "rb") as f:
        Res[int(folder)]["config"] = pickle.load(f)

# + active=""
# save_pickle("/home/pal/notebooks/data/tmp/centroids.pk", res["archive"].centroids)
# -

# # Plot

if parameters is None:
    keys = defaultdict(list)
    for key, res in Res.items():
        keys[res["config"]["name"].split("_")[0]].append(key)
    names = keys
    w, h = len(keys), 1
else:
    params_keys = [key for key, vals in parameters.items() if len(vals)>1]
    params_vals = [vals for key, vals in parameters.items() if len(vals)>1]
    keys = defaultdict(list)
    for key, res in Res.items():
        keys[tuple(res["config"][param] for param in params_keys)].append(key)
    names = {vals : keys[vals] for vals in product( *tuple(params_vals) )}  
    if len(params_vals) == 2:
        w, h = (len(p) for p in params_vals)
    elif len(params_vals) == 1:
        w, h = len(params_vals[0]), 1    
    elif len(params_vals) == 0:
        w, h = len(Res), 1
        keys = {key: key for key, res in Res.items() }
        params_keys = [i for i in range(w)]
        names = {(i,) : i for i in range(len(Res))}  

R = 

np.mean([np.max([x["reward"] for x in Res[i]["archive"].samples]) for i in range(10)])

# ## QD-Score

arm_min = open_pickle("/home/pal/notebooks/data/PT-ME/arm_minimum.pk")
arm_max = open_pickle("/home/pal/notebooks/data/PT-ME/arm_maximum.pk")


def compute_cell_min_max(step_size):
    cells_min = defaultdict(list)
    cells_max = defaultdict(list)
    for key in arm_min.keys():
        ckey = tuple([int(x) for x in np.array(arm_min[key]["s"])/step_size])
        cells_min[ckey].append(arm_min[key]["r"])
        cells_max[ckey].append(arm_max[key]["r"])
    for key in cells_min.keys():
        cells_min[key] = np.min(cells_min[key])
        cells_max[key] = np.max(cells_max[key])
    return cells_min, cells_max


def compute_qd_score(res, resolutions):
    qd_score = {}
    for step_size in resolutions:
        cells = defaultdict(list)
        if env in ["talos", "arm"]:
            cells_min, cells_max = cells_min_max[step_size]
        for i in range(len(res["archive"].samples)):
            key = tuple([int(x) for x in res["archive"].samples[i]["situation"]/step_size])
            cells[key].append(res["archive"].samples[i]["reward"])
        for key in cells.keys():
            if env == "arm":
                cells[key] = (np.max(cells[key])-cells_min[key])/(cells_max[key]-cells_min[key])
            elif env == "talos":
                cells[key] = (np.max(cells[key])-cells_min)/(cells_max-cells_min)
            else:
                cells[key] = np.max(cells[key])
        qd_score[step_size] = sum((x for x in cells.values())) * step_size**len(key)
    return qd_score


# +
env_dim = 3 if env == "talos" else 2

resolutions = []
for x in [1/int(x) for x in np.logspace(0, 5/env_dim, 50)]:
    if x not in resolutions:
        resolutions.append(x)

# +
if env == "arm":
    cells_min_max = {}
    for x in resolutions:
        cells_min_max[x] = compute_cell_min_max(x)
elif env == "talos": 
    cells_min_max = {}
    for x in resolutions:
        cells_min_max[x] = (0, 1.2840939118540298)

batch_size = 50
Args = {}
for j in range(int(np.ceil(len(Res)/batch_size))):
    args = {}
    for i in range(j*batch_size, min((j+1)*batch_size, len(Res))):
        args[i] = (Res[i], resolutions)
    Args[j] = args 
    
QD_scores = {}
for j, args in Args.items():
    jobs = make_general_jobs(compute_qd_score, args)
    QD_scores[j] = general_master(jobs, n_proc, 1)
# -

for key, res in tqdm(Res.items()):
    j = 0
    while key not in QD_scores[j]:
        j +=1
    res["density_coverage"] = [QD_scores[j][key][N] for N in resolutions]

qd_score_PT_ME = {}
for j, (name, indices) in enumerate(names.items()):
    qd_score_PT_ME[name] = np.array([Res[i]["density_coverage"] for i in indices]).transpose()

# + active=""
# with open(f"/home/pal/notebooks/data/PT-ME/evaluations/PT-ME_{env}_qd_score.pk", "wb") as f: 
#     pickle.dump(qd_score_PT_ME, f)
# -

# ### no baseline plot

# +
data, labels, X = [], [], []
line_styles = []

for j, (name, qd_score) in enumerate(qd_score_PT_ME.items()):
    data.append(qd_score)
    line_styles.append(linestyles[j%len(linestyles)])
    labels.append(name)
    X.append(1/np.array(resolutions)**2)

plt.subplots(figsize=(16*2,9))
plt.subplot2grid((1,2), (0,0))
plot_with_spread(data, labels, X, cm.tab10, line_styles=line_styles)
#plt.xlim((1e3, 1e5))
#plt.ylim((0.6,0.8))
plt.legend(bbox_to_anchor=(1,1,0,0))
plt.xscale("log")
plt.xlabel("Resolution")
plt.ylabel(f"QD-Score")
plt.title(f"QD-Score {env_name}")
#savefig("comparison_arm")
# -

datas = [np.mean(X, axis=0) for X in data]
plot_boxplot(datas, labels, ylabel="Mean QD-Score", rotation=90, swarmsize=7, figsize=(16,9), use_table=True,)

for i, name in enumerate(labels):
    X = datas[i]
    print(name, f"{np.mean(X):1.4f}+/-{np.std(X):1.4f} median:{np.median(X):1.4f} [25%, 75%]=[{np.quantile(X, 0.25):1.4f}, {np.quantile(X, 0.75):1.4f}] [5%, 95%]=[{np.quantile(X, 0.05):1.4f}, {np.quantile(X, 0.95):1.4f}]")

# ### Baselines

env = "archery"
env_names = {'arm': '10-D Arm', "archery": "Archery", "talos": "Talos Door Opening"}
env_name = env_names[env]

# +
env_dim = 3 if env == "talos" else 2

resolutions = []
for x in [1/int(x) for x in np.logspace(0, 5/env_dim, 50)]:
    if x not in resolutions:
        resolutions.append(x)
# -

with open(f"/home/pal/notebooks/data/PT-ME/evaluations/PT-ME_{env}_qd_score.pk", "rb") as f: 
    qd_score_PT_ME = pickle.load(f)

# +
if env == "talos":
    pass
else:
    with open(f"/home/pal/notebooks/data/PT-ME/evaluations/cma_es_{env}_qd_score.pk", "rb") as f: 
        cma_es_qdscore = pickle.load(f)

    cma_data = np.array([list(cma_es_qdscore[(i, np.inf)].values()) for i in range(20)]).transpose()
    cma_X = 1/np.array(list(cma_es_qdscore[(0, np.inf)].keys()))**2
    cma_label = "CMA-ES (no limit)"

    cma2_data = np.array([list(cma_es_qdscore[(i, 10)].values()) for i in range(20)]).transpose()
    cma2_X = 1/np.array(list(cma_es_qdscore[(0, 10)].keys()))**2
    cma2_label = "CMA-ES (max 10 iterations)"

    with open(f"/home/pal/notebooks/data/PT-ME/evaluations/250_random_{env}_100k_qd_score.pk", "rb") as f: 
        random_qdscore = pickle.load(f)

    random_data = np.array([list(qd_score.values()) for qd_score in random_qdscore.values()]).transpose()
    random_X = 1/np.array(list(random_qdscore[0].keys()))**2
    random_label = "Random"

if env == "talos":
    with open(f"/home/pal/notebooks/data/PT-ME/evaluations/10_ppo_{env}_100k_qd_score.pk", "rb") as f: 
        ppo_qdscore = pickle.load(f)
else:
    with open(f"/home/pal/notebooks/data/PT-ME/evaluations/20_ppo_{env}_100k_qd_score.pk", "rb") as f: 
        ppo_qdscore = pickle.load(f)
    
ppo_data = np.array([list(qd_score.values()) for qd_score in ppo_qdscore.values()]).transpose()
ppo_X = 1/np.array(list(ppo_qdscore[0].keys()))**2
ppo_label = "PPO"
# -

# ### Plot 

# +
if env == "talos":
    data, labels, X = [ppo_data], [ppo_label], [ppo_X]
    line_styles = ["-"]
else:
    data, labels, X = [random_data, ppo_data, cma_data, cma2_data], [random_label, ppo_label, cma_label, cma2_label], [random_X, ppo_X, cma_X, cma2_X]
    line_styles = ["-", "--", ":", "-."]

for j, (name, qd_score) in enumerate(qd_score_PT_ME.items()):
    if True: #name in ["MT-ME", "PT-ME (sbx)", "PT-ME (50% sbx)", "PT-ME (Oracle sbx)"]:
        data.append(qd_score)
        line_styles.append(linestyles[j%len(linestyles)])
        labels.append(name)
        X.append(1/np.array(resolutions)**2)

plt.subplots(figsize=(16*2,9))
plt.subplot2grid((1,2), (0,0))
plot_with_spread(data, labels, X, cm.tab10, line_styles=line_styles)
#plt.xlim((1e3, 1e5))
#plt.ylim((0.6,0.8))
plt.legend(bbox_to_anchor=(1,1,0,0))
plt.xscale("log")
plt.xlabel("Resolution")
plt.ylabel(f"QD-Score")
plt.title(f"QD-Score {env_name}")

if False:
    plt.subplot2grid((1,2), (0,1))
    plot_with_spread(data, labels, X, cm.gist_rainbow, line_styles=line_styles)
    plt.xlim((1e4, 1e5))
    plt.ylim((0.4, 1.))
    plt.legend(handles=[])
    plt.xscale("log")
    plt.xlabel("Resolution")
    plt.ylabel(f"QD-Score")
    plt.title(f"QD-Score {env_name}")

#savefig("comparison_arm")
# -

# ### Plot Talos

c = 1.2840939118540298 * np.pi/2  # to put back into rad 

# +
X = [1/np.array(resolutions)**3, 1/np.array(resolutions)**3]
data = [qd_score_PT_ME["PT-ME"]*c, ppo_data*c]
names = ["PT-R-ME (ours)", "PPO"]
line_styles = ["-", (0, (1, 1))]
colors = [blue, Teal]

plt.subplots(figsize=(8,5))
plot_with_spread(data, names, X, colors=colors, line_styles=line_styles)
plt.legend(bbox_to_anchor=(1,1,0,0))
plt.xscale("log")
plt.xlabel("Resolution")
plt.ylabel(f"QD-Score (rad)")
plt.title(f"QD-Score {env_name}")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
#plt.ylim((0,1))
#savefig("comparison_talos")
# -

data = [[np.mean(x)*c for x in qd_score_PT_ME["PT-ME"]], [np.mean(x)*c for x in ppo_data]]
names = ["PT-R-ME (ours)", "PPO"]
plot_boxplot(data, names, ylabel="Mean QD-Score", colors=colors, rotation=0, swarmsize=4, figsize=(4,5), use_table=False, use_stick=True)
plt.title(env_name)
#savefig(f"boxplots_{env_name}")

for i, d in enumerate(data):
    X = np.array(d) 
    print(names[i], f"{np.mean(X):1.3f}+/-{np.std(X):1.3f} median:{np.median(X):1.3f} [25%, 75%]=[{np.quantile(X, 0.25):1.3f}, {np.quantile(X, 0.75):1.3f}] [5%, 95%]=[{np.quantile(X, 0.05):1.3f}, {np.quantile(X, 0.95):1.3f}]")

# ### Plot main comparison qd-score vs resolutions Arm & Archery

# +
plot_dict = {
    "PPO": {"qd_score": ppo_data, "ls": (0, (1, 1)), "color": Teal},
    
    'PT-RME': {"qd_score": qd_score_PT_ME['PT-ME (50% regression / 50% closest2parent sbx)'], "ls": "-", "color": blue},
    'PT-RME (no tournament)': {"qd_score": qd_score_PT_ME['PT-ME (50% regression / 50% sbx)'], "ls":  (0, (1, 10)), "color": LightBlue},
    'PT-RME (100% regression)': {"qd_score": qd_score_PT_ME['PT-ME (regression)'], "ls": (0, (3, 5, 1, 5, 1, 5)), "color": Brown},
    
    'PT-ME': {"qd_score": qd_score_PT_ME['PT-ME (closest2parent sbx)'], "ls": (0, (3, 1, 1, 1, 1, 1)), "color": Orange },
    'PT-ME (no tournament)': {"qd_score": qd_score_PT_ME['PT-ME (sbx)'], "ls": (0, (5, 10)), "color": LightGreen},
    
    'MT-ME': {"qd_score": qd_score_PT_ME['MT-ME'], "ls": (0, (3, 1, 1, 1)), "color": Yellow},
    
    "CMA-ES": {"qd_score": cma_data, "ls": (0, (5, 1)) , "color": Purple},
    "CMA-Es (max 10 iter)": {"qd_score": cma2_data, "ls": (0, (5, 5)), "color": Pink},
    
    "Random": {"qd_score": random_data, "ls": (0, (5, 5)), "color": Gray},
}

plt.subplots(figsize=(16*2,9))

data = [dic["qd_score"] for dic in plot_dict.values()]
labels = list(plot_dict.keys())
X = [1/np.array(resolutions)**2] * len(data)
colors = [dic["color"] for dic in plot_dict.values()]
line_styles = [dic["ls"] for dic in plot_dict.values()]
plot_with_spread(data, labels, X, colors, line_styles=line_styles, lw=3, alpha=0.1)

plt.legend(bbox_to_anchor=(1.1,0,0,0), numpoints=1, ncol=5, fontsize=24)
plt.xscale("log")
plt.xlabel("Resolution")
plt.ylabel(f"QD-Score")
plt.title(f"QD-Score {env_name}")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
#savefig(f"plot_with_spread_{env_name}")
# -

# ### Statistical Comparison

labels = {
    'PPO':'PPO',
    'PT-RME': 'PT-RME',
    'PT-RME\n(no trnmnt)': 'PT-RME (no tournament)',
    'PT-RME\n(reg)': 'PT-RME (100% regression)',
    'PT-ME': 'PT-ME',
    'PT-ME\n(no trnmnt)': 'PT-ME (no tournament)',
   # 'MT-ME': 'MT-ME',
   # "Random": "Random",
    #"CMA-ES\n(max 10 iter)": "CMA-ES (max 10 iter)",
    #"CMA-ES": "CMA-ES",
}

data = [np.mean(plot_dict[key]["qd_score"], axis=0) for key in labels.values()]
plot_boxplot(data, list(labels.keys()), ylabel="Mean QD-Score", colors=[plot_dict[key]["color"] for key in labels.values()], rotation=0, swarmsize=3, figsize=(16,9), use_table=True)
plt.title(env_name)
#savefig(f"boxplots_{env_name}")

for name, dic in plot_dict.items():
    X = np.mean(dic["qd_score"], axis=0)
    print(name, f"{np.mean(X):1.3f}+/-{np.std(X):1.3f} median:{np.median(X):1.3f} [25%, 75%]=[{np.quantile(X, 0.25):1.3f}, {np.quantile(X, 0.75):1.3f}] [5%, 95%]=[{np.quantile(X, 0.05):1.3f}, {np.quantile(X, 0.95):1.3f}]")

# ## Bandit analysis

Counts = {}
for name, indices in names.items():
    if name not in ['No Tournament', 'Random Size']:
        counts = defaultdict(list)
        for i in indices:
            bandit = Res[i]["archive"].bandit_closest2parent
            count = Counter(bandit.log)
            n = np.sum(list(count.values()))
            for key, val in count.items():
                counts[key].append(100 * val/n)
        Counts[name] = counts

for name, counts in Counts.items():
    print(name)
    keys = list(counts.keys())
    keys.sort()
    for key in keys:
        print(f"\t{key}: {np.mean(counts[key]):2.1f}%")

# +
bandit = Res[17]["archive"].bandit_closest2parent
step = 1000
counts = {key: [] for key in bandit.values}
for t in np.arange(0, len(bandit.log)+step, step):
    k = step//2
    X = bandit.log[max(0, int(t)-k):int(t)+k]
    count = Counter(X)
    for key in counts:
        if key in count:
            counts[key].append(100*count[key]/(len(X)))
        else:
            counts[key].append(0)
            
for key, val in counts.items():
    plt.plot(val, label=key)
plt.legend()
# -

# ## Solutions Distributions

# ### Points

w = len(names)
h = np.max([len(idx) for idx in names.values()])

for i, (name, indices) in enumerate(names.items()):
    for j, idx in enumerate(indices[2:3]): 
        res = Res[idx]
        rewards = np.array([ev["reward"] for ev in res["archive"].samples])
        print(np.mean(rewards))

if env == "talos": 
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (name, indices) in enumerate(names.items()):
        for j, idx in enumerate(indices[0:1]): 
            res = Res[idx]

            situation = np.array([ev["situation"] for ev in res["archive"].samples])
            rewards = np.array([ev["reward"] for ev in res["archive"].samples])
            indices = np.argsort(rewards)

            for j in range(3):
                x, y = [(0,1), (0,2), (1,2)][j]
                ax = plt.subplot2grid((w, 3), (i, j))
                plt.scatter(situation[indices, x], situation[indices, y], s=1, c=rewards[indices], vmin=0, vmax=1.2840939118540298)
                plt.axis("off")
                plt.xlim((0,1.))  # (0.42,0.58))
                plt.ylim((0,1.))
                plt.title((["x", "y", "rz"][x], ["x", "y", "rz"][y]))
                plt.gca().set_aspect('equal')
            #plt.colorbar()
else:
    fig, ax = plt.subplots(figsize=(6*w, 6*h))
    for i, (name, indices) in enumerate(names.items()):
        for j, idx in enumerate(indices[:1]): 
            res = Res[idx]
            ax = plt.subplot2grid((h, w), (j, i))
            situation = np.array([ev["situation"] for ev in res["archive"].samples])
            rewards = np.array([ev["reward"] for ev in res["archive"].samples])
            plt.scatter(situation[:,0], situation[:,1], s=1, c=rewards, vmin=0, vmax=1)

            if  j == 0:
                plt.title(name, fontsize=10)
            if  i == 0:
                txt = "" 
                x = -0.11
                y = 0.5 - len(str(txt)) * 0.025
                plt.text(x, y, txt, rotation=90, horizontalalignment="center")
            #plt.text(0.5, 0.5, f"{res['evaluation']:2.1f}%", fontsize= 14, horizontalalignment="center")
            plt.axis("off")
            plt.xlim((0,1.))  # (0.42,0.58))
            plt.ylim((0,1.))  #(0.49,0.58))
#savefig("talos_ours")

# +
T = [0, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
w, h = len(Res), len(T)-1

fig, ax = plt.subplots(figsize=(4*w, 4*h))
for j in range(len(T)-1):
    for i, (name, indices) in enumerate(names.items()):
        res = Res[indices[1]]
        ax = plt.subplot2grid((h, w), (j, i))
        situation = np.array([ev["situation"] for ev in res["archive"].samples if ev["it"] < T[j+1] and ev["kind"] != "regression" and ev["reward"] >= (-0.2 if env == "arm" else 10)])
        if len(situation)>0:
            plt.scatter(situation[:,0], situation[:,1], s=2, color="tab:blue")#, color=cmap.to_rgba(res["evaluation"]))
        situation = np.array([ev["situation"] for ev in res["archive"].samples if ev["it"] < T[j+1] and ev["kind"] == "regression" and ev["reward"] >= (-0.2 if env == "arm" else 10)])
        if len(situation)>0:
            plt.scatter(situation[:,0], situation[:,1], s=1, color="tab:orange")#, color=cmap.to_rgba(res["evaluation"]))
        
        if (j) == 0:
            plt.title(name, fontsize=10)
        if (i) == 0:
            txt = f"{T[j+1]}"
            x = -0.11
            y = 0.5 - len(str(txt)) * 0.025
            plt.text(x, y, txt, rotation=90, horizontalalignment="center")
        #plt.text(0.5, 0.5, f"{res['evaluation']:2.1f}%", fontsize= 14, horizontalalignment="center")
        plt.axis("off")
        plt.xlim((0.,1.))  # (0.42,0.58))
        plt.ylim((0.,1.))  #(0.49,0.58))
# -

w, h = len(Res), len(T)-1
fig, ax = plt.subplots(figsize=(4*w, 4*h))
for j in range(len(T)-1):
    for i, (name, indices) in enumerate(names.items()):
        res = Res[indices[0]]
        ax = plt.subplot2grid((h, w), (j, i))
        situation = np.array([ev["command"] for ev in res["archive"].samples if ev["it"] < T[j+1] and  ev["kind"] != "regression"])
        if len(situation)>0:
            plt.scatter(situation[:,0], situation[:,1], s=2, color="tab:blue")#, color=cmap.to_rgba(res["evaluation"]))
        situation = np.array([ev["command"] for ev in res["archive"].samples if ev["it"] < T[j+1] and ev["kind"] == "regression"])
        if len(situation)>0:
            plt.scatter(situation[:,0], situation[:,1], s=1, color="tab:orange")#, color=cmap.to_rgba(res["evaluation"]))
        
        if (j) == 0:
            plt.title(name, fontsize=10)
        if (i) == 0:
            txt = f"{T[j+1]}"
            x =0.41
            y =0.
            #plt.text(x, y, txt, rotation=90, horizontalalignment="center")
        #plt.text(0.5, 0.5, f"{res['evaluation']:2.1f}%", fontsize= 14, horizontalalignment="center")
        plt.axis("off")
        #plt.xlim((0.42,0.58))
        #plt.ylim((0.49,0.58))

# ### Selected Tasks 

# +
T = [0, 100, 200 ,500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
w, h = len(Res), len(T)-1

fig, ax = plt.subplots(figsize=(4*w, 4*h))
for j in range(len(T)-1):
    for i, (name, indices) in enumerate(names.items()):
        res = Res[indices[0]]
        ax = plt.subplot2grid((h, w), (j, i))
        situation = np.array(res["archive"].task_log)[T[j]:T[j+1]]
        X = res["archive"].log_sampler[T[j]:T[j+1]]
        colors_choices = {'regression': "tab:blue", 'iso_dd': "tab:orange", 'sbx': "tab:red", 'closest2parent': "tab:purple"}
        colors = [colors_choices[sorted(dic.items(), key=lambda x : x[1])[-1][0]] for dic in X]

        #color = color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"][i%6]
        plt.scatter(situation[:,0], situation[:,1], s=1, color=colors)
        
        if (j) == 0:
            plt.title(name, fontsize=10)
        if (i) == 0:
            txt = f"{T[j]}-{T[j+1]}"
            x = -0.11
            y = 0.5 - len(str(txt)) * 0.025
            plt.text(x, y, txt, rotation=90, horizontalalignment="center")
        plt.axis("off")
        plt.xlim((0.,1.)) 
        plt.ylim((0.,1.)) 

# -

# ### selected operator per cells

centroids = cvt(100, 2, 100)
centroids = centroids * (1.2+0.2) - 0.2
points = np.array(centroids)
vor = Voronoi(points)
good = [np.where(vor.point_region == i)[0][0] for i, region in enumerate(vor.regions) if region and -1 not in region]
tree = cKDTree(centroids, leafsize=2)  

# +
T = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 5000, 10_000, 20_000, 50_000, 70000, 80_000, 85000, 90_000, 95_000, 100_000]

Colors = []
for j in range(len(T)-1):
    counts = [[] for _ in range(len(centroids))]

    res = Res[names["PT-ME (Oracle sbx)"][0]]
    shift = len(res["archive"].samples) - len(res["archive"].log_sampler)
    situations = np.array([ev["situation"] for ev in res["archive"].samples[shift+T[j]:shift+T[j+1]]])
    _, indexes = tree.query(situations)
    X = res["archive"].log_sampler[T[j]:T[j+1]]
    for i in range(min(len(X), len(indexes))):
        counts[indexes[i]].append(sorted(X[i].items(), key=lambda x : x[1])[-1][0]) 
    colors = [sorted(Counter(count).items(), key=lambda x : x[1])[-1][0] if count else None for count in counts]
    Colors.append(colors)

# +
plt.subplots(figsize=(4*5, 4*4))

res = Res[1]
vop_colors = {'regression': "tab:blue", 'iso_dd': "tab:orange", 'sbx': "tab:red", 'closest2parent': "tab:purple", None: "black"}

for k, colors in enumerate(Colors):
    ax = plt.subplot2grid((4,5), (k//5, k%5))
    for i, region in enumerate(vor.regions):
        if region and -1 not in region:
            ind = np.where(vor.point_region == i)[0][0]
            color = vop_colors[colors[ind]]
            alpha = 0 if colors[ind] is None else 1.
            ax.fill(vor.vertices[region,0], vor.vertices[region,1], c=color, alpha=alpha)
    ax.set_xlim((-0.0,1.))
    ax.set_ylim((-0.0,1.))   
    ax.set_title(f"{T[k]}-{T[k+1]}")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([]);
    
handles = []
for key, color in vop_colors.items():
    if key:
        handles.append(Patch(facecolor=color, edgecolor='white', label=key))
plt.legend(handles=handles, bbox_to_anchor=(1,1,0,0))

#plt.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), label="Density")
# -

# ### Voronoi

centroids = cvt(100, 2, 100)
centroids = centroids * (1.2+0.2) - 0.2
points = np.array(centroids)
vor = Voronoi(points)
good = [np.where(vor.point_region == i)[0][0] for i, region in enumerate(vor.regions) if region and -1 not in region]

tri = Delaunay(points)

tree = cKDTree(centroids, leafsize=2)  

centroids_counts = sample_Sobol(1_000_000, dim=2)
sobol_counts = np.zeros(len(centroids))
dist, indexes = tree.query(centroids_counts, 1)
for ind in indexes:
    sobol_counts[ind] += 1

for res in Res.values():
    solutions = [ev["situation"] for ev in res["archive"].solutions]
    n_solutions_per_cell = np.zeros(len(centroids))
    dist, indexes = tree.query(solutions, 1)
    for ind in indexes:
        n_solutions_per_cell[ind] += 1
    n_solutions = n_solutions_per_cell
    res["ratio"] = np.array([(n_solutions[i]*np.sum(sobol_counts))/(sobol_counts[i]*np.sum(n_solutions)) if sobol_counts[i]>0 else 0 for i in range(len(n_solutions))])

# +
plt.subplots(figsize=(5*5, 4*5))

for k, (name, key) in enumerate(names.items()):

    res = Res[1]
    vmin = np.min(res['ratio'][np.where(res["ratio"]>0) and good])
    vmax = np.max(res['ratio'][np.where(res["ratio"]>0) and good])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

    ax = plt.subplot2grid((5,5), (k%5, k//5))
    for i, region in enumerate(vor.regions):
        if region and -1 not in region:
            ind = np.where(vor.point_region == i)[0][0]
            ax.fill(vor.vertices[region,0], vor.vertices[region,1], c=cmap.to_rgba(res["ratio"][ind]) )
            x, y = points[ind ][0], points[ind ][1]
            
            ax.text(x, y, ind, fontsize=8)
    ax.set_xlim((-0.0,1.))
    ax.set_ylim((-0.0,1.))   
    ax.set_title(name)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([]);
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), label="Density")
# -

delauney = Delaunay(points)
neighbors = [[] for _ in range(len(centroids))]
for i, j, k, l in delauney.simplices:
    neighbors[i].append(j)
    neighbors[i].append(k)
    neighbors[j].append(i)
    neighbors[j].append(k)
    neighbors[k].append(i)
    neighbors[k].append(j)
neighbors = [list(set(n)) for n in neighbors]

# # Generalisation evaluation

# ## Load

env = "talos"

if env == "arm":
    if os.uname()[1] == "evo256":
        path = "/home/pal/notebooks/data/360d_reflex/datasets/2023/11/08/11h38m33s/"  # Arm x20
    else:
        path = "/home/pal/notebooks/data/PT-ME/datasets/PT-ME/arm/"  # Arm x20
elif env == "archery":
    if os.uname()[1] == "evo256":
        path = "/home/pal/notebooks/data/360d_reflex/datasets/2023/11/08/14h39m15s/"  # Archery x20
    else:
        path = "/home/pal/notebooks/data/PT-ME/datasets/PT-ME/archery/"  # Archery x20
elif env == "talos":
    if os.uname()[1] == "evo256":
        path = "/home/pal/notebooks/data/PT-ME/datasets/PT-ME/talos/"
    else:
        path = "/home/pal/notebooks/data/PT-ME/datasets/PT-ME/talos/"
env_name = env_names[env]

Res = {}
for folder in tqdm(os.listdir(path)):
    with open(path+folder+"/archive.pk", "rb") as f:
        Res[int(folder)] = {"archive": pickle.load(f)}
    with open(path+folder+"/config.pk", "rb") as f:
        Res[int(folder)]["config"] = pickle.load(f)

keys = defaultdict(list)
for key, res in Res.items():
    keys[res["config"]["name"].split("_")[0]].append(key)
names = keys

if env == "talos":
    archives = [{"archive": res["archive"]} for res in Res.values()]
else:
    archives = [{"archive": Res[i]["archive"]} for i in names["PT-ME (50% regression / 50% closest2parent sbx)"]]

# ## Select evaluation samples

if env == "talos":
    n = 1000
    S = cvt(n, 3, rep=0)     
else:
    n = 10_000
    S = cvt(n, 2, rep=0) 

for archive in tqdm(archives):
    A = []
    for s in S:
        _, idx = archive["archive"].tree.query(s, k=1)
        indexes = archive["archive"].centroid_neighbors[idx]  # find the direct neighbors using the precomputed delauney from the centroids 
        X = [archive["archive"].elites[i]["situation"] for i in indexes]
        Y = [archive["archive"].elites[i]["command"] for i in indexes]
        reg = LinearRegression().fit(X, Y)
        c = reg.predict(np.array([s]))[0] 
        dim = len(c)
        A.append(np.clip(c, np.zeros(dim), np.ones(dim)))
    archive["A"] = A


# ## evaluate

def eval_talos_door_opening_batch(C, S, verbose=0, erase=True ):
    """ C must be a disct with keys as int corresponding to S indices """
    S = [unwrap_door_opening_situation(s, talos_opening_door_situation_bounds) for s in S]
    C = {key: [unwrap_door_opening_command(c, talos_opening_door_command_bounds) for c in C[key]] for key in C}
    batch = generate_param_talos_door_opening(S, C, verbose=verbose)
    evaluations = evaluate_batch_talos_door_opening(batch, erase= erase and verbose != 73, verbose=verbose)
    Rs = {}
    for key, ev in evaluations.items():
        s_id, archive_id = [int(x) for x in key.split("_")]
        if archive_id not in Rs:
            Rs[archive_id] = np.empty(len(S))
        Rs[archive_id][s_id] = ev["door_angle_after_pulling"]/(np.pi/2)
    return Rs


if env == "arm":
    eval_f = lambda c, s : arm(c, s)
elif env == "archery":
    command_bounds = compute_archery_command(1, archery_action_bounds, line_sigma=0.2, iso_sigma=0.01)["bounds"]
    situation_bounds = compute_archery_situation(1, archery_state_bounds)["bounds"]
    eval_f = lambda c, s : eval_archery(unwrap(c, command_bounds), unwrap(s, situation_bounds), verbose=0)["reward"]
elif env == "talos":
    eval_f = None

arm_min = open_pickle("/home/pal/notebooks/data/PT-ME/arm_min_10k.pk")
arm_max = open_pickle("/home/pal/notebooks/data/PT-ME/arm_max_10k.pk")

S_min_max = []
for i, s in enumerate(S):
    if env == "arm":
        S_min_max.append({"min": arm_min[i]["r"], "max": arm_max[i]["r"]})
    else:
        S_min_max.append({"min": 0, "max": 1})

if env == "talos":
    C = {i: [archive["A"][i] for archive in archives] for i in range(len(S))}
    evaluations = eval_talos_door_opening_batch(C, S, verbose=1)
    for i, archive in enumerate(archives):
        archive["R"] = evaluations[i]
else:
    for archive in tqdm(archives):
        R = []
        for i in range(len(S)):
            c = archive["A"][i]
            s = S[i]
            R.append((eval_f(c,s)-S_min_max[i]["min"])/(S_min_max[i]["max"]-S_min_max[i]["min"]))
        archive["R"] = R

# ## Plot

QD_score = [np.mean(archive["R"]) for archive in archives]

f"QD-score: {np.median(QD_score):1.3f} [{np.quantile(QD_score, 0.25):1.3f}, {np.quantile(QD_score, 0.75):1.3f}]"

to_save =  [{"R": archive["R"], "A": archive["A"], "S": S} for archive in archives]
save_pickle("/home/pal/notebooks/data/PT-ME/inference/PT-R-ME/talos.pk", to_save)



env = "talos"
ppo = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/PPO/{env}.pk")
ours = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/PT-R-ME/{env}.pk")
distil = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/NN/{env}.pk")

if env == "talos":
    n = 1000
    S = cvt(n, 3, rep=0) 
else:
    n = 10_000
    S = cvt(n, 2, rep=0) 

if env == "talos":
    w, h = 10, 3
    plt.subplots(figsize=(2*w, 2*h))
    for i, archive in enumerate(distil):
        for k in range(3):
            (x, y) = [(0,1), (0,2), (1,2)][k]
            ax = plt.subplot2grid((h, w), (k, i))
            index = np.argsort(archive["R"])
            plt.scatter(S[index, x], S[index, y], c=archive["R"][index], s=50, vmin=0, vmax=1.2840939118540298 * np.pi/2)
            plt.axis("off")
            plt.xlim((0,1.))  
            plt.ylim((0,1.)) 
            ax.set_aspect('equal')
            plt.tight_layout()
else:
    w, h = 10, 2
    plt.subplots(figsize=(2*w, 2*h))
    for i, archive in enumerate(distil):
        ax = plt.subplot2grid((h, w), (i//w, i%w))
        plt.scatter(S[:, 0], S[:, 1], c=archive["R"], vmin=0, vmax=1)
        #plt.title(f"{env}")
        plt.axis("off")
        plt.xlim((0,1.))  
        plt.ylim((0,1.)) 
        ax.set_aspect('equal')
        #plt.colorbar()
        plt.tight_layout()

# ### Comparison

env = "talos"
ppo = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/PPO/{env}.pk")
ours = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/PT-R-ME/{env}.pk")
disti = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/NN/{env}.pk")

c = np.pi/2



# +

fig, ax = plt.subplots(figsize=(16,4))
font = {'size'   : 12}
mpl.rc('font', **font)

for i, env in enumerate(["arm", "archery", "talos"]):
    ax = plt.subplot2grid((1, 3), (0, i))
    ppo = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/PPO/{env}.pk")
    ours = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/PT-R-ME/{env}.pk")
    disti = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/NN/{env}.pk")
    c = np.pi/2 if env == "talos" else 1.
    data = [[np.mean(dic["R"])*c for dic in ours], [np.mean(dic["R"])*c for dic in disti], [np.mean(dic["R"])*c for dic in ppo.values()]]
    plot_boxplot(data, ["PT-R-ME\n(ours)", "PT-R-ME\n(distilled)", "PPO"], 
                 fig=fig, ax = ax,
                 use_stick=True, use_table=True, swarmsize=3, figsize=(8/3,4), colors=[ blue,  Orange, Teal], rotation=0)
    plt.title({"arm": "10-DoF Arm", "archery": "Archery", "talos": "Talos"}[env])
    plt.ylabel("")
    plt.xticks([])
    plt.box(False)
#savefig(f"Inference_all")
# -

for i in range(3):
    QD_score = data[i]
    print(["PT-R-ME (ours)", "PT-R-ME (distilled)", "PPO"][i], f"{np.median(QD_score):1.3f} [{np.quantile(QD_score, 0.25):1.3f}, {np.quantile(QD_score, 0.75):1.3f}]")

# # Replay

samples = Res[0]["archive"].samples

R = [s["reward"] for s in Res[0]["archive"].samples]

indices = np.argsort(R)
i = indices[-20000]
c = samples[i]["command"]
s = samples[i]["situation"]
print(R[i]*np.pi/2)

res = eval_command_talos_door_opening_ME(c, s, 0, config['command_config']["bounds"], config['situation_config']["bounds"], None, verbose=0, erase=False)
print(res['door_angle_after_pulling'])

path = res["path"] + '/0_0'
replay_talos_door_opening(unwrap_door_opening_situation(s, talos_opening_door_situation_bounds), path, verbose=73, erase=False )

"/home/pal/humanoid_adaptation/data/2023/11/27/15h30m26s802975" # 1.58096 (falling with the handle)
"/home/pal/humanoid_adaptation/data/2023/11/27/15h31m57s802975" # 1.31718
"/home/pal/humanoid_adaptation/data/2023/11/27/15h32m57s802975" # 1.15804
"/home/pal/humanoid_adaptation/data/2023/11/27/15h34m09s802975" # 0.976784
"/home/pal/humanoid_adaptation/data/2023/11/27/15h35m19s802975" # 0.756914
