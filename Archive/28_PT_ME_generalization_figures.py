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

# # Generalisation evaluation

# ## Load

env = "archery"

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


# + active=""
# Res = {}
# for folder in tqdm(os.listdir(path)):
#     with open(path+folder+"/archive.pk", "rb") as f:
#         Res[int(folder)] = {"archive": pickle.load(f)}
#     with open(path+folder+"/config.pk", "rb") as f:
#         Res[int(folder)]["config"] = pickle.load(f)

# + active=""
# keys = defaultdict(list)
# for key, res in Res.items():
#     keys[res["config"]["name"].split("_")[0]].append(key)
# names = keys

# + active=""
# if env == "talos":
#     archives = [{"archive": res["archive"]} for res in Res.values()]
# else:
#     archives = [{"archive": Res[i]["archive"]} for i in names["PT-ME (50% regression / 50% closest2parent sbx)"]]
# -

# ## Recompute Archive for != resolutions

def create_archive_from_samples(samples, n_cells, rep, verbose=0):
    dim = len(samples[0]["situation"])
    centroids = cvt(n_cells, dim, rep=rep)
    tree = cKDTree(centroids, leafsize=2)  
    elites = [None for _ in range(n_cells)]
    for sample in (tqdm(samples) if verbose else samples):
        _, index = tree.query(sample["situation"], k=1)
        if elites[index] is None or sample["reward"] >= elites[index]["reward"]:
            elites[index] = sample
    to_erase = []
    for i in range(n_cells):
        if elites[i] is None:
            to_erase.append(i)
    elites = np.delete(elites, to_erase, axis=0)
    centroids = np.delete(centroids, to_erase, axis=0)
    tree = cKDTree(centroids, leafsize=2) 
    delauney = Delaunay(centroids)
    neighbors = [[i] for i in range(len(centroids))]
    for neighborhood in delauney.simplices:
        for i in neighborhood:
            for j in neighborhood:
                neighbors[i].append(j)
                neighbors[j].append(i)
    centroid_neighbors = [list(set(n)) for n in neighbors]
    return {"tree": tree, "centroids": centroids, "elites": elites, "centroid_neighbors": centroid_neighbors}


N_cells = [int(x) for x in np.logspace(0, 5, 20)[3:-2]]
rep = 42

# + active=""
# Args = {}
# for j, archive in enumerate(archives):
#     for n_cells in N_cells:
#         Args[(j, n_cells)] = (archive["archive"].samples, n_cells, rep)
# jobs = make_general_jobs(create_archive_from_samples, Args)
#
# MR_Archives = general_master(jobs, n_proc, verbose=1, batch_size=50)
# save_pickle(f"/home/pal/notebooks/data/tmp/MR_archives_{env}.pk", MR_Archives)

# + active=""
# MR_Archives = open_pickle(f"/home/pal/notebooks/data/tmp/MR_archives_{env}.pk")
# -

# ## Select evaluation samples

if env == "talos":
    n = 1000
    S = cvt(n, 3, rep=0)     
else:
    n = 10_000
    S = cvt(n, 2, rep=0) 


def predict_action_from_archive(archive, S):
    A = []
    for s in S:
        _, idx = archive["tree"].query(s, k=1)
        indexes = archive["centroid_neighbors"][idx]  # find the direct neighbors using the precomputed delauney from the centroids 
        X = [archive["elites"][i]["situation"] for i in indexes]
        Y = [archive["elites"][i]["command"] for i in indexes]
        reg = LinearRegression().fit(X, Y)
        c = reg.predict(np.array([s]))[0] 
        dim = len(c)
        A.append(np.clip(c, np.zeros(dim), np.ones(dim)))
    return A


# + active=""
# args = {}
# for key, archive in MR_Archives.items():
#     args[key] = (archive, S)
# jobs = make_general_jobs(predict_action_from_archive, args)
# Actions = general_master(jobs, n_proc, 1)
# -

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


def eval_actions(actions, S):
    if env == "talos":
        pass
    else:
        R = []
        for i in range(len(S)):
            c = actions[i]
            s = S[i]
            R.append(min(1, (eval_f(c,s)-S_min_max[i]["min"])/(S_min_max[i]["max"]-S_min_max[i]["min"])))
        return R


# + active=""
# if env == "talos":
#     C = {i: [actions[i] for actions in Actions.values()] for i in range(len(S))}
#     evaluations = eval_talos_door_opening_batch(C, S, verbose=1)
#     Rewards = {}
#     for i, key in enumerate(Actions.keys()):
#         Rewards[key] = evaluations[i]
# else:
#     args = {}
#     for key in MR_Archives.keys():
#         args[key] = (Actions[key], S)
#     jobs = make_general_jobs(eval_actions, args)
#     Rewards = general_master(jobs, n_proc, 1)
# save_pickle(f"/home/pal/notebooks/data/PT-ME/inference/interpolation/{env}.pk", {"R": Rewards, "A": Actions})

# + active=""
# exit()
# -

# ## Plot

ppo = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/PPO/{env}.pk")

ppo_r = [np.mean(dic["R"]) for dic in ppo.values()]
ppo_data = [ppo_r, ppo_r]
PPO_X = [N_cells[0], N_cells[-1]]

PTME_r = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/neural_network/{env}.pk")
MTME_r = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/neural_network/{env}_MT-ME.pk")
MTME_data = [MTME_r, MTME_r]
MTME_X = [N_cells[0], N_cells[-1]]

np.min(PTME_r, axis=1)

# +
plot_with_spread(data=[PTME_r, ppo_data, MTME_data], 
                 names=["PT-ME Distillation", "PPO", "MT-ME Distillation", ], 
                 X=[N_cells, PPO_X, MTME_X], colors=[blue, Teal, Orange, ])

plt.legend()
plt.xscale("log")
plt.title(f"{env_name}")
plt.xlabel("Resolution (#cells in log scale)")
plt.ylabel("Generalization Score")
plt.grid(axis="y", alpha=0.5)
# -

# ### all in one





# +
fig, _ = plt.subplots(figsize=(16,14))
envs = ["arm", "archery", "talos"]
for i, env in enumerate(envs):
    if env in ["arm", "archery"]:
        ppo = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/PPO/{env}.pk")
        ppo_r = [np.mean(dic["R"]) for dic in ppo.values()]
        ppo_data = [ppo_r, ppo_r]
        PPO_X = [N_cells[0], N_cells[-1]]
        PTME_r = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/neural_network/{env}.pk")
        MTME_r = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/neural_network/{env}_MT-ME.pk")
        MTME_data = [MTME_r, MTME_r]
        MTME_X = [N_cells[0], N_cells[-1]]
        data = [PTME_r, ppo_data, MTME_data]
        names = ["PT-ME Distillation", "PPO", "MT-ME Distillation", ]
        X = [N_cells, PPO_X, MTME_X]
        colors = [blue, Teal, Orange, ]
        x_best = np.argmax(np.median(PTME_r, axis=1))
        boxplot_data = [PTME_r[x_best], ppo_r, MTME_r[0]]
    else:
        ppo = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/PPO/{env}.pk")
        ppo_r = [np.mean(dic["R"]) for dic in ppo.values()]
        ppo_data = [ppo_r, ppo_r]
        PPO_X = [N_cells[0], N_cells[-1]]
        
        Rewards = open_pickle(f"/home/pal/notebooks/data/PT-ME/inference/neural_network/{env}.pk")["R"]
        PTME_r = [[np.mean(Rewards[(i, res)]) for i in range(10)] for res in N_cells]
        data = [PTME_r, ppo_data]
        names = ["PT-ME Distillation", "PPO", ]
        X = [N_cells, PPO_X]
        colors = [blue, Teal,]
        x_best = np.argmax(np.median(PTME_r, axis=1))
        boxplot_data = [PTME_r[x_best], ppo_r]
        
    ax1 = plt.subplot2grid((len(envs), 3), (i, 0), colspan=2)
    plot_with_spread(data=data, 
                     names=names, 
                     X=X, colors=colors)
   
    plt.ylabel("Generalization Score" if i<2 else "Generalization Score (rad)")
    ax1.get_legend().remove()
    plt.xscale("log")
    plt.grid(axis="y", alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    if i == 2:
        plt.xlabel("Resolution (#cells in log scale)")
    ax = plt.subplot2grid((len(envs), 3), (i, 2), colspan=1)
    
    plot_boxplot(data=boxplot_data, names=names, colors=colors, 
            use_table=True, fig=fig, ax=ax, swarmsize=3, ylabel="")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    y_lim = ax.get_ylim()
    ax1.plot([N_cells[x_best], N_cells[x_best]], [y_lim[0], y_lim[1]], alpha=0.5, color="grey")

#savefig("generalization_comparison")
# -

# ### Comparison

nn_best = nn_r[np.argmax(np.median(nn_r, axis=1))]

inter_best = inter_r[np.argmax(np.median(inter_r, axis=1))]

plot_boxplot(data=[inter_best, nn_best,], names=["Interpolation", "Distillation", ], colors=[blue, Orange, ], 
            use_table=True)


