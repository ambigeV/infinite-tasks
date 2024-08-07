# -*- coding: utf-8 -*-
# # config

# ## Importations

import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
import numpy as np
import os
import datetime
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
import shutil
import pickle 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA 
import subprocess
import random 
import yaml 
import pandas as pd
import itertools
from tqdm import tqdm
from time import time, sleep
from copy import copy, deepcopy
import multiprocessing as mp
from scipy.stats import ttest_ind, ttest_rel, shapiro, mannwhitneyu, pearsonr
from shutil import copyfile, copytree
import pylab
from colorama import Fore
from IPython.display import HTML
import seaborn as sb
import pandas as pd
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
from ipywidgets import interact
from subprocess import CalledProcessError
from torch import nn
from torch.autograd import Variable
import torch
import queue
import abc
import warnings
from scipy.spatial.transform import Rotation
from collections.abc import Iterable
from cmath import rect, phase
from collections import Counter
import itertools
from time import time 
import multiprocessing as mp
from scipy.stats.qmc import Sobol
from math import log, ceil  
from collections.abc import Iterable
from itertools import product
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from collections import defaultdict

if os.uname()[1] == "multivac":
    n_proc = 60
elif os.uname()[1] == "evo256":
    n_proc = 60
else:
    n_proc = os.cpu_count()-2

# ## Matplotlib

# +
font = {'size'   : 18}
mpl.rc('font', **font)

plt.rcParams["figure.figsize"] = (16, 9)

blue = '#332288'
green = '#117733'
light_green = '#44AA99'
light_blue = '#88CCEE'
yellow = '#DDCC77'
red = "#CC6677"
grad = [blue, light_blue, light_green, green, yellow, "orange", red, "purple", '#AA4499']
colors = [light_green, red, light_blue, green, blue, yellow]


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


# -

# # General

# ## Saving

# +
def save(path, logs):
    with open(path+"/logs.pk", "wb") as f:
        pickle.dump(logs, f)

def date(one_file=True):
    now = datetime.datetime.now()
    return now.strftime("%Y_%m_%d_%Hh%Mm%S") if one_file else now.strftime("%Y/%m/%d/%Hh%Mm%S")

def create_save_folder(rootpath=None):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y/%m/%d/%Hh%Mm%Ss")
    save_path = ("/home/pal/notebooks/data/" if rootpath is None else rootpath) + timestamp
    if rootpath is None:
        print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def open_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            res = pickle.load(f)
        return res 
    
def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


# -

def tmp_save(X, name):
    with open("/home/pal/notebooks/data/tmp/"+name, "wb") as f:
        pickle.dump(X, f)
def tmp_load(name):
    with open("/home/pal/notebooks/data/tmp/"+name, "rb") as f:
        X = pickle.load(f)
    return X


def savefig(name):
    now = datetime.datetime.now()
    save_path = "/home/pal/notebooks/data/figures/"  + now.strftime("%Y/%m/%d/") 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"{save_path}{name}_{now.strftime('%Hh%Mm%S')}.pdf")


# ## Random Sampling

# ### Sampling Sobol

def sample_Sobol(n, bounds=None, dim=None):
    # ideally n is a power of 2 
    if bounds is None:
        sampler = Sobol(dim, scramble=True)
        return sampler.random_base2(int(np.ceil(np.log2(n))))
    else:
        sampler = Sobol(len(bounds["low"]), scramble=True)
        Xmin = bounds["low"]
        Xmax = bounds["high"]
        return np.array([x * (Xmax-Xmin) + Xmin for x in sampler.random_base2(int(np.ceil(np.log2(n))))])


# ### CVT

def cvt(k, dim, coef=10, verbose=False, rep=0):
    root = "/home/pal/notebooks/data/cvt/"
    name = f"{int(k)}_{int(dim)}_{rep}"
   
    if os.path.exists(root+name):
        with open(root+name, "rb") as f:
            X = pickle.load(f)
    else:
        x = np.random.rand(k*coef, dim)
        k_means = KMeans(init='k-means++', n_clusters=k, n_init=1, verbose=False)#,algorithm="full")
        k_means.fit(x)
        X = k_means.cluster_centers_
        with open(root+name, "wb") as f:
            pickle.dump(X, f)
    return X


# ## Change Ref

def change_ref(v, rot_vec, origin, is_vector, reverse=False):
    """ 
    intput: 
        is_vector: if true only consider rotation
        rot_vec, origin: represent the referentiel R in world 
    if reverse == False: 
        input: v (expressed in ref R) 
        output: v in World
    else: 
        input: v (expressed in world)
        output: v in R 
    """
    M_h = np.zeros((4, 4))
    rot_matrix = Rotation.from_rotvec(rot_vec).as_matrix()
    if reverse:
        rot_matrix = rot_matrix.transpose()
    for i in range(3):
        for j in range(3):
            M_h[i, j] = rot_matrix[i, j]
        M_h[i, 3] = -rot_matrix.dot(origin)[i] if reverse else origin[i] 
    M_h[3, 3] = 1
    assert(len(v) == 3)
    x = np.concatenate([v, [0. if is_vector else 1.]])
    return M_h.dot(x)[:3]


# ## General //

def general_master(jobs, n_processes=1, verbose=1, batch_size=None):
    if batch_size is None:
        return general_master_batch(jobs, n_processes, verbose)
    else:
        batches = []
        for i in range(int(np.ceil(len(jobs)/batch_size))):
            batches.append(general_master_batch(jobs[i*batch_size:(i+1)*batch_size], n_processes, verbose))

        Res = {}
        for batch in batches:
            for key, val in batch.items():
                Res[key] = val
        
        return Res 


# +
def make_general_jobs(foo, args):
    jobs = []
    for name, arg in args.items():
        jobs.append((foo, name, arg))
    return jobs

def general_worker(job_queue, res_queue):
    while True:
        job = job_queue.get()
        if job == "Done":
            break
        else:
            f, name, arg = job
            if type(arg) == tuple:
                res_queue.put((name, f(*arg)))
            elif type(arg) == dict:
                res_queue.put((name, f(**arg)))
            else:
                res_queue.put(None)

def general_master_batch(jobs, n_processes=1, verbose=1):
    if len(jobs) == 1:
        for job in jobs:
            f, name, arg = job
            if type(arg) == tuple:
                return {name: f(*arg)}
            elif type(arg) == dict:
                return {name: f(**arg)}
            
    job_queue = mp.Queue()
    res_queue = mp.Queue()
    n_processes = min(n_processes, len(jobs))
    pool = mp.Pool(n_processes, general_worker, (job_queue, res_queue))

    for job in jobs:
        job_queue.put(job)

    for _ in range(n_processes):
        job_queue.put("Done")
    
    res = {}
    for i in (tqdm(range(len(jobs)), smoothing=0.) if verbose else range(len(jobs))):
        name, out = res_queue.get()
        res[name] = out
        
    pool.terminate()
    return res 


# -

# ## Paramaters grid search

def grid_search(parameters):
    """
    parameters is a dict of str: list 
    with the list containing the list of paramaters to eval for the paramater
    """
    jobs = {}
    for vals in list( product( *tuple(vals for vals in parameters.values()) ) ):
        params = {}
        name = ""
        for i, key in enumerate(parameters.keys()):
            params[key] = vals[i]
            if type(vals[i]) == list:
                val_name = ""
                for x in vals[i]:
                    val_name += str(x)+"_"
                val_name = val_name[:-1]
                name += f"{key}_{val_name}_"
            else:
                name += f"{key}_{vals[i]}_"
        jobs[name[:-1]]= params
    return jobs 


# # For Talos

# +
joints = [[0,'rootJoint_pos_x',-np.inf,np.inf],
[1,'rootJoint_pos_y',-np.inf,np.inf],
[2,'rootJoint_pos_z',-np.inf,np.inf],
[3,'rootJoint_rot_x',-np.inf,np.inf],
[4,'rootJoint_rot_y',-np.inf,np.inf],
[5,'rootJoint_rot_z',-np.inf,np.inf],
[6,'leg_left_1_joint',-0.349066,1.5708],
[7,'leg_left_2_joint',-0.5236,0.5236],
[8,'leg_left_3_joint',-2.095,0.7],
[9,'leg_left_4_joint',0,2.618],
[10,'leg_left_5_joint',-1.27,0.68],
[11,'leg_left_6_joint',-0.5236,0.5236],
[12,'leg_right_1_joint',-1.5708,0.349066],
[13,'leg_right_2_joint',-0.5236,0.5236],
[14,'leg_right_3_joint',-2.095,0.7],
[15,'leg_right_4_joint',0,2.618],
[16,'leg_right_5_joint',-1.27,0.68],
[17,'leg_right_6_joint',-0.5236,0.5236],
[18,'torso_1_joint',-1.25664,1.25664],
[19,'torso_2_joint',-0.226893,0.733038],
[20,'arm_left_1_joint',-1.5708,0.785398],
[21,'arm_left_2_joint',0.00872665,2.87107],
[22,'arm_left_3_joint',-2.42601,2.42601],
[23,'arm_left_4_joint',-2.23402,-0.00349066],
[24,'arm_left_5_joint',-2.51327,2.51327],
[25,'arm_left_6_joint',-1.37008,1.37008],
[26,'arm_left_7_joint',-0.680678,0.680678],
[27,'gripper_left_inner_double_joint',-1.0472,0],
[28,'gripper_left_fingertip_1_joint',0,1.0472],
[29,'gripper_left_fingertip_2_joint',0,1.0472],
[30,'gripper_left_inner_single_joint',0,1.0472],
[31,'gripper_left_fingertip_3_joint',0,1.0472],
[32,'gripper_left_joint',-0.959931,0],
[33,'gripper_left_motor_single_joint',0,1.0472],
[34,'arm_right_1_joint',-0.785398,1.5708],
[35,'arm_right_2_joint',-2.87107,-0.00872665],
[36,'arm_right_3_joint',-2.42601,2.42601],
[37,'arm_right_4_joint',-2.23402,-0.00349066],
[38,'arm_right_5_joint',-2.51327,2.51327],
[39,'arm_right_6_joint',-1.37008,1.37008],
[40,'arm_right_7_joint',-0.680678,0.680678],
[41,'gripper_right_inner_double_joint',-1.0472,0],
[42,'gripper_right_fingertip_1_joint',0,1.0472],
[43,'gripper_right_fingertip_2_joint',0,1.0472],
[44,'gripper_right_inner_single_joint',0,1.0472],
[45,'gripper_right_fingertip_3_joint',0,1.0472],
[46,'gripper_right_joint',-0.959931,0],
[47,'gripper_right_motor_single_joint',0,1.0472],
[48,'head_1_joint',-0.20944,0.785398],
[49,'head_2_joint',-1.309,1.309]
]
lower_limits = []
upper_limits = []
for i in range(50):
    if i<3:
        lower_limits.append(0)
        upper_limits.append(1)
    elif i<6:
        lower_limits.append(-np.pi)
        upper_limits.append(np.pi)
    else:
        lower_limits.append(joints[i][2])
        upper_limits.append(joints[i][3])
lower_limits = np.array(lower_limits)
upper_limits = np.array(upper_limits)

talos_body = {
    "left_leg": [6,7,8,9,10,11],
    "right_leg": [12,13,14,15,16,17],
    "torso": [18,19],
    "left_arm": [20,21,22,23,24,25,26],
    "right_arm": [34,35,36,37,38,39,40],
    "head": [48,49],
}

talos_grouped_joints = {
    "l_hip": [6,7,8], 
    "r_hip": [12,13,14],
    "l_knee": [9],
    "r_knee": [15],
    "l_ankle": [10,11],
    "r_ankle": [16,17],
    "torso": [18,19],
    "l_shoulder": [20,21,22],
    "r_shoulder": [34,35,36],
    "l_elbow": [23,24],
    "r_elbow": [37,38],
}

def input_normalization(x):
    return (x-lower_limits)/(upper_limits-lower_limits)

discrepancy_joints_names  = ["leg_left_1_joint", "leg_left_2_joint", "leg_left_3_joint", "leg_left_4_joint", 
                             "leg_left_5_joint", "leg_left_6_joint", "leg_right_1_joint", "leg_right_2_joint", 
                             "leg_right_3_joint", "leg_right_4_joint", "leg_right_5_joint", "leg_right_6_joint",
                             "torso_1_joint", "torso_2_joint", "arm_left_1_joint", "arm_left_2_joint", 
                             "arm_left_3_joint", "arm_left_4_joint", "arm_right_1_joint", "arm_right_2_joint",
                             "arm_right_3_joint", "arm_right_4_joint"]
index32_to_50 = [0, 1, 2, 3, 4, 5,
                 6, 7, 8, 9, 10, 11,
                 12, 13, 14, 15, 16, 17,
                 18, 19,
                 20, 21, 22, 23, 24,
                 34, 35, 36, 37, 38,
                 48, 49]

names32 = [joints[i][1] for i in index32_to_50 ]

low, high = [], []
for i in range(32):
    low.append(joints[index32_to_50[i]][2])
    high.append(joints[index32_to_50[i]][3])
bounds_q32 = {"low": low, "high": high}

# +
short = [
 'L_Leg_1',
 'L_Leg_2',
 'L_Leg_3',
 'L_Leg_4',
 'L_Leg_5',
 'L_Leg_6',
 'R_Leg_1',
 'R_Leg_2',
 'R_Leg_3',
 'R_Leg_4',
 'R_Leg_5',
 'R_Leg_6',
 'Torso_1',
 'Torso_2',
 'L_Arm_1',
 'L_Arm_2',
 'L_Arm_3',
 'L_Arm_4',
 'L_Arm_5',
 'L_Arm_6',
 'L_Arm_7',
 'R_Arm_1',
 'R_Arm_2',
 'R_Arm_3',
 'R_Arm_4',
 'R_Arm_5',
 'R_Arm_6',
 'R_Arm_7',
]
    
long = [
 'leg_left_1_joint',
 'leg_left_2_joint',
 'leg_left_3_joint',
 'leg_left_4_joint',
 'leg_left_5_joint',
 'leg_left_6_joint',
 'leg_right_1_joint',
 'leg_right_2_joint',
 'leg_right_3_joint',
 'leg_right_4_joint',
 'leg_right_5_joint',
 'leg_right_6_joint',
 'torso_1_joint',
 'torso_2_joint',
 'arm_left_1_joint',
 'arm_left_2_joint',
 'arm_left_3_joint',
 'arm_left_4_joint',
 'arm_left_5_joint',
 'arm_left_6_joint',
 'arm_left_7_joint',
 'arm_right_1_joint',
 'arm_right_2_joint',
 'arm_right_3_joint',
 'arm_right_4_joint',
 'arm_right_5_joint',
 'arm_right_6_joint',
 'arm_right_7_joint',
]

short_to_long = {short[i]: long[i] for i in range(len(short))}
# -

# ## Conditions

# +
Conditions_raw = {
    "Nothing": {"Default": None},

    "Collision": {
        "Sphere_wrist": [True,[-0.05,  0.45, 0.95, 0., 0., 0., 0.2, 0.2, 0.2]],
        "Sphere_elbow": [True, [-0.23,  0.45, 1.25, 0., 0., 0., 0.2, 0.2, 0.2]],
        "Sphere_hand": [True, [0.15, 0.55, 1., 0., 0., 0., 0.2, 0.2, 0.2]],
        "Vertical_bar":  [False,[0.05, 0.55, 1., 0., 0., 0., 0.1, 0.1, 1.]],
        "Horizontal_bar": [False, [0.1, 0.55, 1.05, 0., 0., 0., 1, 0.1, 0.1]],
        "Cube_wrist":[False, [-0.05,  0.5, 0.9, 0., 0., 0., 0.2, 0.2, 0.2]],
        "Cube_elbow": [False, [0.1,  0.45, 1.25, 0., -0.7, 0., 0.175, 0.175, 0.175]],
        "Cube_hand": [False, [0.16, 0.57, 1., 0., 0., 0., 0.175, 0.175, 0.175]],
        "Rotated_cube": [False, [0.3, 1.02, 0.7 ,0.78, 0, 0.78, 0.2, 0.2, 0.2]],
        "Flat_cube": [False, [0.18, 0.25, 1.1, -0.75, 0, 0., 0.2, 0.3, 0.4]],
        "Wall": [False, [-0.1, 0.56, 1.1, 0., 0., 0., 1, 0.05, 1]],
    },

    "Locked": deepcopy(short_to_long),

    "Passive": deepcopy(short_to_long),
    
    "Weak": deepcopy(short_to_long),
    
    "Cut":  {key: long.replace("joint", "link") for (key,long) in short_to_long.items() if "torso" not in long},
}

collision_acronymes = {
                        "Sphere_wrist": "SW", "Sphere_elbow": "SE", "Sphere_hand": "SH", "Vertical_bar": "VB",
                        "Horizontal_bar": "HB", "Cube_wrist": "CW","Cube_elbow": "CE","Cube_hand": "CH", 
                        "Rotated_cube": "RC","Flat_cube": "FC", "Wall": 'WA', 
                       }

Conditions = {}           
yaml_types = {"Passive": 'passive', "Collision": 'colision_shapes', "Locked": 'locked', "Weak": 'weak_joints', "Cut": "amputated", "Nothing": None}    

for condition_type in Conditions_raw:
    conditions = Conditions_raw[condition_type]
    conditions_dict = {}
    for i, key in enumerate(conditions):
        if condition_type == "Collision": 
            acronyme = collision_acronymes[key]
        elif condition_type == "Default":
            acronyme = "DE"
        else:
            acronyme = ("L" if condition_type == "Locked" else "P") + str(i)
        full_name = condition_type + "_" + key
        if condition_type == "Weak":
            conditions_dict[full_name] = {"full name": full_name, "type":yaml_types[condition_type], "name": key, 
                                          "joints": conditions[key], "acronyme": acronyme, "weakness": 0.}
        else:
            conditions_dict[full_name] = {"full name": full_name, "type":yaml_types[condition_type], "name": key, 
                                          "joints": conditions[key], "acronyme": acronyme}
    Conditions[condition_type[0]] = conditions_dict

Conditions["A"] = {**Conditions["N"], **Conditions["L"],  **Conditions["P"], **Conditions["W"], **Conditions["C"]}
Conditions["D"] = {**Conditions["N"], **Conditions["L"], **Conditions["W"]}
for key in ["A", "D", "L", "P", "C", "N", "W"]:
    Conditions[f"{key}_names"] = list(Conditions[key].keys())

def one_condition(key):
    return {key: Conditions["A"][key]}


# -

def get_all_R_Leg_conditions(n=7):
    Conditions = [(True, [])]
    old_Conditions = []
    for i in range(1, n):
        old_Conditions = deepcopy(Conditions)
        Conditions = []
        for (b, condition) in old_Conditions:
            if b:
                c0, c1, c2, c3 = copy(condition), copy(condition), copy(condition), copy(condition)
                c1.append(f"Passive_R_Leg_{i}")
                c2.append(f"Locked_R_Leg_{i}")
                c3.append(f"Cut_R_Leg_{i}")
                Conditions.append((True,c1))
                Conditions.append((True,c2))
                Conditions.append((False,c3))
                Conditions.append((True,c0))
            else:
                Conditions.append((False,copy(condition)))
    return [c for (_,c) in Conditions if len(c)>0]


# ## Fonctions Definitions 

# ### create_xp_folder, set_yaml, execute

# +
def create_xp_folder(stamp_id=None):
    now = datetime.datetime.now()
    if stamp_id is None:
        timestamp = now.strftime("%Y/%m/%d/%Hh%Mm%Ss") + str(os.getpid())
    else:
        timestamp = now.strftime("%Y/%m/%d/%Hh%Mm%Ss") + str(stamp_id)
    xp_folder = "/home/pal/humanoid_adaptation/data/"+timestamp
    return xp_folder

def set_yaml(yaml_path, yaml_name, run_folder, config, subtype):
    with open(yaml_path+yaml_name, 'r') as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
    if subtype in ["TASK"]:
        if "torso" in doc:
            doc["torso"]["mask"] = "000110"
        if "momentum" in doc:
            doc["momentum"]["mask"] = "000110"
        for key, val in config[subtype].items():
            if key in doc:
                for key2, val2 in val.items():
                        doc[key][key2] = val2
            #else:
            #    print(f"WARNING:{key} is not in yaml original file")
    else:
        for key, val in config[subtype].items():
            if key in doc[subtype]:
                if type(val) == dict:
                    for key2, val2 in val.items():
                        doc[subtype][key][key2] = val2
                else:
                    doc[subtype][key] = val
            #else:
            #    print(f"WARNING:{key} is not in yaml original file")

    with open(run_folder + "/" + yaml_name, 'w') as f:
        yaml.dump(doc, f)

def execute(config, run_folder, actuator="spd", verbose=0, with_video=False, with_recording=False, with_ghost=False, fast=False, first=True):
    assert(actuator in ["torque", "spd", "servo"])
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(run_folder+"/behaviors", exist_ok=True)
    os.makedirs(run_folder+"/stabilizer", exist_ok=True)
    os.makedirs(run_folder+"/environment", exist_ok=True)
    with open(run_folder+"/config.pk", 'wb') as f:
        pickle.dump(config, f)
    yaml_path = "/home/pal/humanoid_adaptation/etc/"
    controller_name = "damage_controller.yaml" 
    behavior = config["arg"]["behavior"]
    # stabilizer
    for stab in ["double_support.yaml", "fixed_base.yaml", "single_support.yaml"]:
        copyfile(yaml_path+"stabilizer/"+stab, run_folder+"/stabilizer/"+stab)
    # controller 
    set_yaml(yaml_path, controller_name, run_folder, config, "CONTROLLER")
    # behavior 
    set_yaml(yaml_path+"behaviors/", behavior, run_folder+"/behaviors", config, "BEHAVIOR")
    # tasks
    set_yaml(yaml_path, "tasks.yaml", run_folder, config, "TASK")
    for file in ["configurations.srdf", "collision_thresholds.yaml", "frames.yaml", "talos_collisions_margin.yaml"]:
        copyfile(yaml_path+file, run_folder+"/"+file)
    if with_video or with_recording:
        exe = f"/home/pal/humanoid_adaptation/build/{'replay' if config['arg']['replay'] else 'damage'}_run_graphics"
    else:
        exe = f"/home/pal/humanoid_adaptation/build/{'replay' if config['arg']['replay'] else 'damage'}_run"
    conf = [exe, "-c", run_folder + "/" + controller_name, "-b", run_folder + "/behaviors/" + behavior , "-a", actuator]
    if with_video:
        conf.append("-woff")
    else:
        conf.append("-won")
    if with_ghost:
        conf.append("-g")
    if actuator in ["torque", "servo"]:
        conf.append("--closed_loop")
    if with_recording:
        conf.append("-mvideo.mp4")
    if fast:
        conf.append("-f")
        conf.append("-k")
        conf.append("dart")
    t1 = time()
    try:
        if verbose > 2:
            txt = subprocess.check_output(conf, timeout=120).decode()
            if "Talos Damage Controller initialized" in txt:
                txt = txt.split("Talos Damage Controller initialized")[-1]
            if verbose == 4: 
                with open("/home/pal/humanoid_adaptation/error/tmp.txt", "w") as f:
                    f.write(txt)
         
            else:
                print(txt)
        else:
            subprocess.check_output(conf, timeout=120) # removed 11/08
        return time()-t1
    except subprocess.TimeoutExpired:
        return time()-t1
    except subprocess.CalledProcessError as e:
        error_folder = "/home/pal/humanoid_adaptation/error/"
        os.makedirs(error_folder ,exist_ok=True)
        now = datetime.datetime.now()
        copytree(run_folder, error_folder+ now.strftime("%Y/%m/%d/%H:%M:%S") +"_"+ str(os.getpid()) +"/")
        print("subprocess.CalledProcessError: ", e)
        return time()-t1


# -

# ### gather_data

def gather_data(folder_path, files):
    data = {}
    for file in files:
        file_name = file.split(".")[0]
        if ".pk" in file:
            with open(folder_path + file, 'rb') as f:
                data[file_name] = pickle.load(f)
        else:
            data[file_name] = None  
            try:
                if os.path.exists(folder_path + file):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data[file_name] = np.loadtxt(folder_path + file)
            except OSError:
                pass 
            except ValueError:
                pass
    return data


# ### median filter, compute_*, detect_collision, falling

# +
def median_filter(L, k):
    filtered = []
    for i in range(len(L)):
        filtered.append( np.median(L[max(0,i-k):i+1], axis=0))
    return filtered

def median_filter2(L, k):
    filtered = []
    for i in range(len(L)):
        filtered.append( np.median(L[max(0,i-k//2):min(len(L), i+k//2)], axis=0))
    return filtered

def compute_discrepancy(data, real='sensor_tau', tsid='tsid_tau'):
    n = min(len(data[real]), len(data[tsid]))
    return np.linalg.norm(median_filter2(data[real][:n]-data[tsid][:n],10), axis=1)

def compute_foot_discrepancy(data, real='sensor_tau', tsid='tsid_tau'):
    assert('real' in real)
    n = min(len(data[real]), len(data[tsid]))
    return np.abs(median_filter2(np.abs(data[real][:n, 2])-np.abs(data[tsid][:n]), 10))

def compute_tracking_error(data, func=np.mean):
    n = min(len(data['rh_real']),len(data['rh_ref']))
    return func(np.linalg.norm(data['rh_real'][:n]-data['rh_ref'][:n], axis=1))

def compute_max_discrepancy(data):
    n = min(len(data['sensor_tau']),len(data['tsid_tau']))
    return np.max(np.linalg.norm(median_filter2(data['tsid_tau'][:n]-data['sensor_tau'][:n], 10), axis=1)[200:])

def falling(data):
    imu= median_filter(data['imu'][:,2],20)
    i= 100
    while i <len(imu) and imu[i]<-8.45:
        i+=1
    return i


# -

# ### Extract... Read_stop_reason, run_online

# +
def extract_body_part_in_contact(data, side):
    assert(side in ["r", "l"])
    if f'{side}contact_pos' in data and data[f'{side}contact_pos'] is not None:
        if len(data[f'{side}contact_pos'].shape) == 2:
            return int(data[f'{side}contact_pos'][0][4])
        elif len(data[f'{side}contact_pos'].shape) == 1:
            return int(data[f'{side}contact_pos'][4])
        else:
            return None 
    else:
        return None

def extract_hand_contact(data, side):
    assert(side in ["r", "l"])
    pos, wall_indices = None, []
    if f'{side}contact_pos' in data and data[f'{side}contact_pos'] is not None:
        if len(data[f'{side}contact_pos'].shape) == 2:
            pos = data[f'{side}contact_pos'][0][1:4]
            wall_indices = list(set(int(data[f'{side}contact_pos'][i][5]) for i in range(len(data[f'{side}contact_pos']))))
        elif len(data[f'{side}contact_pos'].shape) == 1 and len(data[f'{side}contact_pos']) == 6:
            pos = data[f'{side}contact_pos'][1:4]
            wall_indices = [int(data[f'{side}contact_pos'][5])]
    return {"pos": pos, "wall": wall_indices}

def extract_contact_time(data, side):
    assert(side in ["r", "l"])
    if f'{side}contact_pos' in data and data[f'{side}contact_pos'] is not None:
        if len(data[f'{side}contact_pos'].shape) == 2:
            return float(data[f'{side}contact_pos'][0][0])
        elif len(data[f'{side}contact_pos'].shape) == 1:
            return float(data[f'{side}contact_pos'][0])
        else:
            return None 
    else:
        return None
    
def read_stop_reason(data):
    reasons = ["Error", "Falling", "Running", "Fallen_floor", "Fallen_wall", 
               "Unfallen", "Recovered", "Timeout", "Auto_collision"]
    if "end" in data:
        try:
            end = data['end']
        except KeyError:
            return "Error", 0
        if end is None:
            return "Error", 0
        if len(end) == 0:
            return "Error", 0
        if len(end.shape) > 1:
            end = end[-1]
        i = int(end[1])
        if i>= len(reasons):
            return "Error", 0
        else:
            reason = reasons[i]
            if reason == "Recovered" and "contact_pos" in data:
                body = extract_body_part_in_contact(data)
                if body is not None:
                    reason = reason + str(body)
            return reason, end[0]
    else:
        return "Error", 0
    
def WrongConditionERROR(Exception):
    pass

def PathTooLongERROR(Exception):
    pass

def run_online(dic, verbose=0):
    myconfig = dic["config"]
    video = myconfig["arg"]["video"]
    recording = myconfig["arg"]["recording"]
    ghost =  myconfig["arg"]["ghost"]
    condition_names = ""
    conditions = dic["conditions"]
    for condition in conditions:
        if condition_names != "":
            condition_names += "&"
        condition_names += condition["full name"]
        if condition["type"] is not None and condition["type"] in "weak_joints":
            condition_names += f'_{condition["weakness"]}'
    xp_folder = dic["folder"] + "/" + dic["name"] + "/"
    myconfig["CONTROLLER"]["base_path"] = xp_folder #+= xp_folder.split("humanoid_adaptation")[1] 
    myconfig["CONTROLLER"]["xp_folder"] = xp_folder
    
    config = [
        ("CONTROLLER", "tasks", "tasks.yaml"),
        ('BEHAVIOR', "name", "hands"),
    ]
    for condition in conditions:
        if condition["type"] in ['passive', 'colision_shapes', 'locked']:
            myconfig['CONTROLLER'][condition["type"]].append(condition["joints"])
        if condition["type"] in ['amputated']:
            myconfig['CONTROLLER'][condition["type"]].append(condition["joints"])
        if condition["type"] == "weak_joints":
            myconfig['CONTROLLER']["weak_joints"].append(condition["joints"])
            myconfig['CONTROLLER']["weakness"].append(condition["weakness"])
    walltime = execute(
        config=myconfig, 
        run_folder=xp_folder, 
        actuator=myconfig["arg"]["actuator"],
        with_video=video, 
        with_recording=recording, 
        with_ghost=ghost,
        verbose=verbose
    )
    return dic["name"]


# -

len("/home/pal/humanoid_adaptation/data/2023/06/01/12h07m42s931794/00000000000000000000/config.pk")


# ### make_jobs, worker, master

# +
def make_jobs(dicts, verbose=0):
    jobs = {}
    for key, dic in dicts.items():
        jobs[key] = ((run_online, (dic, verbose)))
    return jobs

def worker(job_queue, res_queue):
    while True:
        job = job_queue.get()
        if job == "Done":
            break
        else:
            f, arg = job
            res_queue.put(f(*arg))

def master(jobs, n_processes=1, verbose=1, xp_folder=None, log_folder=None):
    job_queue = mp.Queue()
    res_queue = mp.Queue()
    n_processes = min(n_processes, len(jobs))
    pool = mp.Pool(n_processes, worker, (job_queue, res_queue))
    xp_folder = create_xp_folder() if xp_folder is None else xp_folder
    jobs_logs_path = create_save_folder(rootpath="/home/pal/notebooks/data/jobs_logs/") if log_folder is None else log_folder 
    
    with open(jobs_logs_path+"/jobs.pk", 'wb') as f:
        pickle.dump(jobs, f)
    if verbose:
        print("xp folder: ", xp_folder)
        print("jobs logs: ", jobs_logs_path)
        
    for job in jobs.values():
        (_, (dic, _)) = job
        dic["folder"] = xp_folder
        job_queue.put(job)

    for _ in range(n_processes):
        job_queue.put("Done")
    
    done = []
    for i in (tqdm(range(len(jobs)), smoothing=0.) if verbose else range(len(jobs))):
        out = res_queue.get()
        done.append(out)
        with open(jobs_logs_path + f"/jobs_log{i%2}.pk", 'wb') as f:
            pickle.dump(done, f)
        
    pool.terminate()
    return xp_folder

def resume_master(log_folder, xp_folder, verbose=1):
    with open(log_folder+"/jobs.pk", 'rb') as f:
        all_jobs = pickle.load(f)
    with open(log_folder+"/jobs_log0.pk", 'rb') as f:
        log0 = pickle.load(f)
    with open(log_folder+"/jobs_log1.pk", 'rb') as f:
        log1 = pickle.load(f)  
    log = log0 if len(log0) > len(log1) else log1
    jobs_todo = {}
    for key, job in tqdm(all_jobs.items()):
        if key not in log:
            jobs_todo[key] = job
    if len(jobs_todo)>0:
        master(jobs_todo, n_processes=n_proc, verbose=verbose, xp_folder=xp_folder, log_folder=log_folder)
    else:
        print("All Done")


# -

# ### Load // 

# +
def loader(path, name, files=['end']):
    stop_reasons = {}
    xp = path+"/"+name
    res = {}
    if "config" in files:
        with open(xp+"/config.pk", "rb") as f:
            config = pickle.load(f)
        res["config"] = config
        files.remove("config")
    data = gather_data(xp+"/", files)
    res["data"] = data
    if "end.dat" in files:
        res["stop_reason"] = read_stop_reason(data)
    return name, res

def load_worker(job_queue, res_queue):
    while True:
        job = job_queue.get()
        if job == "Done":
            break
        else:
            f, arg = job
            res_queue.put(f(**arg))
            
def load_master(dicts, jobs, n_jobs, n_processes=50, verbose=1):
    job_queue = mp.Queue()
    res_queue = mp.Queue()
    n_processes = min(n_processes, n_jobs)
    pool = mp.Pool(n_processes, load_worker, (job_queue, res_queue))

    for job in jobs:
        job_queue.put(job)

    for _ in range(n_processes):
        job_queue.put("Done")
    
    todos = []

    for _ in (tqdm(range(n_jobs)) if verbose else range(n_jobs)):
        todos.append(res_queue.get())
    
    for (name, dic) in todos:
        dicts[name] = dic 
    pool.terminate()


# -

# ### test_default 

def test_default(name, actuator="spd", recording=False, video=False, myconfig={}):
    config = deepcopy(default_params)
    config["arg"]["actuator"] = actuator
    config["arg"]["recording"] = recording
    config["arg"]["video"] = video
    for key, subconfigs in myconfig.items():
        for subkey, subconfig in subconfigs.items():
            config[key][subkey] = subconfig
    dic = {
        "name": name, 
        "folder": None, #create_xp_folder(),
        "config": config,
        "walltime": {},
        "time": {},
        "stop_reason": {},
        "mem": {},
    }
    return dic 


# ## default parameters

default_params = {
    "CONTROLLER": { 
        "stabilizer": {"activated": False},    
        "base_path": "/home/pal/humanoid_adaptation",
        "urdf": "talos_fast_collision(2H).urdf",
        "closed_loop": False,
        "xp_folder": "",
        "duration": 4.,
        "use_falling_early_stopping": True,
        "use_contact_early_stopping": True,
        "stop_at_first_contact": False,
        "fallen_treshold": 0.4,
        "colision_shapes": [[False, [-0.1, -0.9, 1.1, 0., 0., 0., 2, 0.05, 2.]]],
        "damage_time": 0.,
        "locked": [],
        "passive": [],
        "amputated": [],
        "use_push": False,
        "push_start": 0.,
        "push_vec": [0, 0, 0],
        "push_end": 0.,
        
        "reflex_time": 1000.1,
        "use_baseline": False,
        "remove_rf_tasks": False,
        "reflex_x": 0.,
        "collision_env_urdf": "",
        "use_training_early_stopping": False,
        "min_dist_to_wall": 0.07,
        "max_dist_to_wall": 0.08,
        "max_dist_to_target": 0.075,
        "dt": 0.001,
    }, 
    "BEHAVIOR": {
        #"com_trajectory_duration": 0.25,
        #"time_before_moving_com": 0.,
        #"com_ratio": 0,
        #"z": 0.0,
        "rh_shift": [0.,0.,0.,0.,0.,0.],
        "trajectory_duration": 1,
    },

    "arg": {
        "actuator": "spd",
        "video": False,
        "recording": False,
        "ghost": False,
        "behavior": "hands.yaml",
        "replay": False,
    },
    "TASK": {
        "com": {
            "weight": 1000.,
        },
        "posture": {
            "weight": 0.3,
        },
        "rh": {
            "weight": 0.,
            "mask": "111000",
        },
        "lh": {
            "weight": 0.,
        },
        "torso": {
            "weight": 0.,
        }, 
    },
    "ENV": {
 
    }
}


# ## Create a new episode 

# ### to sample situation 

# +
def to_flist(a):
    res = []
    for x in a:
        res.append(float(x))
    return res 

def sample_params(situations_params):
    # hands positions 
    lh = np.random.uniform(low=situations_params["lh"]["low"], high=situations_params["lh"]["high"])
    rh = np.random.uniform(low=situations_params["rh"]["low"], high=situations_params["rh"]["high"])
    while (np.linalg.norm(lh-rh))>1:
        lh = np.random.uniform(low=situations_params["lh"]["low"], high=situations_params["lh"]["high"])
        rh = np.random.uniform(low=situations_params["rh"]["low"], high=situations_params["rh"]["high"])
    d = np.random.uniform(low=situations_params["d"]["low"], high=situations_params["d"]["high"])
    alpha = np.random.uniform(low=situations_params["alpha"]["low"], high=situations_params["alpha"]["high"])
    collision_shape = [False, to_flist([d * np.cos(alpha), d * np.sin(alpha), 1.5, 0, 0, alpha, 0.01, 3., 3.])]
    key = (d, alpha)
    for X in [lh, rh]:
        key += tuple(X)
    return key, {"lh": to_flist(lh), "rh": to_flist(rh), "d": d, "alpha": alpha, "collision_shapes": [collision_shape]}

def compute_key(dic):
    config = dic["config"]
    lh, rh = config['BEHAVIOR']["lh_shift"], config['BEHAVIOR']["rh_shift"]
    w_p, w_n = config['CONTROLLER']['w_p'], config['CONTROLLER']['w_n']
    key = ()
    for X in [lh, rh, w_p, w_n]:
        key += tuple(X)
    return key, [lh, rh, w_p, w_n]

def sample_body_condition(body, proba_damaged, proba_cut, n_min_damaged):
    body_lengths = {"R_Leg": 6, "L_Leg": 6, "R_Arm": 7, "L_Arm": 7, "Torso": 2}
    assert body in body_lengths
    body_length = body_lengths[body]
    condition = []
    J = np.zeros(body_length)
    has_cut = False
    for i in range(1, body_length+1):
        if np.random.random() < proba_damaged:
            r = np.random.random()
            if r < proba_cut:
                condition.append(f"Cut_{body}_{i}")
                has_cut = True
                for j in range(i-1, body_length):
                    J[j] = True
                break
            elif r < (1+proba_cut)/2:
                condition.append(f"Passive_{body}_{i}")
                J[i-1] = True
            else:
                condition.append(f"Locked_{body}_{i}")
                J[i-1] = True
    if np.sum(J) < n_min_damaged:
        return sample_body_condition(body, proba_damaged, proba_cut, n_min_damaged)
    else:
        return condition, has_cut


def sample_condition(condition_config):
    if condition_config is None:
        condition_config = {"proba_damaged_leg": 0.5, "proba_damaged_upper_body": 0., "proba_amputation": 0.25}
    condition = []
    stop_cut = False
    for body in ["R_Leg", "R_Arm", "L_Arm", "Torso"]:
        n_min_damaged = 1 if "Leg" in body else 0
        proba_damaged = condition_config["proba_damaged_leg"] if "Leg" in body else condition_config["proba_damaged_upper_body"]
        proba_cut = 0. if (body == "Torso" or stop_cut) else condition_config["proba_amputation"]
        c, has_cut = sample_body_condition(body, proba_damaged, proba_cut, n_min_damaged)
        if has_cut and body == "R_Arm":
            stop_cut = True
        condition += c
    return condition

def sample_push(s, config):
    s["use_push"] = config["use_push"]
    s["push_start"] = 4.
    s["push_vec"] = np.random.uniform(**config["push_vec_bounds"])
    s["push_end"] = 4. + np.random.uniform(**config["push_duration_bounds"])


# -

# ### to compute situation

# +
def to_robot_referentiel(X, rotation, translation):
    return rotation.transpose() @ (X-translation)

def distance2edge(P, A, B):
    """
    Compute the distance between the point P and the edge AB
    """
    if np.linalg.norm(B-A) == 0.:
        return np.linalg.norm(P-A)
    s1 = (B-A).dot(P-A)
    s2 = (A-B).dot(P-B)
    if s1<0: 
        return np.linalg.norm(P-A)
    if s2<0:
        return np.linalg.norm(P-B)
    return np.linalg.norm(np.cross(P-A,B-A))/np.linalg.norm(B-A)

def compute_wall_distance_unit_vec(dic):
    """
    d : distance from the base to the wall 
    u : unitary vector of the wall (equale to [cos(alpha), sin(alpha)])
    C : aboslute position of the closest point on the wall to the robot base 
    """
    alpha, center, normal = dic["w_alpha"], dic["w_p"], dic["w_n"]
    A = center[:2] + np.array([3 * np.sin(alpha), -3 * np.cos(alpha)])
    B = center[:2] + np.array([-3 * np.sin(alpha), 3 * np.cos(alpha)])
    AB = B - A
    u = np.cross(normal, np.array([0,0,1]))
    u = u / np.linalg.norm(u)
    assert( np.linalg.norm(u-np.array([-np.sin(alpha), np.cos(alpha), 0])) < 0.01)
    # compute
    # dic["base"][:3] if using dart base  
    base = dic["q"][:3]
    R = np.array([base[0], base[1]])  # Robot base 
    d = distance2edge(R, A, B)
    
    # RA² = AC² + RC² 
    # OC = OR + RA + AC = OA + AC
    C = A + AB/np.linalg.norm(AB) * (np.linalg.norm(A-R)**2 - d**2 )**0.5
    return d, u, np.array([C[0], C[1], base[2]]) 

def compute_situation_talos(dic):
    d, alpha, q = dic["d"], dic["alpha"], dic["q"]
    p = [d * np.cos(alpha), d * np.sin(alpha), 0]
    n = np.array([-np.cos(alpha), -np.sin(alpha), 0])
    u = np.array([np.sin(alpha), -np.cos(alpha), 0])
    p_robot = change_ref(p, q[3:6], q[:3], is_vector=False, reverse=True)
    alpha_robot = alpha - Rotation.from_rotvec(dic["q"][3:6]).as_euler("xyz")[2]
    n_robot = np.array([-np.cos(alpha_robot), -np.sin(alpha_robot), 0])
    d_robot = abs(n_robot.dot(p_robot))
    C_robot = -d * n
    C = change_ref(C_robot, q[3:6], q[:3], is_vector=False, reverse=False)
    s = {"d_robot": d_robot, "n": n, "alpha_robot": alpha_robot, "C": C, "C_robot": C_robot, "n_robot": n_robot, "u": u}
    for key in ["q", "alpha", "d", "rh", "lh", "condition", "collision_shapes"]:
        if key in dic:
            s[key] = copy(dic[key])
    return s                                                                 
   
 # def compute_situation_talos(dic):
   # d, u, C = compute_wall_distance_unit_vec(dic)
   # Rotation.from_quat(dic["base"][-4:]).as_euler("xyz")[2] if using dart base
   # theta = Rotation.from_rotvec(dic["q"][3:6]).as_euler("xyz")[2]
   # s = {"d": d, "alpha": dic["w_alpha"]-theta, "w_C": C}
   # for key in ["q", "w_alpha", "w_p", "w_n", "rh", "lh", "condition"]:
   #     if key in dic:
   #         s[key] = copy(dic[key])
   # return s

def sample_situation_talos(wall_bounds):
    _, dic = sample_params(wall_bounds)
    dic["base"] = [0,0,0,1]
    dic["condition"] = sample_condition(None)
    return compute_situation_talos(dic)


# -

# ## For Sampling 

# ### sample_new_params(N, wall_bounds, verbose=1)

def sample_new_params(N, situations_params, n_proc, verbose=1):
    tested = 0
    good_samples = []
    while len(good_samples) < N:
        # sample params
        if tested != 0 and len(good_samples) != 0:
            n = max(n_proc, int((N-len(good_samples))/(len(good_samples)/tested)))
        else:
            n = max(n_proc, N-len(good_samples))
        tested += n
        params = {}
        for i in range(n):
            key, param = sample_params(situations_params)
            params[i] = param
        # test params
        data_dicts = {}
        for i, param in enumerate(params.values()):
            rh, lh = param["rh"], param["lh"]
            name = f"{i}"
            myconfig = {
                "CONTROLLER": {  
                    "duration": 4.002,
                    "colision_shapes": param["collision_shapes"], 
                    "damage_time": 4.,
                    "reflex_time": 4.002,
                    "use_baseline": False,
                    "log_level": 1, 
                    "use_reflex_trajectory": False,
                    "update_contact_when_detected": False,
                    "reflex_arm_stiffness": 1.,
                    "reflex_stiffness_alpha": 0.9999,
                    "remove_rf_tasks": False,
                    "w_p": to_flist([param["alpha"], param["d"]]), 
                    #"w_n": w_n,
                    "use_training_early_stopping": True, 
                    "min_dist_to_wall": 0.07,
                    "max_dist_to_wall": 0.08,
                    "max_dist_to_target": 0.075,
                    "check_model_collisions": True,
                    "dt": 0.001,
                }, 
                "TASK": {
                    "contact_rhand": {
                        "x": 0,
                        "y": 0,
                        "z": 0,
                        "kp": 30., 
                    },
                    "lh": {
                        "weight": 1000.,
                    },
                    "rh": {
                        "weight": 1000.,
                    },
                },
                "BEHAVIOR": {
                    "lh_shift": [float(lh[0]), float(lh[1]), float(lh[2]), 0., 0., 0.],
                    "rh_shift": [float(rh[0]), float(rh[1]), float(rh[2]), 0., 0., 0.],
                    "trajectory_duration": 4.,
                },
                "arg": {
                    "recording": False,
                    "video": verbose in [42],
                },
            }
            data_dicts[name] = test_default(name, actuator="spd", myconfig=myconfig)
            data_dicts[name]["conditions"] = [Conditions["A"]["Nothing_Default"]]
        if len(data_dicts) == 1:
            path = create_xp_folder()
            param = list(data_dicts.values())[0]
            param["folder"] = path
            _ = run_online(param, verbose)
            dicts = {}
            jobs = []
            (name, dic) = loader(path, name="0", files=["real_q.dat", "tsid_q.dat", "tsid_dq.dat", "lh.dat", "rh.dat", "end.dat", "base.dat", "config"])
            dicts[name] = dic
        else:
            jobs = make_jobs(data_dicts, verbose=verbose)
            path = master(jobs, n_processes=n_proc, verbose=verbose)
            dicts = {}
            jobs = []
            for name in os.listdir(path):
                jobs.append((loader, {"path": path, "name":name, "files":["real_q.dat", "tsid_q.dat", "tsid_dq.dat", "lh.dat", "rh.dat", "end.dat", "base.dat", "config"]}))
            n_jobs = len(jobs)
            load_master(dicts, jobs, n_jobs, n_processes=n_proc, verbose=0)
        if os.path.exists(path):
            subprocess.check_output(["rm", "-r", path])
        for dic in dicts.values():
            if dic["stop_reason"][0] == "Unfallen":
                wall_param = dic["config"]["CONTROLLER"]["w_p"]
                good_samples.append({
                     "base": dic["data"]['base'],
                     "rh": dic["config"]["BEHAVIOR"]["rh_shift"][:3],
                     "lh": dic["config"]["BEHAVIOR"]["lh_shift"][:3],
                     "q": dic["data"]['tsid_q'],
                     "dq": dic["data"]['tsid_dq'], 
                     "tsid_rh": dic["data"]['rh'],
                     "tsid_lh": dic["data"]['lh'],
                     "starting_configuration": dic["data"]['real_q'],
                     "collision_shapes": dic["config"]["CONTROLLER"]["colision_shapes"],
                     "alpha": wall_param[0],
                     "d": wall_param[1],
                })
        if verbose:
            print(f"samples found {len(good_samples)/N*100:2.1f}%")
    random_params = []
    for dic in good_samples[:N]:
        dic["condition"] = sample_condition(situations_params["condition_config"])
        sample_push(dic, situations_params["condition_config"])
        random_params.append(dic) 
    return random_params


# ### batch evaluation 

def generate_prebatch(batch):
    dicts = {} 
    for i, sample in enumerate(batch):
        s = sample["situation"]
        
        w_p, w_n = to_flist(s["w_p"]), to_flist(s["w_n"])
        rh, lh =  to_flist(s["rh"]), to_flist(s["lh"])
        w_alpha = float(s["w_alpha"])

        name = f"{i}"
        myconfig = {
            "CONTROLLER": {  
                "duration": 4.002,
                "colision_shapes": [[False, [w_p[0], w_p[1], w_p[2], 0., 0., w_alpha, 0.01, 6., 3.]]], 
                "damage_time": 4.,
                "reflex_time": 4.002,
                "use_baseline": False,
                "log_level": 1, 
                "use_reflex_trajectory": False,
                "update_contact_when_detected": False,
                "reflex_arm_stiffness": 1.,
                "reflex_stiffness_alpha": 0.9999,
                "remove_rf_tasks": False,
                "w_p": w_p, 
                "w_n": w_n,
                "use_training_early_stopping" : True,   
                "min_dist_to_wall": 0.07,
                "max_dist_to_wall": 0.08,
                "max_dist_to_target": 0.075,
                "check_model_collisions": True,
                "dt": 0.001,
            }, 
            "TASK": {
                "contact_rhand": {
                    "x": 0,
                    "y": 0,
                    "z": 0,
                    "kp": 30., 
                },
                "lh": {
                    "weight": 1000.,
                },
                "rh": {
                    "weight": 1000.,
                },
            },
            "BEHAVIOR": {
                "lh_shift": [float(lh[0]), float(lh[1]), float(lh[2]), 0., 0., 0.],
                "rh_shift": [float(rh[0]), float(rh[1]), float(rh[2]), 0., 0., 0.],
                "trajectory_duration": 4.,
            },
            "arg": {
                "recording": False,
                "video": False,
            },
        }
        dicts[name] = test_default(name, actuator="spd", myconfig=myconfig)
        dicts[name]["conditions"] = [[Conditions["A"]["Nothing_Default"]]]
    return dicts 


# + active=""
# def generate_batch(batch, verbose=0):
#     prebatch = generate_prebatch(batch)
#     jobs, n_jobs = make_jobs_custom_conditions(prebatch, verbose=0)
#     path = master(prebatch, jobs, n_jobs, n_processes=n_proc, verbose=verbose)
#     dicts = {}
#     jobs = []
#     for name in os.listdir(path):
#         jobs.append((loader, {"path": path, "name":name, "files":["real_q.dat", "tsid_q.dat", "tsid_dq.dat", "lh.dat", "rh.dat", "end.dat", "base.dat", "config"]}))
#     n_jobs = len(jobs)
#     load_master(dicts, jobs, n_jobs, n_processes=n_proc, verbose=0)
#     if os.path.exists(path):
#         subprocess.check_output(["rm", "-r", path])
#     final_samples = []
#     for name, dic in dicts.items():
#         _, [lh, rh, w_p, w_n] = compute_key(dic)
#         p = [w_p[0], w_p[1]]
#         d = np.linalg.norm(p)
#         w_alpha = float(np.arccos(p[0]/d)) if p[1]>=0 else -float(np.arccos(p[0]/d))
#         if check_sample(dic):
#             config = {
#                 "base": dic["data"]["Nothing_Default"]['base'],
#                 "w_n": w_n, 
#                 "w_p": w_p,
#                 "w_alpha": w_alpha,
#                 "lh": lh,
#                 "rh": rh,
#                 "condition": batch[int(name)]["situation"]["condition"],
#                 "q": dic["data"]["Nothing_Default"]['tsid_q'],
#             }
#             final_samples.append({"situation": compute_situation(config), "command": batch[int(name)]["command"]})
#     return generate_finalbatch(final_samples), final_samples
# -

def generate_finalbatch(final_samples):
    final_batch = {} 
    for i, sample in enumerate(final_samples):
        c = sample["command"]
        s = sample["situation"]
        
        w_p, w_n = to_flist(s["w_p"]), to_flist(s["w_n"])
        w_C = to_flist(s["w_C"])
        lh, rh = s["lh"], s["rh"]
        w_alpha = s["w_alpha"]
        condition = s["condition"]
        
        [r_x, r_z, use_right, l_x, l_z, use_left] = c
        use_right = use_right > 0
        use_left = use_left > 0
        
        rhc = [ w_C[0] + 0.07 * w_n[0] - r_x * np.sin(w_alpha),
                w_C[1] + 0.07 * w_n[1] + r_x * np.cos(w_alpha),
                w_C[2] + r_z]
        
        lhc = [ w_C[0] + 0.07 * w_n[0] - l_x * np.sin(w_alpha),
                w_C[1] + 0.07 * w_n[1] + l_x * np.cos(w_alpha),
                w_C[2] + l_z]
        
        name = sample["name"] if "name" in sample else f"{i}"
        myconfig = {
            "CONTROLLER": {  
                "duration": 10.,
                "colision_shapes": [[False, [w_p[0], w_p[1], w_p[2], 0., 0., w_alpha, 0.01, 6., 3.]]],
                "damage_time": 4.,
                "reflex_time": 4.002,
                "use_baseline": False,
                "log_level": 0, 
                "use_reflex_trajectory": False,
                "update_contact_when_detected": True,
                "remove_rf_tasks": False,
                "reflex_arm_stiffness": 1.,
                "reflex_stiffness_alpha": 0.9999,
                "w_p": w_p, 
                "w_n": w_n, 
                "r_x": float(r_x),
                "r_z": float(r_z),
                "l_x": float(l_x),
                "l_z": float(l_z),
                "condition": condition,
                "use_left_hand": bool(use_left),
                "use_right_hand": bool(use_right), 
                "use_training_early_stopping" : True, 
                "use_falling_early_stopping": True,
                "use_contact_early_stopping": True,
                "min_dist_to_wall": 0.07,
                "max_dist_to_wall": 0.08,
                "max_dist_to_target": 0.075,
                "check_model_collisions": True,
                "dt": 0.001,
            }, 
            "TASK": {
                "contact_rhand": {
                    "x": float(rhc[0]),
                    "y": float(rhc[1]),
                    "z": float(rhc[2]),
                    "normal": w_n,
                    "kp": 30., 
                },
                "contact_lhand": {
                    "x": float(lhc[0]),
                    "y": float(lhc[1]),
                    "z": float(lhc[2]),
                    "normal": w_n,
                    "kp": 30., 
                },
                "lh": {
                    "weight": 1000.,
                },
                "rh": {
                    "weight": 1000.,
                },
            },
            "BEHAVIOR": {
                "lh_shift": [float(lh[0]), float(lh[1]), float(lh[2]), 0., 0., 0.],
                "rh_shift": [float(rh[0]), float(rh[1]), float(rh[2]), 0., 0., 0.],
                "trajectory_duration": 4.,
            },
            "arg": {
                "recording": False,
                "video": False,
            },
        }
        final_batch[name] = test_default(name, actuator="spd", myconfig=myconfig)
        final_batch[name]["conditions"] = [[Conditions["A"][c] for c in condition]]
    return final_batch


# ## Door opening

# ### Sample new situation

def sample_door_opening_param(situations_params):
    orientation = np.random.uniform(**situations_params["orientation"])
    position = np.random.uniform(**situations_params["position"])
    situation = {
        "door_pos": to_flist(list(orientation) + list(position)),
        "handle_opening_angle": float(np.random.uniform(**situations_params["handle_opening_angle"])),
        "handle_kp": float(np.random.uniform(**situations_params["handle_kp"])),
        "handle_kd":float(np.random.uniform(**situations_params["handle_kd"])),
    } 
    return situation


def sample_situation_talos_door_opening(N, situations_params, n_proc, verbose=0):
    tested = 0
    good_samples = []
    while len(good_samples) < N:
        # sample params
        if tested != 0 and len(good_samples) != 0:
            n = max(n_proc, int((N-len(good_samples))/(len(good_samples)/tested)))
        else:
            n = max(n_proc, N-len(good_samples))
        tested += n
        params = {}
        for i in range(n):
            params[i] = sample_door_opening_param(situations_params)
        # test params
        data_dicts = {}
        for i, s in enumerate(params.values()):
            name = f"{i}"           
            myconfig = {
                "CONTROLLER": {  
                    "duration": 0.01,
                    "log_level": 1, 
                    "check_model_collisions": True,
                    "urdf": "talos.urdf",
                    "door_urdf": "/home/pal/humanoid_adaptation/urdf/door/door.urdf",
                    "door_pos": s["door_pos"],
                    "handle_opening_angle": s["handle_opening_angle"],
                    "handle_kp": s["handle_kp"],
                    "handle_kd": s["handle_kd"],
                    "damage_time": 1000,
                    "colision_shapes": [], 
                    "update_contact_when_detected": False,
                    "condition": "Nothing_Default",
                    "use_left_hand": False,
                    "use_right_hand": False, 
                    "use_training_early_stopping": True,
                    "use_falling_early_stopping": False,
                    "use_contact_early_stopping": False,
                    "dt": 0.001,
                }, 
                "TASK": {
                    "lh": {
                        "weight": 0.,
                    },
                    "rh": {
                        "weight": 1000.,
                        "mask": "111111",
                    },
                    "torso": {
                        "weight": 100.,
                    },
                    "posture": {
                        "weight": 10.,
                    },

                },
                "arg": {
                    "recording": verbose == 73,
                    "video": verbose in [42, 73],
                    "behavior": "door_opening.yaml",
                    "ghost": False,
                },
            }
            param = test_default(name, actuator="spd", myconfig=myconfig)
            param["conditions"] = [Conditions["A"][c] for c in ["Nothing_Default"]]
            data_dicts[name] = param
        if len(data_dicts) == 1:
            path = create_xp_folder()
            param = list(data_dicts.values())[0]
            param["folder"] = path
            _ = run_online(param, verbose)
            dicts = {}
            jobs = []
            (name, dic) = loader(path, name="0", files=["real_q.dat", "tsid_q.dat", "tsid_dq.dat", "lh.dat", "rh.dat", "end.dat", "base.dat", "config"])
            dicts[name] = dic
        else:
            jobs = make_jobs(data_dicts, verbose=verbose)
            path = master(jobs, n_processes=n_proc, verbose=verbose)
            dicts = {}
            jobs = []
            for name in os.listdir(path):
                jobs.append((loader, {"path": path, "name":name, "files":["real_q.dat", "tsid_q.dat", "tsid_dq.dat", "lh.dat", "rh.dat", "end.dat", "base.dat", "config"]}))
            n_jobs = len(jobs)
            load_master(dicts, jobs, n_jobs, n_processes=n_proc, verbose=0)
        if os.path.exists(path):
            subprocess.check_output(["rm", "-r", path])
        for dic in dicts.values():
            if dic["stop_reason"][0] == "Unfallen":
                good_samples.append({
                     "door_pos": dic["config"]["CONTROLLER"]["door_pos"],
                     "handle_opening_angle": dic["config"]["CONTROLLER"]["handle_opening_angle"],
                     "handle_kp": dic["config"]["CONTROLLER"]["handle_kp"],
                     "handle_kd": dic["config"]["CONTROLLER"]["handle_kd"],
                })
        if verbose:
            print(f"samples found {len(good_samples)/N*100:2.1f}%")
    random_params = []
    for dic in good_samples[:N]:
        random_params.append(dic) 
    return random_params


# ### create_batch_talos_door_opening_situation_check

def create_batch_talos_door_opening_situation_check(params, verbose=0):
    data_dicts = {}
    for i, s in enumerate(params):
        name = f"{i}"           
        myconfig = {
            "CONTROLLER": {  
                "duration": 0.01,
                "log_level": 1, 
                "check_model_collisions": True,
                "urdf": "talos.urdf",
                "door_urdf": "/home/pal/humanoid_adaptation/urdf/door/door.urdf",
                "door_pos": s,
                "handle_opening_angle": np.pi/4,
                "handle_kp": 0.1,
                "handle_kd": 0.1,
                "damage_time": 1000,
                "colision_shapes": [], 
                "update_contact_when_detected": False,
                "condition": "Nothing_Default",
                "use_left_hand": False,
                "use_right_hand": False, 
                "use_training_early_stopping": True,
                "use_falling_early_stopping": False,
                "use_contact_early_stopping": False,
                "dt": 0.001,
            }, 
            "TASK": {
                "lh": {
                    "weight": 0.,
                },
                "rh": {
                    "weight": 1000.,
                    "mask": "111111",
                },
                "torso": {
                    "weight": 100.,
                },
                "posture": {
                    "weight": 10.,
                },

            },
            "arg": {
                "recording": verbose == 73,
                "video": verbose in [42, 73],
                "behavior": "door_opening.yaml",
                "ghost": False,
            },
        }
        param = test_default(name, actuator="spd", myconfig=myconfig)
        param["conditions"] = [Conditions["A"][c] for c in ["Nothing_Default"]]
        data_dicts[name] = param
    return data_dicts 


def evaluate_batch_talos_door_opening_situation_check(data_dicts, n_proc, verbose=1):
    if len(data_dicts) == 1:
        path = create_xp_folder()
        param = list(data_dicts.values())[0]
        param["folder"] = path
        _ = run_online(param, verbose)
        dicts = {}
        jobs = []
        (name, dic) = loader(path, name="0", files=["real_q.dat", "tsid_q.dat", "tsid_dq.dat", "lh.dat", "rh.dat", "end.dat", "base.dat", "config"])
        dicts[name] = dic
    else:
        jobs = make_jobs(data_dicts, verbose=verbose)
        path = master(jobs, n_processes=n_proc, verbose=verbose)
        dicts = {}
        jobs = []
        for name in os.listdir(path):
            jobs.append((loader, {"path": path, "name":name, "files":["real_q.dat", "tsid_q.dat", "tsid_dq.dat", "lh.dat", "rh.dat", "end.dat", "base.dat", "config"]}))
        n_jobs = len(jobs)
        load_master(dicts, jobs, n_jobs, n_processes=n_proc, verbose=0)
    samples = []
    for dic in dicts.values():
        samples.append({
             "door_pos": dic["config"]["CONTROLLER"]["door_pos"],
             "handle_opening_angle": dic["config"]["CONTROLLER"]["handle_opening_angle"],
             "handle_kp": dic["config"]["CONTROLLER"]["handle_kp"],
             "handle_kd": dic["config"]["CONTROLLER"]["handle_kd"],
             "stop_reason": dic["stop_reason"][0],
        })
    if os.path.exists(path):
        subprocess.check_output(["rm", "-r", path])
    return samples


# ### Sample action 

def sample_command_from_dic(bounds):
    return { key: np.random.uniform(**bounds[key]) for key in bounds.keys()}


def sample_action(command_config):
    return sample_command_from_dic(command_config["bounds"])


# ### generate_param_talos_door_opening

# +
def from_mine_to_quaternion(x):
    """
    yaw [-np.pi, np.pi] around z (pi/2 = left, -pi/2 = right)
    pitch [-np.pi/2, np.pi/2] around the new y (pi/2 = down, -pi/2 up) 
    roll [-np.pi, np.pi] around u (pi/2 = clockwise)
    u = (direction of the palm) [np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)]
    """
    [yaw, pitch, roll] = x 
    base = Rotation.from_rotvec([np.pi/2, 0, 0]).as_matrix().dot(Rotation.from_rotvec([0, -np.pi/2, 0]).as_matrix())  # put the hand to exterior, the thumb down
    yaw_rot_matrix = Rotation.from_rotvec([0, 0, yaw]).as_matrix()
    pitch_rot_matrix = Rotation.from_rotvec([-np.sin(yaw) * pitch, np.cos(yaw) * pitch, 0]).as_matrix()
    u = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    roll_rot_matrix = Rotation.from_rotvec(u * roll).as_matrix()
    q = Rotation.from_matrix(roll_rot_matrix.dot(pitch_rot_matrix.dot(yaw_rot_matrix.dot(base)))).as_quat()
    return [float(q[3])] + to_flist(q[:3])

def generate_param_talos_door_opening(situations, commands, verbose=0):
    batch = {}
    for i, s in enumerate(situations):
        for j, c in enumerate(commands[i] if type(commands) == dict else commands):
            name = f"{i}_{j}"
            handle_pos_in_base = [0.46, 0.12, 1]  # fixed in URDF
            handle_pos = s[3:6] + Rotation.from_rotvec(s[:3]).as_matrix().dot(handle_pos_in_base)
            gripper_pos = to_flist(handle_pos + c["gripper_shift"])
            pull_handle_orientation = np.array(c["gripper_orientation"]) + c["pull_handle_shift"]
            
            myconfig = {
                "CONTROLLER": {  
                    "duration": 7,
                    "log_level": 1, 
                    "check_model_collisions": True,
                    "urdf": "talos.urdf",
                    "door_urdf": "/home/pal/humanoid_adaptation/urdf/door/door_vertical.urdf",
                    "door_pos": s,
                    "damage_time": 1000,
                    "colision_shapes": [], 
                    "update_contact_when_detected": False,
                    "condition": "Nothing_Default",
                    "use_left_hand": False,
                    "use_right_hand": False, 
                    "use_training_early_stopping": False,
                    "use_falling_early_stopping": False,
                    "use_contact_early_stopping": False,
                    "dt": 0.001,
                }, 
                "TASK": {
                    "lh": {
                        "weight": 0.,
                    },
                    "rh": {
                        "weight": 1000.,
                        "mask": "111111",
                    },
                    "torso": {
                        "weight": 100.,
                    },
                    "posture": {
                        "weight": 10.,
                    },

                },
                "BEHAVIOR": {
                  "reach_handle_shift": to_flist(gripper_pos) + from_mine_to_quaternion(c["gripper_orientation"]),
                  "reach_handle_duration": 3.,

                  "close_gripper_joint": -0.8,
                  "close_gripper_duration": 1,

                  "turn_handle_shift": to_flist(c["pull_handle_shift"]) + from_mine_to_quaternion(pull_handle_orientation),
                  "turn_handle_duration": 3,
                    
                },
                "arg": {
                    "recording": verbose in [73, 730],
                    "video": verbose in [42, 73],
                    "behavior": "door_opening.yaml",
                    "ghost": False,
                },
            }
            param = test_default(name, actuator="spd", myconfig=myconfig)
            param["conditions"] = [Conditions["A"][c] for c in ["Nothing_Default"]]
            batch[name] = param
    return batch


# -

# ### evaluate_batch_talos_door_opening

def evaluate_batch_talos_door_opening(batch, verbose=0, erase=True, n_processes=None):
    files = ["end.dat", "tsid_q.dat", "config", "door.dat", "door_contact.dat"]
    dicts, jobs, evaluations = {}, [], {}
    if len(batch) == 1:
        if verbose > 1:
            print("One process evaluation")
        path = create_xp_folder()
        if verbose == 73:
            print(path)
        name, param = list(batch.items())[0]
        param["folder"] = path
        _ = run_online(param, verbose)

        (name, dic) = loader(path, name=name, files=files)
        dicts[name] = dic
    else:
        if verbose > 1:
            print(f"{n_proc} processes evaluations")
        jobs = make_jobs(batch, verbose=verbose)
        path = master(jobs, n_processes=n_processes if n_processes is not None else n_proc, verbose=verbose)
        jobs = []
        for name in os.listdir(path):
            jobs.append((loader, {"path": path, "name":name, "files":files}))
        n_jobs = len(jobs)
        load_master(dicts, jobs, n_jobs, n_processes=n_processes if n_processes is not None else n_proc, verbose=0)
     
    for name, dic in dicts.items():
        config = dic["config"]
        
        door_contact = dic["data"]["door_contact"]

        reach_handle_duration = config["BEHAVIOR"]["reach_handle_duration"]
        close_gripper_duration = config["BEHAVIOR"]["close_gripper_duration"]
        turn_handle_duration = config["BEHAVIOR"]["turn_handle_duration"]
        
        end_pulling = reach_handle_duration + close_gripper_duration + turn_handle_duration

        end_pulling = int(end_pulling * 1000)-1

        # only count if the robot has not fallen and if it had let go of the door (no contact with the door for the last 100ms)
        if dic["data"]["door"] is not None and len(dic["data"]["door"].shape) == 1:
            evaluations[name] = {
                "stop_reason": dic["stop_reason"],
                "path": path, 
                "name": name, 
                "door_angle_after_pulling": max(0., dic["data"]["door"][end_pulling]) if len(dic["data"]["door"]) > end_pulling else 0.,
                "max_door_angle": np.max(dic["data"]["door"][:end_pulling]), 
            }
        else:
                evaluations[name] = {
                "stop_reason": dic["stop_reason"],
                "path": path, 
                "name": name, 
                "door_angle_after_pulling": 0.,
                "max_door_angle": 0., 
            }
        if verbose > 1:
            evaluations[name]["param"] = dic
    
    if erase and os.path.exists(path):
        subprocess.check_output(["rm", "-r", path])
    
    return evaluations


# ### eval_talos_door_opening

def eval_talos_door_opening(c, s, verbose=0, erase=True ):
    batch = generate_param_talos_door_opening([s], [c], verbose=verbose)
    evaluations = evaluate_batch_talos_door_opening(batch, erase= erase and verbose != 73, verbose=verbose)
    return evaluations["0_0"]


def eval_talos_door_opening_batch(C, S, verbose=0, erase=True ):
    """ C must be a disct with keys as int corresponding to S indices """
    batch = generate_param_talos_door_opening(S, C, verbose=verbose)
    evaluations = evaluate_batch_talos_door_opening(batch, erase= erase and verbose != 73, verbose=verbose)
    return evaluations


def replay_talos_door_opening(s, path, verbose=0, erase=True ):
    batch = generate_param_talos_door_opening_replay(s, path, verbose=verbose)
    evaluations = replay_batch_talos_door_opening(batch, erase= erase and verbose != 73, verbose=verbose)


# ### Replay

def generate_param_talos_door_opening_replay(s, path, verbose=0):
    batch = {}
    name = f"{0}_{0}"
    myconfig = {
        "CONTROLLER": {  
            "duration": 7,
            "log_level": 1, 
            "check_model_collisions": True,
            "urdf": "talos.urdf",
            "door_urdf": "/home/pal/humanoid_adaptation/urdf/door/door_vertical.urdf",
            "door_pos": s,
            "damage_time": 1000,
            "colision_shapes": [], 
            "update_contact_when_detected": False,
            "condition": "Nothing_Default",
            "use_left_hand": False,
            "use_right_hand": False, 
            "use_training_early_stopping": False,
            "use_falling_early_stopping": False,
            "use_contact_early_stopping": False,
            "dt": 0.001, 
            "dart_q_path": path + "/dart_q.dat",
            "door_path": path + "/door.dat",
        }, 
        "arg": {
            "recording": verbose in [73, 730],
            "video": verbose in [42, 73],
            "behavior": "door_opening.yaml",
            "ghost": False,
            "replay": True,
        },
    }
    param = test_default(name, actuator="spd", myconfig=myconfig)
    param["conditions"] = [Conditions["A"][c] for c in ["Nothing_Default"]]
    batch[name] = param
    return batch


def replay_batch_talos_door_opening(batch, verbose=0, erase=True, n_processes=None):
    files = ["end.dat", "tsid_q.dat", "config", "door.dat", "door_contact.dat"]
    dicts, jobs, evaluations = {}, [], {}
    if len(batch) == 1:
        if verbose > 1:
            print("One process evaluation")
        path = create_xp_folder()
        if verbose == 73:
            print(path)
        name, param = list(batch.items())[0]
        param["folder"] = path
        _ = run_online(param, verbose)

        (name, dic) = loader(path, name=name, files=files)
        dicts[name] = dic
    else:
        if verbose > 1:
            print(f"{n_proc} processes evaluations")
        jobs = make_jobs(batch, verbose=verbose)
        path = master(jobs, n_processes=n_processes if n_processes is not None else n_proc, verbose=verbose)
        jobs = []
        for name in os.listdir(path):
            jobs.append((loader, {"path": path, "name":name, "files":files}))
        n_jobs = len(jobs)
        load_master(dicts, jobs, n_jobs, n_processes=n_processes if n_processes is not None else n_proc, verbose=0)
    
    if erase and os.path.exists(path):
        subprocess.check_output(["rm", "-r", path])



# # Gym Env

# ## General

# +
def compute_situation_gym(situations):
    return situations

def sample_new_params_gym(N, gym_env, gym_params):
    env = gym_env(**gym_params)
    situations = {}
    for i in range(N):
        situations[i] = env.reset()
    return situations

def sample_situation_gym(env):
    return env.reset()


# -

# ## Circle

class CircleEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, action_bounds, observation_bounds, thresholds):
        super(CircleEnv, self).__init__()
        self.action_bounds = action_bounds
        self.action_space = spaces.Box(low=np.array(action_bounds["low"], dtype=np.float64), high=np.array(action_bounds["high"], dtype=np.float64), dtype=np.float64)
        self.observation_bounds = observation_bounds
        self.observation_space = spaces.Box(low=np.array(observation_bounds["low"], dtype=np.float64), high=np.array(observation_bounds["high"], dtype=np.float64), dtype=np.float64)
        self.thresholds = thresholds
        self.current_threshold = None 
        self._state = None 

    def step(self, action):
        assert action.shape ==  self._state.shape, f"action {action} has not the good shape: has {action.shape} but should have shape {self._state.shape}"
        distance = np.linalg.norm(self.unwrap_action(action) - self._state)
        if self.current_threshold == "easy":
            reward = 0 if (distance < self.thresholds["easy"]) else -(distance-self.thresholds["easy"])/self.thresholds["easy"]
        else:
            reward = 0 if (distance < self.thresholds["hard"]) else -(distance-self.thresholds["hard"])/self.thresholds["hard"]
        return None, reward, True, {"threshold": self.current_threshold, "truth": distance < self.thresholds[self.current_threshold]}
    
    def reset(self, s=None):
        self._state = self.observation_space.sample() if s is None else self.unwrap_observation(s) 
        self.current_threshold = "easy" if np.random.random() < self.thresholds["proba"] else "hard"
        return self.wrap_observation(self._state)
    
    def get_info(self):
        return {"state": self._state, "threshold": self.current_threshold}
    
    def wrap_observation(self, x):
        # from [L, H] -> [-1, 1]
        H = self.observation_bounds["high"]
        L = self.observation_bounds["low"]
        return (2 * x - (H + L)) / (H-L)
    
    def unwrap_observation(self, x):
        # from [-1, 1] -> [L, H]
        H = self.observation_bounds["high"]
        L = self.observation_bounds["low"]
        return (H-L)/2 * x + (H+L)/2 
    
    def unwrap_action(self, x):
        # from [-1, 1] -> [L, H]
        H = self.action_bounds["high"]
        L = self.action_bounds["low"]
        return (H-L)/2 * x + (H+L)/2 
    
    def wrap_action(self, x):
        # from [L, H] -> [-1, 1]
        H = self.action_bounds["high"]
        L = self.action_bounds["low"]
        return (2 * x - (H + L)) / (H-L)
    
    def sample_action(self):
        action = self.action_space.sample()
        return self.wrap_action(action)

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass 


# ## Archery

# +
def eval_archery(c, s, verbose=0):
    n = len(s)//2
    assert(n==1)
    rewards = {}
    for i in range(n):
        [yaw, pitch] = c[2*i:2*i+2]
        if type(s) == dict:
            d = s[f"d_{i}"] # distance
            w = s[f"w_{i}"] # wind
        else:
            d = s[2*i] # distance
            w = s[2*i+1] # wind
        v0 = 70
        # average velocity of an arrow 70 m.s-1 
        # with yaw = 0, pitch = 0.5 * arcsin(g*d/v0**2)
        v = v0 * np.array([-np.sin(yaw), np.cos(yaw)*np.cos(pitch), np.cos(yaw) * np.sin(pitch)])
        if v[1] <= 0:
            rewards[i] = 0
        else:
            t = d/v[1]
            contact = np.array([0.5*w*t**2+v[0]*t, -0.5*9.8*t**2+v[2]*t])
            distance = np.linalg.norm(contact)
            # 6.1cm rayon par tranche, 10 tranches, 122cm au total
            rewards[i] = max(0, int(10-distance//0.061))/10
    reward = rewards[0]
    return {"reward": reward, "rewards": rewards}

def sample_archery_situation(situations_bounds):
    s = {key: np.random.uniform(low=situations_bounds[key]["low"], high=situations_bounds[key]["high"]) for key in situations_bounds}
    return s

def sample_archery_situations(N, situations_bounds):
    bounds = {"low": np.array([x["low"] for x in situations_bounds.values()]), "high": np.array([x["high"] for x in situations_bounds.values()])}
    S = []
    for s_array in sample_Sobol(N, bounds):
        s = {}
        for i in range(len(bounds["low"])//2):
            s[f"d_{i}"] = s_array[2*i]
            s[f"w_{i}"] = s_array[2*i+1]
        S.append(s)
    return S

def sample_archery_command(command_bounds):
    return np.random.uniform(**command_bounds)


# -

class ArcheryEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, action_bounds, observation_bounds, reward_config):
        super(ArcheryEnv, self).__init__()
        self.action_bounds = action_bounds
        self.action_space = spaces.Box(low=np.zeros(2, np.float64), high=np.ones(2, np.float64), dtype=np.float64)
        
        self.observation_bounds = observation_bounds
        self.observation_space = spaces.Box(low=np.zeros(2, np.float64), high=np.ones(2, np.float64), dtype=np.float64)
        
        self.state = None 
        self.reward_config = reward_config
        self.log = []
        self.it = 0
        
    def reset(self, s=None):
        if s is None:
            self.state = np.random.random(2)
        else:
            self.state = s
        return self.state
    
    def get_log(self):
        return self.log
    
    def step(self, action, verbose=False):
        evaluation = eval_archery(unwrap(action, self.action_bounds), unwrap(self.state, self.observation_bounds), verbose=0)
        reward = evaluation["reward"]
        self.log.append({"a": action, "s": self.state, "r": reward, "it": self.it})
        self.it += 1 
        return None, reward, True, {"rewards": evaluation["rewards"]}

    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass 

# ## N-D Arm

# +
from math import cos, sin, pi, sqrt
import math 

class Arm:
    def __init__(self, lengths):
        self.n_dofs = len(lengths)
        self.lengths = np.concatenate(([0], lengths))
        self.joint_xy = []

    def fw_kinematics(self, p):
        assert(len(p) == self.n_dofs)
        p = np.append(p, 0)
        self.joint_xy = []
        mat = np.matrix(np.identity(4))
        for i in range(0, self.n_dofs + 1):
            m = [[cos(p[i]), -sin(p[i]), 0, self.lengths[i]],
                 [sin(p[i]),  cos(p[i]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            mat = mat * np.matrix(m)
            v = mat * np.matrix([0, 0, 0, 1]).transpose()
            self.joint_xy += [v[0:2].A.flatten()]
        return self.joint_xy[self.n_dofs], self.joint_xy

def arm(angles, task):
    angular_range = task[0] / len(angles)
    lengths = np.ones(len(angles)) * task[1] / len(angles)
    target = 0.5 * np.ones(2)
    a = Arm(lengths)
    # command in
    command = (angles - 0.5) * angular_range * math.pi * 2
    ef, _ = a.fw_kinematics(command)
    f = np.exp(-np.linalg.norm(ef - target))
    return f


# -

class ArmEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, dim, reward_config):
        super(ArmEnv, self).__init__()

        self.dim = dim
        self.action_space = spaces.Box(low=np.zeros(self.dim, np.float64), high=np.ones(self.dim, np.float64), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.zeros(2, np.float64), high=np.ones(2, np.float64), dtype=np.float64)
        
        self.state = None 
        self.reward_config = reward_config
        self.log = []
        self.it = 0
        
    def reset(self, s=None):
        if s is None:
            self.state = np.random.random(2)
        else:
            self.state = s
        return self.state
    
    def get_log(self):
        return self.log
    
    def step(self, action, verbose=False):
        reward = arm(action, self.state)
        self.log.append({"a": action, "s": self.state, "r": reward, "it": self.it})
        self.it += 1 
        return None, reward, True, {}

    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass 


# ## Talos Wall Reflex

# ### Evaluation

def eval_command_talos(c, s, verbose=0):
    batch = generate_batch_talos(c, s, verbose)
    evaluation = evaluate_batch_talos(batch, verbose)
    return evaluation


# +
def express_command_in_world(s, c):
    C, u = s["C"], s["u"]
    z = np.array([0,0,1])
    [x, y] = c      
    c = C + x * u + y * z
    return {
        "r_x": c[0],
        "r_y": c[1],
        "r_z": c[2],
        "r_n": s["n"],
    }

def generate_batch_talos(c, s, verbose=0):
    lh, rh = s["lh"], s["rh"]
    param = express_command_in_world(s, c)
    if verbose > 1:
        print("c: ", c, "s: ", s, "param:", param)
    name = "0"
    myconfig = {
        "CONTROLLER": {  
            "duration": 10.,
            "colision_shapes": s["collision_shapes"], 
            "damage_time": 4.,
            "reflex_time": 4.002,
            "use_baseline": False,
            "log_level": 1, 
            "use_reflex_trajectory": False,
            "update_contact_when_detected": True,
            "remove_rf_tasks": False,
            "reflex_arm_stiffness": 1.,
            "reflex_stiffness_alpha": 0.9999,
            "w_p": to_flist(c),
            "condition": s["condition"],
            "use_left_hand": False,
            "use_right_hand": True, 
            "use_training_early_stopping": True,
            "use_falling_early_stopping": True,
            "use_contact_early_stopping": True,
            "min_dist_to_wall": 0.07,
            "max_dist_to_wall": 0.08,
            "max_dist_to_target": 0.075,
            "check_model_collisions": True,
            "dt": 0.001,
        }, 
        "TASK": {
            "contact_rhand": {
                "x": float(param["r_x"]),
                "y": float(param["r_y"]),
                "z": float(param["r_z"]),
                "normal": to_flist(param["r_n"]),
                "kp": 30., 
            },
            "lh": {
                "weight": 1000.,
            },
            "rh": {
                "weight": 1000.,
            },
        },
        "BEHAVIOR": {
            "lh_shift": [float(lh[0]), float(lh[1]), float(lh[2]), 0., 0., 0.],
            "rh_shift": [float(rh[0]), float(rh[1]), float(rh[2]), 0., 0., 0.],
            "trajectory_duration": 4.,
        },
        "arg": {
            "recording": False,
            "video": verbose in [42, 73],
        },
    }
    param = test_default(name, actuator="spd", myconfig=myconfig)
    param["conditions"] = [Conditions["A"][c] for c in s["condition"]]
    return param


# +
def check_contact(collision_shape, normal, contact_pos):
    wall_pos = collision_shape[1][:3]
    wall_size = min(collision_shape[1][7]/2, collision_shape[1][8]/2) - 0.01
    diff = np.array(contact_pos) - np.array(wall_pos) 
    return 0.004 < diff.dot(normal) < 0.006 and np.linalg.norm(diff - diff.dot(normal) * normal) < wall_size

def evaluate_batch_talos(param, verbose=0):
    path = create_xp_folder()
    param["folder"] = path
    _ = run_online(param, verbose)
    dicts = {}
    jobs = []

    (name, dic) = loader(path, name="0", files=["end.dat", "tsid_q.dat", "config", "rcontact_pos.dat", "lcontact_pos.dat", "wall_contact.dat"])
    
    if verbose > 1:
        print("DICT: ", dic)

    if os.path.exists(path):
        subprocess.check_output(["rm", "-r", path])

    config = dic["config"]
    collision_shapes = config["CONTROLLER"]["colision_shapes"]
    r_n = np.array(config["TASK"]["contact_rhand"]["normal"])
    use_left, use_right = config['CONTROLLER']['use_left_hand'], config['CONTROLLER']['use_right_hand']
    rh_reached = extract_hand_contact(dic["data"], "r")
    stop_reason = dic["stop_reason"]
    if verbose > 1:
        print("stop_reason: ", stop_reason)
    if stop_reason[0] == "Recovered":
        r = extract_body_part_in_contact(dic["data"], side="r")
        if r == 0:
            ending = "Recovered" if (rh_reached["wall"] == [0]) and check_contact(collision_shapes[0], r_n, rh_reached["pos"]) else "bad_contact"
        else:
            ending = "Wrong_contact"
    else:
        ending = stop_reason[0]
    if verbose > 1:
        print("ending:",  ending)
    truth = ending in ["Recovered", "Unfallen"]
    command = np.array(config['CONTROLLER']["w_p"])
    evaluation = {
        "command": command,
        "q": dic["data"]['tsid_q'],
        "rh_reached": rh_reached,
        "truth": truth, 
        "stop_reason": stop_reason[0],
        "clean_contact": ending not in ["bad_contact", "Wrong_contact", "Unfallen", 'Auto_collision'],
        "fitness": stop_reason[1],
        "path": path, 
        "name": name, 
        "condition": config["CONTROLLER"]["condition"],
    }
    return evaluation


# -

# ### The Gym Env

class TalosWallReflexEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, action_bounds, observation_bounds, situations=None, situations_config=None):
        super(TalosWallReflexEnv, self).__init__()
        self.action_bounds = action_bounds
        self.action_space = spaces.Box(low=-1, high=1, shape=action_bounds["high"].shape, dtype=np.float64)
        self.observation_bounds = observation_bounds
        dic = {}
        for name, bounds in observation_bounds.items():
            dic[name] = spaces.Box(low=-1, high=1, shape=bounds["high"].shape, dtype=np.float64)
        spaces.Dict(dic)
        self.observation_space = spaces.Dict(dic)
        assert situations is not None or situations_config is not None
        self.situations_config = situations_config
        self.situations = situations
        self.state = None 
        
    def reset(self, verbose=0):
        if verbose:
            print("reseting")
        if self.situations is None:
            param = sample_new_params(N=1, situations_params=self.situations_config, n_proc=1, verbose=verbose)
            self.state = compute_situation_talos(param[0])
        else:
            self.state = self.situations[np.random.randint(len(self.situations))]
        return self.wrap_observation(self.state, verbose)

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass 
    
    def step(self, action, verbose=0):
        assert action.shape == (2,), f"the action should of shape {action.shape} should be of shape (5,)"
        param = {"s": self.state, "c": self.unwrap_action(action), "verbose": verbose}
        if verbose:
            print("param: ", param)
        evaluation = eval_command_talos(**param)
        reward = evaluation["fitness"]
        return None, reward, True, {}

    def wrap_observation(self, x, verbose=0):
        # from [L, H] -> [-1, 1, 0]
        s = {
            "d": np.array([x["d_robot"]]),
            "alpha": np.array([x["alpha_robot"]]),
            "q": np.array(x["q"])[6:],
        }
        if verbose:
            print("before wrapping: ", s)
        for name in s:
            H = self.observation_bounds[name]["high"]
            L = self.observation_bounds[name]["low"]
            s[name] = (2 * s[name] - (H + L)) / (H-L)
        if verbose:
            print("after wrapping: ", s)
        return s
    
    def get_unwrap_observation(self):
        # from [-1, 1, 0] -> [L, H]
        s = {
            "d": np.array([self.state["d_robot"]]),
            "alpha": np.array([self.state["alpha_robot"]]),
            "q": np.array(self.state["q"])[6:],
        }
        return s
    
    def unwrap_action(self, x):
        # from [-1, 1] -> [L, H]
        H = self.action_bounds["high"]
        L = self.action_bounds["low"]
        unwrapped_x = (H-L)/2 * x + (H+L)/2 
        return unwrapped_x
    
    def wrap_action(self, x):
       # from [L, H] -> [-1, 1]
        H = self.action_bounds["high"]
        L = self.action_bounds["low"]
        return (2 * x - (H + L)) / (H-L)
    
    def sample_action(self):
        return self.action_space.sample()
    
    def set_state(self, s):
        self.state = s 

# ## Talos Opening Door

def dict_command_to_vec(dic):
    vec = np.concatenate([val for val in dic.values()])
    assert len(vec) == 12
    return vec


def vec_command_to_dict(vec):
    assert len(vec) == 12
    dic = {
         "gripper_shift": vec[:3],
         "gripper_orientation": vec[3:6],
         "pull_handle_shift": vec[6:9],
         "pull_handle_rotation": vec[9:12],
    }
    return dic 


# ### The Gym Env

class TalosOpeningDoorEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, command_bounds, observation_bounds):
        super(TalosOpeningDoorEnv, self).__init__()
        self.command_bounds = command_bounds
        self.action_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float64)
        self.observation_bounds = observation_bounds
        self.observation_space = spaces.Box(low=-1, high=1, shape=observation_bounds["high"].shape, dtype=np.float64)
        self.state = None 
        self.log = []
        self.it = 0
        
    def reset(self, verbose=0):
        self.state = np.random.random(3)
        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass 
    
    def step(self, action, verbose=0):
        assert action.shape == (9,), f"the action of shape {action.shape} should be of shape (9,)"
        evaluation = eval_talos_door_opening(unwrap_door_opening_command(action, self.command_bounds), unwrap_door_opening_situation(self.state, self.observation_bounds), verbose=verbose)
        reward = evaluation["door_angle_after_pulling"]/(np.pi/2)
        self.log.append({"a": action, "s": self.state, "r": reward, "it": self.it})
        self.it += 1 
        return None, reward, True, {}
    
    def sample_action(self):
        return self.action_space.sample()
    
    def get_log(self):
        return self.log
    
    def set_state(self, s):
        self.state = s 


# # For PPO

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0, path="", steps_to_save=[]):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.path = path 
        self.step_to_save = steps_to_save

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        if self.num_timesteps in self.step_to_save:
            self.model.save(self.path + f"{self.num_timesteps:06d}")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.model.save(self.path + f"{self.num_timesteps:06d}")


# # NN Classifier/Regressor

# ## Create input

def create_input(input_name_list, sample):
    input_val_list = []
    for input_name in input_name_list:
        if "__" in input_name:
            l = input_name.split("__")
            if len(l) == 2:
                if l[0] == "uni":
                    input_val =  np.random.uniform(-1,1,int(l[1]))
                elif l[0] == "bool":
                    input_val =  np.random.randint(0,2,int(l[1]))
            else:
                print("bad argument ", input_name)
        else:
            input_val = sample[input_name]
        if type(input_val) in [float, np.float64]:
            input_val_list.append([input_val])
        elif type(input_val) in [list, np.ndarray]:
            input_val_list.append(input_val)
        else:
            print(f"Wrong input type! {input_val, type(input_val)}")
    return np.concatenate(input_val_list)


# ## Preprocess

# +
def pre_z_score(X, standardization, label):
    X_means = standardization[f"{label}_mean"] 
    X_stds = standardization[f"{label}_std"]
    X = (X - X_means) / X_stds
    return X

def pre_min_max(X, standardization, label):
    X_max = standardization[f"{label}_max"] 
    X_min = standardization[f"{label}_min"] 
    X = ( 2 * X - X_max - X_min) / (X_max - X_min)
    return X
    
def pre_tanh(X, standardization, label):
    X_means = standardization[f"{label}_mean"] 
    X_stds = standardization[f"{label}_std"]
    X = np.tanh(((X - X_means) / X_stds)) 
    return X

def pre_sigmoid(X, standardization, label):
    X_means = standardization[f"{label}_mean"] 
    X_stds = standardization[f"{label}_std"]
    X = 1 / (1 + np.exp(-((X - X_means) / X_stds)))
    return X

preprocessors = {
    "z_score": pre_z_score,
    "tanh": pre_tanh,
    "sigmoid": pre_sigmoid,
    "min_max": pre_min_max,
}

def preprocess(config, X, standardization):
    if config["input_preprocess"] in preprocessors:
        X = preprocessors[config["input_preprocess"]](X, standardization, "X")
    return X


# -

# ## Postprocess

# +
def post_sigmoid(Y, standardization, label):
    Y_means = standardization[f"{label}_mean"] 
    Y_stds = standardization[f"{label}_std"]
    Y = -np.log(1/Y-1) 
    Y = Y * Y_stds + Y_means
    return Y

def post_z_score(X, standardization, label):
    X_means = standardization[f"{label}_mean"] 
    X_stds =standardization[f"{label}_std"] 
    X = X * X_stds + X_means
    return X

def post_min_max(X, standardization, label):
    X_min = standardization[f"{label}_min"] 
    X_max =standardization[f"{label}_max"] 
    X = 0.5 * ( X * (X_max - X_min) + X_max + X_min) 
    return X 

postprocessors = {
    "z_score": post_z_score,
    "tanh": None,
    "sigmoid": post_sigmoid,
    "min_max": post_min_max,
}

def postprocess(config, Y, standardization):
    if config["output_preprocess"] in preprocessors:
        Y = postprocessors[config["output_preprocess"]](Y, standardization, "Y")
    return Y


# -

# ## Model 

class NN(nn.Module):

    def __init__(self, config):
        super(NN, self).__init__()
        layers = []
        layers_dim = [config["input_dim"]] + config["layers"]
        for i in range(1,len(layers_dim)):
            layers.append(nn.Linear(layers_dim[i-1], layers_dim[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config["dropout"]))
        layers.append(nn.Linear(layers_dim[-1], config["output_dim"]))
        if config["criterion"] == "MSELoss":
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.net(x)
        return x


def load_NN(path, config, use_cuda=False):
    model = NN(config)
    model.load_state_dict(torch.load(path)["model"])  
    if use_cuda:
        model.cuda()
    return model 


def predict(model, X, use_BCE):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        if use_BCE:
            pred = torch.sigmoid(pred)
    return pred 

# # Sampling 

# ## Boxes  

# ### generate boxes

def generate_boxes(compute_situation, sample_new_params, **kwargs):
    random_params = sample_new_params(**kwargs)
    boxes = []
    for dic in random_params.values():
        s = compute_situation(dic)
        boxes.append({"situation": s, "best_command": None, 'best_fitness': None, "solved": False, "best_situation": None}) 
    return boxes 


# ### update

def precompute_vectors(boxes, kind):
    if kind == "talos":
        precomputed_vectors = {
            "d": [s["situation"]["d"] for s in boxes],
            "alpha" : [s["situation"]["alpha"] for s in boxes],
            "lh" : [s["situation"]["lh"] for s in boxes],
            "rh" : [s["situation"]["rh"] for s in boxes],
            "condition" : [encode_condition(s["situation"]["condition"]) for s in boxes],
            }
    elif kind == "circle":
        precomputed_vectors = [s["situation"] for s in boxes]
    return precomputed_vectors


def reshape_distance(x):
    return 1 - np.exp(-x)


# +
def angle_distance(a1, a2):
    """
    distance between two angles (radians)
    result between 0 and 1
    """
    return (1 - (np.cos(a1) * np.cos(a2) + np.sin(a1) * np.sin(a2)))/2

def distance_wall(s1, s2):
    d1, alpha1 = s1["d"], s1["alpha"]
    d2, alpha2 = s2["d"], s2["alpha"]
    return reshape_distance(abs(d1-d2)) + angle_distance(alpha1, alpha2)

def distance_posture(s1, s2):
    lh1, rh1 = np.array(s1["lh"]), np.array(s1["rh"])
    lh2, rh2 = np.array(s2["lh"]), np.array(s2["rh"])
    return reshape_distance(np.linalg.norm(lh1-lh2) + np.linalg.norm(rh1-rh2))

def distance_condition(s1, s2):
    c1 = s1["condition"]
    c2 = s2["condition"]
    distance = 0
    for i in range(7):
        d1 = c1[np.argmax([str(i) in d for d in c1])] if (np.sum([str(i) in d for d in c1]) == 1) else None
        d2 = c2[np.argmax([str(i) in d for d in c2])] if (np.sum([str(i) in d for d in c2]) == 1) else None 
        distance += (d1 != d2)   
    return distance/7

def encode_condition(cond):
    code = []
    for i in range(7):
        d = cond[np.argmax([str(i) in d for d in cond])] if (np.sum([str(i) in d for d in cond]) == 1) else None
        if d is None:
            code.append(0)
        elif "Cut" in d:
            code.append(1)
        elif "Passive" in d:
            code.append(2)
        elif "Locked" in d:
            code.append(3)
        else:
            print("Unkown Condition in encode condition!!!")
    return code 


# +
def find_closest_box_vect_talos(boxes, s, precomputed_vectors):
    wall = reshape_distance(np.abs(s["d"]-precomputed_vectors['d'])) + reshape_distance(angle_distance(s["alpha"], precomputed_vectors['alpha']))
    posture = reshape_distance(np.linalg.norm(np.array(s["lh"])-precomputed_vectors['lh'], axis=1) + np.linalg.norm(np.array(s["rh"])-precomputed_vectors['rh'],axis=1))
    conditions = np.sum(np.not_equal(encode_condition(s["condition"]), precomputed_vectors['condition']), axis=1)/7
    distances = wall + conditions + posture 
    return boxes[np.argmin(distances)], distances

def find_closest_box_vect_gym(boxes, s, precomputed_vectors):
    distances = reshape_distance(np.linalg.norm(np.array(s)-precomputed_vectors, axis=1))
    return boxes[np.argmin(distances)], distances

def update_boxes(boxes, evaluations, precomputed_vectors, find_closest_box_vect):
    pos, neg, new_pos = 0, 0, 0
    true_pos, new_true_pos = 0, 0
    fitness = []
    for _, ev in evaluations.iterrows():
        b, distances = find_closest_box_vect(boxes, ev["situation"], precomputed_vectors)
        if ev["truth"]:
            true_pos += 1 
        fitness.append(ev["fitness"])
        new_pos += 1 if b["best_command"] is None else 0
        if b["best_command"] is None or ev["fitness"] > b["best_fitness"]:
            b["best_command"] = ev["command"]
            b["best_fitness"] = ev["fitness"]
            b["best_situation"] = ev["situation"]
            pos += 1 
            if not b["solved"] and ev["truth"]:
                new_true_pos += 1 
            b["solved"] = ev["truth"]
        else:
            neg += 1 
    return pos, neg, new_pos, true_pos, new_true_pos, np.mean(fitness)


# -

# ## Samplers

# +
def pick_n_random(l, n, replace=False):
    if isinstance(l, Iterable):
        l = np.array(l)
        if n > 1:
            return l[np.random.choice(range(len(l)), size=n, replace=replace)]
        else:
            return l[np.random.choice(range(len(l)), size=n, replace=replace)][0]
    else:
        raise NotIterableError(type(l))
        
def half_smart_sampler(boxes, batch_size, sigma, command_bounds, sample_situation, **kwargs):
    samples = []
    if sample_situation == sample_situation_talos:
        S = [sample_situation(**kwargs) for i in range(batch_size)] 
    elif sample_situation == sample_situation_gym:
        S = sample_situation(**kwargs)
    else:
        print("Error in sample_situation unknown")
        return None 
    
    for i in range(batch_size):
        C = []
        for b in pick_n_random(boxes, 2, replace=False):
            if b["best_command"] is None:
                c = np.random.uniform(low=command_bounds["low"], high=command_bounds["high"])
            else:
                c = b["best_command"]
            C.append(c)
        beta = np.random.random()
        c = merge_commands(C, beta, sigma, command_bounds)
        samples.append({"situation": S[i], "command": c})
    return samples

def random_sampler(boxes, batch_size, sigma, action_bounds, wall_bounds, sample_situation, **kwargs):
    samples = []
    for _ in range(batch_size):
        s = sample_situation(**kwargs)
        c = np.random.uniform(low=action_bounds["low"], high=action_bounds["high"])
        samples.append({"situation": s, "command": c})
    return samples


# -

def merge_commands(C, beta, sigma, bounds):
    return np.clip((1 - beta) * C[0] + beta * C[1] + np.random.normal(0, sigma), bounds["low"], bounds["high"]) 


# ## Evaluation 

# + active=""
# def eval_talos(samples, verbose=0):
#     batch, final_samples = generate_batch(samples, verbose=verbose)
#     return evaluate_batch(batch, final_samples, verbose=verbose)
# -

def eval_gym(samples, env, verbose=0):
    actions = [sample["command"] for sample in samples]
    _, r, _ , dic = env.step(actions)
    evaluations = []
    for i in range(len(samples)):
        evaluations.append({
        "command": samples[i]["command"],
        "situation": samples[i]["situation"],
        "truth": dic[i]["truth"],
        "treshold": dic[i]["threshold"],
        "fitness": r[i],
        })
    return pd.DataFrame(evaluations)


# # Args

# +
def unwrap(x, bounds):
    # [0, 1] -> [low, high]
    return x * (bounds["high"]-bounds["low"]) + bounds["low"]

def wrap(x, bounds):
    # [low, high] -> [0, 1] 
    return (x - bounds["low"]) / (bounds["high"]-bounds["low"]) 


# -

# ## Archery

# +
def compute_archery_situation(n, archery_state_bounds):
    res = {}
    bounds = {"low": [], "high": []}
    for i in range(n):
        for key, bound in archery_state_bounds.items():
            bounds["low"].append(bound["low"])
            bounds["high"].append(bound["high"])
    res["bounds"] = {"low": np.array(bounds["low"]), "high": np.array(bounds["high"])}
    res["dim"] = len(bounds["high"])
    return res

def compute_archery_command(n, archery_action_config, iso_sigma, line_sigma):
    action_bounds = {"low": np.concatenate([archery_action_config["low"] for _ in range(n)]), "high":  np.concatenate([archery_action_config["high"] for _ in range(n)])}
    return {"bounds": action_bounds, "iso_sigma": iso_sigma, "line_sigma": line_sigma, "dim": len(action_bounds["low"])}


# +
archery_state_bounds = {"d": {"low": 5, "high": 40}, "w": {"low": -10, "high": 10}}
archery_action_bounds = {"low": np.array([-np.pi/12, -np.pi/12]), "high": np.array([np.pi/12, np.pi/12])}

archery_reward_config = {
    "max_reward": 10,
}

arch_args = {"action_bounds": archery_action_bounds, 
             "observation_bounds": compute_archery_situation(1, archery_state_bounds)["bounds"], 
             "reward_config": archery_reward_config}
# -

# ## Arm

arm_args = {"dim": 10, "reward_config": {"max_reward": -0.2}}


# ## Talos Wall Reflex

# + active=""
# Xmin, Xmax = -2, 2  # hard mode for ppo (-2, 4)
# Zmin, Zmax = 0., 2.  # hard mode for ppo (0, 2)
# action_bounds = {"low": np.array([Xmin, Zmin]), "high": np.array([Xmax, Zmax])}
# observation_bounds = {
#     "d": {"low": np.array([0]), "high": np.array([2])},
#     "alpha": {"low": np.array([-np.pi]), "high": np.array([np.pi])},
#     "q": {"low": np.array(bounds_q32["low"])[6:], "high": np.array(bounds_q32["high"])[6:]},
# }
#
# condition_config = {
#     "proba_damaged_leg": 0.5, 
#     "proba_damaged_upper_body": 0., #0.25
#     "proba_amputation": 0.25,
#     "use_push": False,
#     "push_vec_bounds": {"low": [-100, -100, -100], "high": [100, 100, 100]},
#     "push_duration_bounds": {"low": 0., "high": 1.},
# }
#
# situations_config = {
#     "rh": {"low":  [-0.1, -0.4, -0.4], "high": [0.2, 0.1, 0.4]},
#     "lh": {"low":  [-0.1, -0.1, -0.4], "high": [0.2, 0.4, 0.4]},
#     "d" : {"low":  0.5, "high": 1.2},
#     "alpha" : {"low":  -np.pi/2, "high": 0.},
#     "condition_config": condition_config,
# }
#
# talos_kwargs = {"action_bounds": action_bounds, "observation_bounds": observation_bounds, "situations": None, "situations_config": situations_config}
# -

# ## Talos Opening Door

# +
def unwrap_door_opening_command(c, bounds):
    gripper_shift = unwrap(c[:3], bounds["gripper_shift"])
    [ox, oz] = unwrap(c[3:5], bounds["gripper_orientation"])
    gripper_orientation = [ox, 0, oz]
    pull_handle_shift = unwrap(c[5:8], bounds["pull_handle_shift"])
    rx = unwrap(c[8:9], bounds["pull_handle_rotation"])
    pull_handle_rotation = [rx, 0, 0]
    return {
         "gripper_shift": to_flist(gripper_shift),
         "gripper_orientation": to_flist(gripper_orientation),
         "pull_handle_shift": to_flist(pull_handle_shift),
         "pull_handle_rotation": to_flist(pull_handle_rotation),
        }

def unwrap_door_opening_situation(s, situation_bounds):
    x, y, rz = unwrap(s, situation_bounds)
    return to_flist([0, 0, rz, x, y ,0])


# +
shift = {"low": np.array([-0.3, -0.3, -0.3]), "high": np.array([0.3, 0.3, 0.3])}  # shift = {"low": [-0.3, -0.3, -0.3], "high": [0.3, 0.3, 0.3]}
orientation = {"low": np.array([-np.pi, np.pi/2]), "high": np.array([np.pi, np.pi/2])}  # orientation = {"low": [-np.pi, 0., np.pi/2], "high": [np.pi, 0., np.pi/2]}
rotation = {"low": np.array([-np.pi/2]), "high": np.array([np.pi/2])}  # rotation = {"low": [-np.pi/2, 0., 0.], "high": [np.pi/2, 0., 0.]}

talos_opening_door_command_bounds = {
     "gripper_shift": shift,
     "gripper_orientation": orientation,
     "pull_handle_shift": shift,
     "pull_handle_rotation": rotation,
}

talos_opening_door_action_bounds = {
    "low": np.concatenate([bounds["low"] for bounds in talos_opening_door_command_bounds.values()]), 
    "high": np.concatenate([bounds["high"] for bounds in talos_opening_door_command_bounds.values()]), 
}

talos_opening_door_situation_bounds = { 
    "low": np.array([0.8, -0.5, 3*np.pi/8]),   # [x, y, rz]
    "high": np.array([1.2, 0., 5*np.pi/8]),  # [x, y, rz]
}

talos_door_args = {
    "observation_bounds": talos_opening_door_situation_bounds,
    "command_bounds": talos_opening_door_command_bounds,
}
# -


