# # Import 

from base_talos_01 import *
from plot_01 import *

from numpy.random import default_rng
import cma

# # functions 

command_bounds = compute_archery_command(1, archery_action_bounds, iso_sigma=0.01, line_sigma=0.2)["bounds"]
situation_bounds = compute_archery_situation(1, archery_state_bounds)["bounds"]


def eval_env(env, c, s):
    if env == "archery":
        return eval_archery(unwrap(c, command_bounds), unwrap(s, situation_bounds), verbose=0)["reward"]
    else:
        return arm(c, s)


def eval_env_batch(env, C, s):
    return [eval_env(env, c, s) for c in C]


def clip_logs_to_budget(logs, budget):
    N = np.sum([np.sum([len(c) for c in l["commands"]]) for l in logs[:-1]])
    final_run = {"commands": [], "rewards": []}
    max_val, max_x = -np.inf, None
    for i in range(len(logs[-1]["commands"])):
        commands, rewards = [], []
        for j in range(len(logs[-1]["commands"][i])):
            r = logs[-1]["rewards"][i][j]
            c = logs[-1]["commands"][i][j]
            if r > max_val:
                max_val = r
                max_x = c
            commands.append(c)
            rewards.append(r)
            N += 1 
            if N == budget:
                break
        final_run["commands"].append(commands)
        final_run["rewards"].append(rewards)
        if N == budget:
            break
    final_run["r"] = max_val
    final_run["c"] = max_x
    final_run["s"] = logs[-1]["s"]
    logs[-1] = final_run


def flatten_logs(logs, env):
    tmp, max_it = [], 0
    global_it = 0
    for k, log in enumerate(logs):
        local_it = 0
        tmp.append([])
        for i in range(len(log["commands"])):
            for j in range(len(log["commands"][i])):
                tmp[k].append({"s": log["s"], "r": log["rewards"][i][j], "c": log["commands"][i][j], "sequential_it": global_it})
                local_it += 1
                global_it += 1 
        max_it = max(max_it, local_it) 
        
    samples = []
    it = 0
    for i in range(max_it):
        for k in range(len(tmp)):
            if i < len(tmp[k]):
                tmp[k][i]["parallel_it"] = it
                samples.append(tmp[k][i])
                it += 1 
    res = {"samples": samples}
    return res 


def estimate_command2success(s, env, use_ftarget=False, maxfevals=np.inf):
    dim = 2 if env == "archery" else 10
    config = {
        "bounds": [[0]*dim, [1]*dim],
        "verbose": -9,
        "maxiter": maxfevals,
    }
    es = cma.CMAEvolutionStrategy(dim * [0.5], 0.5, config)  # 2 = dim of input
    
    commands, rewards = [], []
    
    while not es.stop():  # has a max iter by default
        C = es.ask()
        commands.append(C)
        R = eval_env_batch(env, C, s)
        es.tell(C, -np.array(R))
        rewards.append(R)
    # es.result[0] = best solution, es.result[1] = value of the best solution
    return {"c": es.result[0], "r": -es.result[1], "commands": commands, "rewards": rewards}


def cma_es_while_budget(env, budget, seed, use_ftarget=False, maxfevals=np.inf, verbose=False):
    N = 0
    if verbose:
        t = tqdm(total=budget, smoothing=0.)
    logs = []
    rng = default_rng(seed)
    while N < budget:
        s = rng.random(2)
        res = estimate_command2success(s, env, use_ftarget=use_ftarget, maxfevals=maxfevals)
        res["s"] = s
        logs.append(res)
        n = np.sum([len(c) for c in res["commands"]])
        N += n
        if verbose:
            t.update(n)
    if verbose:
        t.close()
    clip_logs_to_budget(logs, budget)
    return flatten_logs(logs, env) 


# # run

# many small in parallel
n_rep = 1
N = 100_000
env = "archery"
Max_evals = [10, np.inf]
S = {(i,maxfevals): (env, N, i, True, maxfevals) for i in range(n_rep) for maxfevals in Max_evals}

jobs = make_general_jobs(cma_es_while_budget, S)

Res = general_master(jobs, n_processes=60, verbose=1)

# # Eval

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
        cells_min, cells_max = cells_min_max[step_size]
        for i in range(len(res["samples"])):
            key = tuple([int(x) for x in res["samples"][i]["s"]/step_size])
            cells[key].append(res["samples"][i]["r"])
        for key in cells.keys():
            cells[key] = (np.max(cells[key])-cells_min[key])/(cells_max[key]-cells_min[key])
        qd_score[step_size] = sum((x for x in cells.values())) * step_size**2
    return qd_score


# +
resolutions = []
for x in [1/int(x) for x in np.logspace(0, 2.5, 50)]:
    if x not in resolutions:
        resolutions.append(x)

cells_min_max = {}
for x in resolutions:
    cells_min_max[x] = compute_cell_min_max(x)
        
args = {}
for key, res in Res.items():
    args[key] = (res, resolutions)
        
jobs = make_general_jobs(compute_qd_score, args)
QD_scores = general_master(jobs, 250, 1)

# +
data, labels, X = [], [], []
line_styles = []

data = [ np.array([list(QD_scores[(i,n)].values()) for i in range(n_rep)]).transpose() for n in Max_evals]
X = [1/np.array(list(QD_scores[(0, 10)].keys()))**2] * len( Max_evals)
labels = Max_evals

plot_with_spread(data, labels, X, cm.gist_rainbow)
plt.legend(fontsize=12)
plt.xscale("log")
plt.xlabel("Resolution (# uniformly spread centroids)")
plt.ylabel(f"QD-Score")
plt.title(f"Coverage Density {env}")
# -

save_pickle(f"/home/pal/notebooks/data/PT-ME/evaluations/cma_es_{env}_qd_score.pk", QD_scores)


# # Fitness Bounds

def cma_es_arm(s, minimize):
    config = {
        "bounds": [[0]*10, [1]*10],
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(10 * [0.5], 0.5, config)  # 2 = dim of input
    coef = 1 if minimize else -1
    while not es.stop():  # has a max iter by default
        C = es.ask()
        R = eval_env_batch("arm", C, s)
        es.tell(C, coef*np.array(R))
    return {"c": es.result[0], "r": coef * es.result[1]}


# + active=""
# X = np.linspace(0, 1, 317)
# X = (X[1:]+X[:-1])/2
# S = {}
# for i, x in enumerate(X):
#     for j, y in enumerate(X):
#         S[(i, j)] = ([x, y], False)
# -

n = 10_000
S = cvt(n, 2, rep=0)
args = {}
for i, s in enumerate(S):
    args[i] = (s, True)

jobs = make_general_jobs(cma_es_arm, args)

Res = general_master(jobs, n_processes=50, verbose=1)

minimum = {}
for key, dic in Res.items():
    s, _ = S[key]
    minimum[key] = {"r": dic["r"], "c": dic["c"], "s": s}

# + active=""
# save_pickle(f"/home/pal/notebooks/data/PT-ME/arm_min_10k.pk", minimum)
# -

len(minimum)

# # plot map

Res.keys()

S = np.array([s["s"] for s in Res[(0, np.inf)]["samples"]])
R = np.array([s["r"] for s in Res[(0, np.inf)]["samples"]])

indices = np.argsort(R)

fig, ax = plt.subplots(figsize=(6*6, 6*5))
ax = plt.subplot2grid((5, 6), (0, 0))
situation = np.array(S)
rewards = np.array(R)
plt.scatter(situation[indices,0], situation[indices,1], s=1, c=rewards[indices], vmin=0, vmax=1)
plt.title("")
plt.axis("off")
plt.xlim((0,1.))  
plt.ylim((0,1.)) 


