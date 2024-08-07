# # Import 

from base_talos_01 import *
from plot_01 import *

from numpy.random import default_rng
from collections import defaultdict

# # functions 

# +
command_bounds = compute_archery_command(1, archery_action_bounds, iso_sigma=0.01, line_sigma=0.2)["bounds"]
situation_bounds = compute_archery_situation(1, archery_state_bounds)["bounds"]

def eval_random_archery(n, seed):
    samples = []
    rng = default_rng(seed)
    for i in range(n):
        s = rng.random(2)
        c = rng.random(2)
        evaluation = eval_archery(unwrap(c, command_bounds), unwrap(s, situation_bounds), verbose=0)["reward"]
        samples.append({"c": c, "s": s, "it": i, "r": evaluation})
    return {"samples": samples}

# -

def eval_random_arm(n, seed):
    samples = []
    rng = default_rng(seed)
    for i in range(n):
        s = rng.random(2)
        c = rng.random(10)
        evaluation = arm(c, s)
        samples.append({"c": c, "s": s, "it":i, "r": evaluation})
    return {"samples": samples}


# +
rng = default_rng()

T = []
for _ in range(1000):
    c = rng.random(2)
    s = rng.random(2)
    t1 = time()
    eval_archery(unwrap(c, command_bounds), unwrap(s, situation_bounds), verbose=0)["reward"]
    t2 = time()
    T.append(t2-t1)
# -

print(np.mean(T))

# +
# arm = 0.6ms
# archery = 0.06ms
# -

# # run

# + active=""
# # one large in parallel
# n_proc = 20
# N = 100_000
# env = "archery"
# S = {i: (N//n_proc, i) for i in range(n_proc)}
# -

# many small in parallel
n_proc = 20
N = 100_000
env = "archery"
S = {i: (N, i) for i in range(n_proc)}

jobs = make_general_jobs(eval_random_arm if env == "arm" else eval_random_archery, S)

Res = general_master(jobs, n_processes=250, verbose=1)

# + active=""
# with open("/home/pal/notebooks/data/tmp/10B_random_archery.pk", "wb") as f:
#     pickle.dump(Res, f)
# -

# # Eval

arm_min = open_pickle("/home/pal/notebooks/data/PT-ME/arm_minimum.pk")
arm_max = open_pickle("/home/pal/notebooks/data/PT-ME/arm_maximum.pk")


def compute_qd_score(res, resolutions):
    qd_score = {}
    for step_size in resolutions:
        cells = defaultdict(list)
        if env == "arm":
            cells_min, cells_max = cells_min_max[step_size]
        for i in range(len(res["samples"])):
            key = tuple([int(x) for x in res["samples"][i]["s"]/step_size])
            cells[key].append(res["samples"][i]["r"])
        for key in cells.keys():
            if env == "arm":
                cells[key] = (np.max(cells[key])-cells_min[key])/(cells_max[key]-cells_min[key])
            else:
                cells[key] = np.max(cells[key])
        qd_score[step_size] = sum((x for x in cells.values())) * step_size**2
    return qd_score


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


# +
resolutions = []
for x in [1/int(x) for x in np.logspace(0, 2.5, 50)]:
    if x not in resolutions:
        resolutions.append(x)
if env == "arm":
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

data = [np.array([list(qd_score.values()) for qd_score in QD_scores.values()]).transpose()]
X = [1/np.array(list(QD_scores[0].keys()))**2]
labels = ["random"]

plot_with_spread(data, labels, X, cm.gist_rainbow)
plt.legend(fontsize=12)
plt.xscale("log")
plt.xlabel("Resolution (# uniformly spread centroids)")
plt.ylabel(f"QD-Score")
plt.title(f"Coverage Density {env}")
# -

save_pickle(f"/home/pal/notebooks/data/PT-ME/evaluations/20_random_{env}_100k_qd_score.pk", QD_scores)

# ## Times to cover

data = np.array([list(res["times_to_cover"].values()) for res in Res.values()]).transpose()
plot_with_spread([data], ["random"], [Trees.keys()])
#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("# Solutions Uniformly Spread")
plt.ylabel(f"# Samples to Cover {coverage}%")

# ## Density Coverage

N = 1000
data = np.array([res["density_coverage"] for res in Res.values()]).transpose()
plot_with_spread([data], ["random"], [Trees.keys()])
plt.xlabel("# Solutions Uniformly Spread")
plt.ylabel(f"% Covered")
plt.xscale("log")

# ## Coverage through time

N = 1000
data = np.array([[val for (n, t), val in res["coverage"].items() if n == N] for res in Res.values()]).transpose()
plot_with_spread([data], ["random"], [n_samples])
plt.xlabel("# Samples")
plt.ylabel(f"% Cells Filled ({N} Cells)")

# # MAP PLOT

R, S = [], []
for res in Res.values():
    for sample in res["samples"]:
        S.append(sample["s"])
        R.append(sample['r'])

indices = np.argsort(R)

# +

fig, ax = plt.subplots(figsize=(6*6, 6*5))
ax = plt.subplot2grid((5, 6), (0, 0))
situation = np.array(S)
rewards = np.array(R)
plt.scatter(situation[indices,0], situation[indices,1], s=1, c=rewards[indices], vmin=0, vmax=1)
plt.title("")
plt.axis("off")
plt.xlim((0,1.))  
plt.ylim((0,1.)) 
# -


