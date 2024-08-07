# # Config 

from base_talos_01 import * 
from plot_01 import *

if os.uname()[1] == "multivac":
    n_proc = 60
elif os.uname()[1] == "evo256":
    n_proc = 60
else:
    n_proc = os.cpu_count()-2

# # PPO

env = "talos"

if __name__=="__main__":
    n_envs = 50
    n_steps = 1
    if env == "arm":
        env_class = ArmEnv
        env_kwargs = deepcopy(arm_args)
    elif env == "archery":
        env_class = ArcheryEnv
        env_kwargs = deepcopy(arch_args)
    elif env == "talos":
        env_class = TalosOpeningDoorEnv
        env_kwargs = deepcopy(talos_door_args)

total_timesteps = 100_000
steps_to_save = [1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]

# + active=""
# path = "/home/pal/notebooks/data/"
# logdir = os.path.join(path, datetime.datetime.now().strftime("%Y/%m/%d/%Hh%Mm%Ss"))
# os.makedirs(logdir)
# -

logdir = "/home/pal/notebooks/data/2023/11/09/09h56m07s/"

rep = 0
while os.path.exists(logdir+f"/{rep}"):
    rep += 1
print(rep)

if __name__=="__main__":
    for i in [rep]:
        train_env = make_vec_env(env_class, n_envs=n_envs, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)  # //
        #train_env = env_class(**env_kwargs)  # sequential
        print("Train env done")
        policy ="MlpPolicy" if type(train_env.observation_space) == gym.spaces.box.Box else "MultiInputPolicy" 
        run_logdir = logdir + f"/{i}"
        os.makedirs(run_logdir, exist_ok=True)
        os.makedirs(run_logdir + "/models/", exist_ok=True)
        os.makedirs(run_logdir + "/eval/", exist_ok=True)
        print("tensorboard --logdir ", run_logdir)
        ppo_model = PPO(policy, train_env, verbose=0,  n_steps=n_steps, batch_size=n_envs, tensorboard_log=run_logdir)
        checkpoint_callback = CustomCallback(path=run_logdir + "/models/" , steps_to_save = steps_to_save)
        #eval_callback = EvalCallback(eval_env, best_model_save_path=logdir, log_path=logdir+"/eval/", eval_freq=20, n_eval_episodes=100, deterministic=True, render=False)
        ppo_model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=[checkpoint_callback])
        save_pickle(run_logdir + "/logs.pk", train_env.env_method("get_log"))

# # analyse

# Arm
#path = "/home/pal/notebooks/data/2023/11/02/12h44m48s/"  # 20x arm 
path = "/home/pal/notebooks/data/PT-ME/datasets/PPO/arm/"
# Archery
#path = '/home/pal/notebooks/data/2023/11/03/08h16m34s/'   # 20x archery
path = "/home/pal/notebooks/data/PT-ME/datasets/PPO/archery/"
# Talos 
#path = "/home/pal/notebooks/data/2023/11/09/09h56m07s/"  # test talos 
path = "/home/pal/notebooks/data/PT-ME/datasets/PPO/talos/"

Res = {}
for folder in os.listdir(path):
    logs = open_pickle(path+"/"+folder+"/logs.pk")
    solutions = []
    for i in range(len(logs)):
        for log in logs[i]:
            log["it"] = log["it"] * n_envs + i
            solutions.append(log)
    Res[int(folder)] = {"samples": solutions}

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
        qd_score[step_size] = sum((x for x in cells.values())) * step_size**len(key)
    return qd_score


# +
env_dim = 3 if env == "talos" else 2

resolutions = []
for x in [1/int(x) for x in np.logspace(0, 5/env_dim, 50)]:
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
QD_scores = general_master(jobs, 50, 1)

# +
data, labels, X = [], [], []
line_styles = []

data = [np.array([list(qd_score.values()) for qd_score in QD_scores.values()]).transpose()]
X = [1/np.array(list(QD_scores[0].keys()))**env_dim]
labels = ["ppo"]

plot_with_spread(data, labels, X, cm.gist_rainbow)
plt.legend(fontsize=12)
plt.xscale("log")
plt.xlabel("Resolution (# uniformly spread centroids)")
plt.ylabel(f"QD-Score")
plt.title(f"Coverage Density {env}")
# -

save_pickle(f"/home/pal/notebooks/data/PT-ME/evaluations/10_ppo_{env}_100k_qd_score.pk", QD_scores)

# # Plot

len(Res[2]["samples"])

S = np.array([s["s"] for s in Res[0]["samples"]])
R = np.array([s["r"] for s in Res[0]["samples"]])

indices = np.argsort(R)

fig, ax = plt.subplots(figsize=(12, 4))
labels = ["x", "y", "rz"]
situation = np.array(S)
rewards = np.array(R)
for i in range(3):
    ax = plt.subplot2grid((1, 3), (0, i))
    x,y = [(0,1), (0,2), (1,2)][i]
    plt.scatter(situation[indices, x], situation[indices, y], s=50, c=rewards[indices], vmin=0, vmax=1.2840939118540298)
    
    plt.title(f"({labels[x]}, {labels[y]})")
    plt.axis("off")
    plt.xlim((0,1.))  
    plt.ylim((0,1.)) 
    plt.gca().set_aspect('equal')
#savefig("talos_ppo")

# # Inference evaluation 

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


# ## load 

env = "talos"

if env == "arm":
    path = "/home/pal/notebooks/data/PT-ME/datasets/PPO/arm/"
elif env == "archery":
    path = "/home/pal/notebooks/data/PT-ME/datasets/PPO/archery/"
elif env == "talos":
    path = "/home/pal/notebooks/data/PT-ME/datasets/PPO/talos/"

if env == "arm":
    eval_f = lambda c, s : arm(c, s)
elif env == "archery":
    command_bounds = compute_archery_command(1, archery_action_bounds, line_sigma=0.2, iso_sigma=0.01)["bounds"]
    situation_bounds = compute_archery_situation(1, archery_state_bounds)["bounds"]
    eval_f = lambda c, s : eval_archery(unwrap(c, command_bounds), unwrap(s, situation_bounds), verbose=0)["reward"]
elif env == "talos":
    eval_f = None

PPOs = {}
for logdir in os.listdir(path):
    model = PPO.load(path+logdir+"/models/100000")
    PPOs[logdir] = {"model": model}

# +
'log_std',

'mlp_extractor.policy_net.0.weight',
'mlp_extractor.policy_net.0.bias',
'mlp_extractor.policy_net.2.weight',
'mlp_extractor.policy_net.2.bias',

'mlp_extractor.value_net.0.weight',
'mlp_extractor.value_net.0.bias',
'mlp_extractor.value_net.2.weight',
'mlp_extractor.value_net.2.bias',

'action_net.weight',
'action_net.bias',

'value_net.weight',
'value_net.bias'

# -

model.get_parameters()["policy"]["mlp_extractor.policy_net.2.weight"].shape

if env == "talos":
    n = 1000
    S = cvt(n, 3, rep=0)     
else:
    n = 10_000
    S = cvt(n, 2, rep=0) 

arm_min = open_pickle("/home/pal/notebooks/data/PT-ME/arm_min_10k.pk")
arm_max = open_pickle("/home/pal/notebooks/data/PT-ME/arm_max_10k.pk")

S_min_max = []
for i, s in enumerate(S):
    if env == "arm":
        S_min_max.append({"min": arm_min[i]["r"], "max": arm_max[i]["r"]})
    else:
        S_min_max.append({"min": 0, "max": 1})

for ppo in PPOs.values():
    A, _ = ppo["model"].predict(S, deterministic=True)
    ppo["A"] = A

if env == "talos":
    C = {i: [ppo["A"][i] for ppo in PPOs.values()] for i in range(len(S))}
    evaluations = eval_talos_door_opening_batch(C, S, verbose=1)
    for i, ppo in enumerate(PPOs.values()):
        ppo["R"] = evaluations[i]
else:
    for ppo in tqdm(PPOs.values()):
        R = []
        for i in range(len(S)):
            c = ppo["A"][i]
            s = S[i]
            R.append((eval_f(c,s)-S_min_max[i]["min"])/(S_min_max[i]["max"]-S_min_max[i]["min"]))
        ppo["R"] = R

# ## Plot 

QD_score = [np.mean(ppo["R"]) for ppo in PPOs.values()]

to_save = {key: {"R": dic["R"], "A": dic["A"]} for key, dic in PPOs.items()}
save_pickle("/home/pal/notebooks/data/PT-ME/inference/PPO/talos.pk", to_save)

f"QD-score: {np.median(QD_score):1.3f} [{np.quantile(QD_score, 0.25):1.3f}, {np.quantile(QD_score, 0.75):1.3f}]"

S[0]

for i in range(20):
    plt.subplot2grid((4,5), (i//5, i%5))
    plt.scatter(S[:, 0], S[:, 1], c=PPOs[f"{i}"]["R"], vmin=0, vmax=1)
    #plt.title(f"{env}")
    plt.axis("off")
    plt.xlim((0,1.))  
    plt.ylim((0,1.)) 
    #plt.colorbar()


