from base_talos_01 import *

logdir = create_save_folder(rootpath="/home/pal/notebooks/data/archery/NN/")
print(logdir)


# + active=""
# /home/pal/notebooks/data/archery/NN/2024/01/28/10h01m11s the 300 MR Arm distillations
# -

def make_command(values):
    c = ["python", "/home/pal/notebooks/2024/01/28_learning_NN.py"]
    for key, val in values.items():
        c.append("-" + key)
        c.append(val)
    return c


batch = []
for i in range(20):
    for env in ["archery"]:   
        for method in ["MT-ME"]:
            for res in [5000] if method == "MT-ME" else [int(x) for x in np.logspace(0, 5, 20)[3:-2]] :
                values = {
                    "dataset_name": f"PT-ME/training_sets/{env}/{method}_{i}_{res}.pk",
                    "name": f"{env}_{method}_{res}_{i}",
                    "env": env,
                    "epochs": "100",
                    "n_first_samples": str(100_000),
                    "logdir": logdir,
                }
                batch.append(make_command(values))

len(batch)

for c in tqdm(batch, smoothing=0.):
    res = subprocess.check_output(c)

# # Analyse

if env == "talos":
    Actions = {}
    for reso in [int(x) for x in np.logspace(0, 5, 20)[3:-2]]:
        for i in range(10):
            data = open_pickle(logdir+"/"+f"{env}_{method}_{reso}_{i}"+"/evaluations.pk")
            Actions[(i, reso)] = data["A"]
    save_pickle(f"/home/pal/notebooks/data/PT-ME/inference/neural_network/acions_{env}.pk", Actions)

QD_score = []
for reso in [5000] if method == "MT-ME" else [int(x) for x in np.logspace(0, 5, 20)[3:-2]]:
    res = []
    for i in range(20):
        data = open_pickle(logdir+"/"+f"{env}_{method}_{reso}_{i}"+"/evaluations.pk")
        res.append(np.mean(data["R"]))
    QD_score.append(res)



save_pickle(f"/home/pal/notebooks/data/PT-ME/inference/neural_network/{env}_{method}.pk", QD_score)


