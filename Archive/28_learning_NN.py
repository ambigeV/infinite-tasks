# -*- coding: utf-8 -*-
# # Config

from base_talos_01 import *
from datetime import datetime
from numpy.random import default_rng 
import argparse
rng = default_rng()


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


# +
def foo(x, **k):
    return x

if not is_interactive():
    tqdm = foo
# -

if os.uname()[1] == "multivac":
    #datapath = "/home/tim/Experiences/Deeplearning/notebooks/data/" 
    datapath = "/home/pal/notebooks/data/"
elif os.uname()[1] == "haretis-42":
    datapath = "/home/haretis/Experiences/Deeplearning/notebooks/data/"
elif os.uname()[1] == "evo256":
    datapath = "/home/pal/notebooks/data/"
else: 
    datapath = "/home/pal/notebooks/data/"
    #datapath = "/home/tanne/Experiences/notebooks/data/"  # G5k

# +
font = {'size'   : 18}
mpl.rc('font', **font)

plt.rcParams["figure.figsize"] = (16, 9)


# -

# # NN 

# ## Loading data

# ### load data c2s

# +
def compute_input(sample, config):
    if "door_opening" in config["dataset_name"]:
        door_pos = sample["situation"]["door_pos"]
        command = np.concatenate([c for c in sample["command"].values()])
        inp = np.concatenate([door_pos[2:5],  # orientation, x, y 
                              command,
                             ])
    elif "archery" in config["dataset_name"]:
        inp = np.concatenate([ [val for val in sample["situation"].values()], sample["command"]])
    else:
        print("unknown dataset")
        return None
    return inp

def is_inside(x, bounds):
    return np.sum((bounds["high"]  >= x) * ( x  >= bounds["low"] )) == len(x)

def load_data_c2s(config, verbose=0):
    """
    construct training data for learning to classify the succesful position/command (c2s or p2s)
    """
    assert config["data_loader"] in ["load_data_c2s"]
    
    with open(datapath+config["dataset_name"], "rb") as f:
        data = pickle.load(f)
        
    if config["use_solution_bounding_box"]:
        commands = [sample["command"] for sample in data if sample[config["fitness"]] >= config["success_threshold"]]
        bounding_box = {"low": np.min(commands, axis=0), "high": np.max(commands, axis=0)}
        data = [x for x in data if is_inside(x["command"], bounding_box)]
        with open(config["logdir"]+f"/bounding_box.pk", 'wb') as f:
            pickle.dump(bounding_box, f)
        
    if verbose:
        print(f"data size: {len(data)}")
    In, Out = [], []
    condition_maps = []
    idx = 0
    
    if config["n_first_samples"] is None:
        samples_idx = rng.permutation(len(data))[:config["n_situations"]] 
        if verbose:
            print(f"take {config['n_situations']} random samples")
    else:
        n = min(config["n_first_samples"], len(data))
        print(f"take the first {n} samples")
        samples_idx = rng.permutation(n)
        
    for i in tqdm(samples_idx):
        sample = data[i]
        indices = []
        inp = compute_input(sample, config)
        In.append(inp)
        Out.append([sample[config["fitness"]] >= config["success_threshold"]])
        indices.append(idx)
        idx += 1 
        condition_maps.append({"indices": indices})
    return np.array(In), np.array(Out), condition_maps


# -

# ### Load Data Regression

def load_data_reg(config, verbose=0):
    with open(datapath+config["dataset_name"], "rb") as f:
        data = pickle.load(f)
        
    if verbose:
        print(f"data size: {len(data)}")
    
    In, Out = [], []
    condition_maps = []
    idx = 0
    
    for i in tqdm(rng.permutation(len(data))):
        sample = data[i]
        if sample["it"] <= config["n_first_samples"]:
            indices = []
            In.append(sample["situation"])
            Out.append(sample["command"])
            indices.append(idx)
            idx += 1 
            condition_maps.append({"indices": indices})
    return np.array(In), np.array(Out), condition_maps


# ### data generator

# +
def z_score_data(X, standardization, label):
    X_means = X.mean(dim=(0), keepdim=True)
    X_stds = X.std(dim=(0), keepdim=True) + 1e-10
    X = (X - X_means) / X_stds
    standardization[f"{label}_mean"] = X_means
    standardization[f"{label}_std"] = X_stds
    return X

def min_max_data(X, standardization, label):
    X_max = X.max(dim=(0), keepdim=True).values
    X_min = X.min(dim=(0), keepdim=True).values
    X = ( 2 * X - X_max - X_min) / (X_max - X_min)
    standardization[f"{label}_max"] = X_max
    standardization[f"{label}_min"] = X_min
    return X
    
def tanh_data(X, standardization, label):
    X_means = X.mean(dim=(0), keepdim=True)
    X_stds = X.std(dim=(0), keepdim=True) + 1e-10
    X = np.tanh(((X - X_means) / X_stds)) 
    standardization[f"{label}_mean"] = X_means
    standardization[f"{label}_std"] = X_stds
    return X

def sigmoid_data(X, standardization, label):
    X_means = X.mean(dim=(0), keepdim=True)
    X_stds = X.std(dim=(0), keepdim=True) + 1e-10
    X = 1 / (1 + np.exp(-((X - X_means) / X_stds)))
    standardization[f"{label}_mean"] = X_means
    standardization[f"{label}_std"] = X_stds
    return X

preprocessors_data  = {
    "z_score": z_score_data,
    "tanh": tanh_data,
    "sigmoid": sigmoid_data,
    "min_max": min_max_data,
}

def preprocess_data(config, X, Y):
    standardization = {}
    if config["input_preprocess"] in preprocessors:
        print("Preprocess the input with", config["input_preprocess"])
        X = preprocessors_data[config["input_preprocess"]](X, standardization, "X")
        
    if config["output_preprocess"] in preprocessors:
        print("Preprocess the output with", config["output_preprocess"])
        Y = preprocessors_data[config["output_preprocess"]](Y, standardization, "Y")
    return standardization, X, Y


# -

def data_generator(config, verbose=0):
    X, Y, condition_maps = load_data_reg(config, verbose) if config["data_loader"] == "load_data_reg" else load_data_c2s(config, verbose)
    if len(X) == 0:
        exit()
    print("Sending to Torch")
    X, Y = torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float)
    standardization, X, Y = preprocess_data(config, X, Y)
    with open(config["logdir"]+f"/standardization.pk", 'wb') as f:
        pickle.dump(standardization, f)
    return Variable(X), Variable(Y), condition_maps, standardization


# ### generate_train_test

def generate_train_test(config, permutation=None, training=True, verbose=0):
    X, Y, condition_maps, standardization = data_generator(config, verbose)
    if permutation is None:
        permutation = rng.permutation(len(condition_maps))
    if (config["test_ratio"]+config["val_ratio"]>=1):
        print("Empty training set")
    n_test, n_val = int(len(permutation)*config["test_ratio"]), int(len(permutation)*config["val_ratio"])
    condition_idx_train, condition_idx_val, condition_idx_test = permutation[n_test+n_val:], permutation[n_test:n_test+n_val], permutation[:n_test]
    sample_idx_train, sample_idx_test, sample_idx_val = [], [], []
    maps_train, maps_test, maps_val = [], [], []
    # to load previous dataset 
    # list_conditions = list(condition_maps.values())
    # 
    print("Splitting Dataset")
    for i in condition_idx_train:
        l = len(sample_idx_train)
        sample_idx_train = np.concatenate((sample_idx_train, condition_maps[i]["indices"])) # np.concatenate((sample_idx_train,list(list_conditions[i]["indices"].values())))
        condition_maps[i]["shuffled_indices"] = list(range(l, len(sample_idx_train)))
        maps_train.append(condition_maps[i])
    for i in condition_idx_test:
        l = len(sample_idx_test)
        sample_idx_test = np.concatenate((sample_idx_test,condition_maps[i]["indices"])) # np.concatenate((sample_idx_test,list(list_conditions[i]["indices"].values()))) # 
        condition_maps[i]["shuffled_indices"] = list(range(l, len(sample_idx_test)))
        maps_test.append(condition_maps[i])
    for i in condition_idx_val:
        l = len(sample_idx_val)
        sample_idx_val =  np.concatenate((sample_idx_val,condition_maps[i]["indices"])) # np.concatenate((sample_idx_val,list(list_conditions[i]["indices"].values()))) #
        condition_maps[i]["shuffled_indices"] = list(range(l, len(sample_idx_val)))
        maps_val.append(condition_maps[i])
    X_train, Y_train = X[sample_idx_train], Y[sample_idx_train]
    X_test, Y_test = X[sample_idx_test], Y[sample_idx_test]
    X_val, Y_val = X[sample_idx_val], Y[sample_idx_val]
    config["input_channels"] = X_train.shape[1] 
    if config["cuda"]:
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        X_val = X_val.cuda()
        Y_val = Y_val.cuda()
        X_test = X_test.cuda()
        Y_test = Y_test.cuda()
    X = {"train": X_train, "val": X_val, "test": X_test}
    Y = {"train": Y_train, "val": Y_val, "test": Y_test}
    maps = {"train": maps_train, "val": maps_val, "test": maps_test}
    with open(f'{config["logdir"]}/split.pk', "wb") as f:
        pickle.dump(maps, f)
    return X, Y, permutation, standardization, maps


# ## Model 

# ### Model NN

# +
activations = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
}

class NN(nn.Module):

    def __init__(self, config):
        super(NN, self).__init__()
        layers = []
        layers_dim = [config["input_dim"]] + config["layers"]
        activation = activations[config["activation"]]
        for i in range(1,len(layers_dim)):
            layers.append(nn.Linear(layers_dim[i-1], layers_dim[i]))
            layers.append(activation())
            layers.append(nn.Dropout(config["dropout"]))
        layers.append(nn.Linear(layers_dim[-1], config["output_dim"]))
        if config["criterion"] == "MSELoss":
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.net(x)
        return x


# -

# ### generate model

# +
class WrongCriterionError(Exception):
    pass 

def generate_model(config, Y_train):
    model = NN(config)
    if config["cuda"]:
        model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if config["criterion"] == "BCEWithLogitsLoss":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config["criterion"] == "WeightedBCEWithLogitsLoss":
        pos_weight = (len(Y_train)-torch.sum(Y_train).cpu())/torch.sum(Y_train).cpu()  # virtually balance the positive and negative examples 
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif config["criterion"] == "MSELoss":
        criterion = torch.nn.MSELoss()
    elif config["criterion"] == "WeightedMSELoss":
        def compute_weighted_mse_loss(Y_train):
            pos_weight = (len(Y_train)-torch.sum(Y_train).cpu())/torch.sum(Y_train).cpu()
            def weighted_mse_loss(input, target): 
                return ((1+(pos_weight-1)*target) * (input - target) ** 2).mean()
            return weighted_mse_loss
        criterion = compute_weighted_mse_loss(Y_train)
    else:
        raise WrongCriterionError
    return model, optimizer, criterion


# -

# ## Training functions

# ###  Classification evaluation

def binary_acc(y_pred, y_test, use_BCE=True):
    if use_BCE:
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
    else:
        y_pred_tag = torch.round(y_pred)
    y_test = torch.round(y_test)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc


def confusion_matrix(y_pred, y_test, use_BCE=True):
    if use_BCE:
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
    else:
        y_pred_tag = torch.round(y_pred)
    y_test = torch.round(y_test)
    cfm = np.zeros((2,2))
    for i, y in enumerate(y_test):
        if not torch.isnan(y_pred_tag[i]):
            cfm[int(y_pred_tag[i])][int(y)] += 1 
    return cfm


def MCC(cfm):
    """ Matthews correlation coefficient """
    TP = cfm[1,1]
    TN = cfm[0,0]
    FP = cfm[1,0]
    FN = cfm[0,1]
    denominator = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if denominator:
        mcc = (TP*TN-FP*FN)/denominator
    else:
        mcc = 0.
    return mcc


# ### evaluate

def evaluate(model, criterion, X, Y, situations, use_BCE=True, verbose=0):
    model.eval()
    res = {"loss": None, "acc": None, "mcc": None, "u": None, "c-u": None}
    with torch.no_grad():
        output = model(X)
        loss = criterion(output, Y)
        res["loss"] = -loss.cpu()
        if verbose > 0:
            print("o", output)
            print("t", Y) 
        if use_BCE:
            acc = binary_acc(output, Y, use_BCE)
            cfm = confusion_matrix(output, Y, use_BCE)
            mcc = MCC(cfm)
            res["acc"] = acc.cpu()
            res["mcc"] = mcc
        return res 


def compute_proba_pred(X, model, dropout=False, use_BCE=True):
    model.eval()
    y_pred_prob = 0
    with torch.no_grad():
        y_pred = model.forward(X)
        if use_BCE:
            y_pred_prob = torch.sigmoid(y_pred).cpu()
        else:
            y_pred_prob = y_pred.cpu()
    return y_pred_prob


# ### train

def train(epoch, model, optimizer, criterion, X_train, Y_train, verbose=0):
    model.train()
    batch_idx = 1
    batch_size = config["batch_size"]
    permutation = np.random.permutation(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        if i + batch_size > X_train.size(0):
            x, y = X_train[permutation[i:]], Y_train[permutation[i:]]
        else:
            x, y = X_train[permutation[i:(i+batch_size)]], Y_train[permutation[i:(i+batch_size)]]
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        if config['clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
        optimizer.step()


# ### learning

def learning(dic, X_train, Y_train, X_val, Y_val, X_test, Y_test, maps, verbose=0):
    config, model, optimizer, criterion = dic['config'], dic["model"], dic["optimizer"], dic["criterion"]
    use_BCE = type(dic["criterion"]) == torch.nn.BCEWithLogitsLoss
    if use_BCE:
        to_save = ["loss", "acc", "mcc"]
    else:
        to_save = ["loss"]
    assert np.sum([x not in to_save for x in config["final_save"]])  == 0
    
    tmp_model = deepcopy(model)
    bests = {key: {"value": -np.inf, "epoch": 0., "model": tmp_model.cpu().state_dict()} for key in to_save}
    logs = {key: {"train": [], "val": [], "test": []} for key in to_save}
    t = tqdm(range(config["epochs"]), ncols=180)
    for ep in t:
        train(ep, model, optimizer, criterion, X_train, Y_train, verbose)
        train_eval = evaluate(model, criterion, X_train, Y_train, maps["train"], use_BCE)
        val_eval = evaluate(model, criterion, X_val, Y_val, maps["val"], use_BCE, verbose=verbose)
        for key in to_save:
            if val_eval[key] > bests[key]["value"]:
                bests[key]["value"] = val_eval[key]
                bests[key]["epoch"] = ep
                tmp_model = deepcopy(model)
                bests[key]["model"] = deepcopy(tmp_model.cpu().state_dict())
            logs[key]["train"].append(train_eval[key])
            logs[key]["val"].append(val_eval[key])
        
        txt = ""
        for key in to_save:
            if key == "loss":
                txt += f"{key}:{train_eval[key]:.1e}\\{val_eval[key]:.1e}[{bests[key]['value']:.1e}|{bests[key]['epoch']}] "
            else:
                txt += f"{key}:{train_eval[key]*100:2.1f}%\\{val_eval[key]*100:2.1f}%[{bests[key]['value']*100:2.1f}%|{bests[key]['epoch']}] "
        if is_interactive():
            t.set_description(txt)
        
        if ep % config["save_frequency"] == 0 and ep > 0:
            save(config, to_save, logs, bests, ep)
        dic["bests"] = bests
        dic["log"] = logs
    if use_BCE and torch.sum(Y_val) == 0:
        print("Save last model.")
        # If there was no success in the validation save the last model
        for key in to_save:
            tmp_model = deepcopy(model)
            bests[key]["model"] = deepcopy(tmp_model.cpu().state_dict())
    save(config, config["final_save"], logs, bests, "final")


# ### save

def save(config, to_save, logs, bests, ep):
    with open(f'{config["logdir"]}/measures_ep{ep}.pk', "wb") as f:
        pickle.dump(logs, f)
    with open(f'{config["logdir"]}/config_ep{ep}.pk', "wb") as f:
        pickle.dump(dic['config'], f)
    for model_name in to_save:
        torch.save(bests[model_name], f'{config["logdir"]}/model_{model_name}_ep{ep}.trch')


# # Config

config = {
    # XP 
    "name": None, 
    "dataset_name": None, 
    "logdir": "",
    "save_frequency": 100, 

    # data 
    "input_preprocess": "z_score",  # in ["tanh", "min_max", "z_score"]
    "output_preprocess": "min_max", #"z_score",
    "n_situations": 0, 
    "n_first_samples": np.inf,
    "fitness": "reward",  # in ["max_handle_angle", "max_door_angle", "door_angle_after_pulling"]
    "success_threshold": 10,
    "data_loader": "load_data_reg",  # in ["load_data_p2s", "load_data_c2s", "load_data_c2p"]  
    "use_solution_bounding_box": False,
    ###################
    
    # Model
    "n_hidden": 64,  # 256 test
    "levels": 2,  # 3 test

    # Training
    "batch_size": 64,
    "epochs": 100, 
    "dropout": 0.,  # 0. test
    "clip": -1,
    'lr': 1e-4,
    'weight_decay': 0.01,  # 0. test
    "activation": "ReLU",  # in ["ReLU", "Tanh'],
    # c2s, p2s -> "WeightedBCEWithLogitsLoss"
    # c2p -> "MSELoss"
    "criterion": "MSELoss",  # in [BCEWithLogitsLoss, WeightedBCEWithLogitsLoss, WeightedMSELoss, MSELoss]
    "test_ratio": 0.,
    "val_ratio": 0.1,
    "final_save": ["loss"],
    "cuda": False,
}

# +
if config["data_loader"] in ["load_data_reg"]:
    if config["output_preprocess"] not in ["min_max"]:
        print("WARNING! the output isn't normalized! CORRECTED")
        config["output_preprocess"] = "min_max"
    
    if config["criterion"] != "MSELoss":
        print("WARNING! criterion not set to MSELoss! CORRECTED")
        config["criterion"] = "MSELoss"
        
    if "loss" not in config["final_save"]:
        print("WARNING! saved model should be loss! CORRECTED")
        config["final_save"] = ["loss"]

if config["data_loader"] in ["load_data_p2s", "load_data_c2s"]:
    if config["output_preprocess"] in ["tanh", "min_max", "z_score"]:
        print("WARNING! the output is normalized! CORRECTED")
        config["output_preprocess"] = ""
    
    if config["criterion"] not in ["BCEWithLogitsLoss", "WeightedBCEWithLogitsLoss"]:
        print("WARNING! criterion not set to WeightedBCEWithLogitsLoss! CORRECTED")
        config["criterion"] = "WeightedBCEWithLogitsLoss"
        
    if "mcc" not in config["final_save"]:
        print("WARNING! saved model should be mcc! CORRECTED")
        config["final_save"].append("mcc")
# -

# ## Parse args

# + active=""
# archery/dataset/regression/k{k}_n{n}.pk
# -

if config["dataset_name"] is None:
    env = "arm"
    i = 0
    res = 5000
    method = "MT-ME"
    config["dataset_name"] = f"PT-ME/training_sets/{env}/{method}_{i}_{res}.pk"
    config["env"] = env
if config["name"] is None:
    config["name"] = "MTMB_archery_reg_100k"

if not is_interactive():
    parser = argparse.ArgumentParser()
    # XP
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-name", type=str)
    parser.add_argument("-dataset_name", type=str)
    parser.add_argument("-env", type=str)
    # Data
    parser.add_argument("-input", nargs='+', type=str)
    parser.add_argument("-n_samples", type=int)
    parser.add_argument("-n_first_samples", type=int)
    parser.add_argument("-success_threshold", type=float)
    #Model
    parser.add_argument("-n_hidden", type=int)
    parser.add_argument("-levels", type=int)
    # Training
    parser.add_argument("-epochs", type=int)
    parser.add_argument("-dropout", type=float)
    parser.add_argument("-lr", type=float)
    parser.add_argument("-weight_decay", type=float)
    parser.add_argument("-val_ratio", type=float)
    parser.add_argument("-cuda", type=bool)
    parser.add_argument("-criterion", type=str)
    parser.add_argument("-activation", type=str)
    
    arguments = parser.parse_args()

    # XP
    if arguments.logdir is not None: config['logdir'] = arguments.logdir
    if arguments.name is not None: config['name'] = arguments.name
    if arguments.dataset_name is not None: config['dataset_name'] = arguments.dataset_name
    if arguments.env is not None: config['env'] = arguments.env
    # Data
    if arguments.input is not None: config['input'] = arguments.input 
    if arguments.n_samples is not None: config['n_samples'] = arguments.n_samples 
    if arguments.n_first_samples is not None: config['n_first_samples'] = arguments.n_first_samples 
    if arguments.success_threshold is not None: config['success_threshold'] = arguments.success_threshold
    #Model
    if arguments.n_hidden is not None: config['n_hidden'] = arguments.n_hidden
    if arguments.levels is not None: config['levels'] = arguments.levels
    # Training
    if arguments.epochs is not None: config['epochs'] = arguments.epochs
    if arguments.dropout is not None: config['dropout'] = arguments.dropout
    if arguments.lr is not None: config['lr'] = arguments.lr
    if arguments.weight_decay is not None: config['weight_decay'] = arguments.weight_decay
    if arguments.val_ratio is not None: config['val_ratio'] = arguments.val_ratio
    if arguments.cuda is not None: config['cuda'] = arguments.cuda
    if arguments.criterion is not None: config['criterion'] = arguments.criterion
    if arguments.activation is not None: config['activation'] = arguments.activation
    
    logdir = f"{config['logdir']}/{config['name']}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    config["logdir"] = logdir 
else:
    now = datetime.now()
    timestamp = now.strftime("%Y/%m/%d/%Hh%Mm%Ss")
    logdir = f"{datapath}{timestamp}/{config['name']}/{config['name']}_replicate_{0}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    config["logdir"] = logdir 
    print(logdir)

# ## Check config

config["layers"] = [config["n_hidden"] for _ in range(config["levels"])]
dic = {"config": config}

# # Loading data

print("Dataset to load: ", config["dataset_name"])
X, Y, permutation, standardization, maps = generate_train_test(config, verbose=True)
X_train, Y_train = X["train"], Y["train"]
X_val, Y_val = X["val"], Y["val"]
X_test, Y_test = X["test"], Y["test"]

if config["criterion"] in ["WeightedBCEWithLogitsLoss", "BCEWithLogitsLoss"]:
    print(f"training success: {Y_train.sum()/len(Y_train)*100:2.2f}%")
    print(f"validation success: {Y_val.sum()/len(Y_val)*100:2.2f}%")
else:
    print(f"training mean: {torch.mean(Y_train)}")
    print(f"validation mean: {torch.mean(Y_val)}")

print(len(X_val) + len(X_train) + len(X_test) )
if len(X_train) == 0:
    exit()

config["input_dim"] = X_train.shape[1] 
config["output_dim"] = Y_train.shape[1] 
config["input_dim"], config["output_dim"]

standardization

# # Actual Training

model, optimizer, criterion = generate_model(dic['config'], Y_train)
dic["model"] = model
dic["optimizer"] = optimizer
dic["criterion"] = criterion

learning(dic, X_train, Y_train, X_val, Y_val, X_test, Y_test, maps, verbose=0)

trained_dic = deepcopy(dic)

# + active=""
# exit()
# -

if is_interactive():
    logs = trained_dic["log"]
    bests = trained_dic["bests"]
    to_plot = ["loss"] if config["criterion"] == "MSELoss" else ["loss", "acc", "mcc"]
    N = len(to_plot)
    plt.subplots(figsize=(16, 4*N))
    for i, key in enumerate(to_plot):
        ax = plt.subplot2grid((N, 1), (i, 0), colspan=1)
        ax.plot(logs[key]["train"], lw=3, color="forestgreen", label=f"train {key}")
        ax.plot(logs[key]["val"], lw=3, color="royalblue", label=f"val {key}")
        m = min(np.min(logs[key]["val"]), np.min(logs[key]["train"]))
        M = max(np.max(logs[key]["val"]), np.max(logs[key]["train"]))
        ax.vlines(x=bests[key]["epoch"], ymin=m, ymax=M, lw=3, ls="--", color="royalblue", label=f"best {key}")
        plt.title(key)
        plt.legend()
        plt.grid()

# # Evaluate

model = NN(config)
model.load_state_dict(dic["bests"]["loss"]["model"])  

env = config["env"]

if env == "talos":
    n = 1000
    S = cvt(n, 3, rep=0)     
else:
    n = 10_000
    S = cvt(n, 2, rep=0) 

X  = torch.tensor(S, dtype=torch.float)
inp = preprocess(config, X, standardization)
output = predict(model, inp, use_BCE=False)
pred = postprocess(config, output, standardization)

if env == "arm":
    eval_f = lambda c, s : arm(c, s)
elif env == "archery":
    command_bounds = compute_archery_command(1, archery_action_bounds, line_sigma=0.2, iso_sigma=0.01)["bounds"]
    situation_bounds = compute_archery_situation(1, archery_state_bounds)["bounds"]
    eval_f = lambda c, s : eval_archery(unwrap(c, command_bounds), unwrap(s, situation_bounds), verbose=0)["reward"]
elif env == "talos":
    save_pickle(config["logdir"]+"/evaluations.pk", {"A": np.array(pred.cpu())})
    exit()
    eval_f = None

arm_min = open_pickle("/home/pal/notebooks/data/PT-ME/arm_min_10k.pk")
arm_max = open_pickle("/home/pal/notebooks/data/PT-ME/arm_max_10k.pk")

S_min_max = []
for i, s in enumerate(S):
    if env == "arm":
        S_min_max.append({"min": arm_min[i]["r"], "max": arm_max[i]["r"]})
    else:
        S_min_max.append({"min": 0, "max": 1})

R = []
for i in range(len(S)):
    c = pred[i]
    s = S[i]
    R.append((eval_f(c,s)-S_min_max[i]["min"])/(S_min_max[i]["max"]-S_min_max[i]["min"]))

save_pickle(config["logdir"]+"/evaluations.pk", {"R": R, "A": np.array(pred.cpu())})
