import random
import time

import gpytorch
import torch
import numpy as np
from models import VanillaGP, ModelList, next_sample, device, model_lcb, evaluation
from problems import get_problem
from utils import plot_hist, cvt, plot_tot, plot_box, \
    plot_details, plot_convergence, plot_iteration_convergence, \
    debug_tot, debug_each, debug_hist, ax_plot_iteration_convergence
import matplotlib.pyplot as plt
from model_problems import ec_alg_moo
from scipy.stats import qmc

# method_name_list = ["ind_gp", "context_gp", "unified_gp",
#                     "context_gp_plain", "unified_gp_plain",
#                     "fixed_context_gp", "inverse_context_gp_plain",
#                     "inverse_context_gp_inner_plain",
#                     "forward_inverse_context_gp_inner_plain",
#                     "forward_inverse_context_gp_plain"]

# method_name_list = ["ind_gp", "fixed_context_gp",
#                     "context_gp_plain", "forward_inverse_context_gp_plain"]

# method_name_list = ["10_50_ind_gp_ard",
#                     "10_50_fixed_context_gp_ard"]
method_name_list = ["10_50_ind_gp_hetero",
                    "10_50_fixed_context_gp_hetero"]
                    # "20_50_fixed_context_gp_fixed"]
                    # "20_50_fixed_context_inverse_random_cut_gp"]
                    # "20_50_context_gp_plain",
                    # "20_50_context_inverse_active_gp_plain"]
                    # "20_50_fixed_context_gp_fixed"]
                    # "10_20_fixed_context_gp_fixed"]
                    # "10_20_fixed_context_gp_fixed"]
                    # "10_1_fixed_context_gp"]
                    # "10_50_fixed_context_gp_smooth_hetero"]
                    # "10_50_fixed_context_inverse_cut_gp_hetero",
                    # "10_50_fixed_context_gp_smooth_hetero"]

# method_name_list = ["10_1_ind_gp", "10_1_fixed_context_gp", "10_1.0_forward_inverse_fixed_context_gp_plain"]

# method_name_list = ["inverse_context_gp_plain",
#                     "inverse_context_gp_inner_plain",
#                     "forward_inverse_context_gp_inner_plain",
#                     "context_gp"]

# problem_name = "sep_arm"
# problem_name = "linear_rastrigin_20"
problem_name = "linear_ackley_high"
# problem_name = "linear_ackley"
dim_size = 4
task_params = 5 # Default value should be 2
direct_name = "{}_result_{}_{}".format(problem_name, dim_size, task_params)
# direct_name = "{}_result_{}".format(problem_name, dim_size)
task_number = 20
beta_ucb = 50
# direct_name = "result_physics"


def configure_problem(problem_name):
    params = dict()
    params["ind_size"] = task_number
    params["tot_init"] = 200
    params["tot_budget"] = 2000
    # params["tot_budget"] = 300
    params["aqf"] = "ucb"
    params["train_iter"] = 200
    params["test_iter"] = 30
    params["switch_size"] = task_number * 2
    params["problem_name"] = problem_name
    params["n_obj"] = 1
    params["n_dim"] = dim_size
    params["n_task_params"] = task_params
    return params


def configure_method(method_name):
    params = dict()
    params["if_ind"] = False
    params["if_cut"] = False
    params["if_inverse"] = False
    params["if_forward"] = False
    params["if_unified"] = False
    params["if_cluster"] = True
    params["if_inner_cluster"] = False
    params["if_fixed"] = False
    params["if_smooth"] = False
    params["if_random"] = False
    params["if_active"] = False
    params["if_switch"] = False
    params["method_name"] = method_name

    if method_name == "ind_gp":
        params["if_ind"] = True

    if method_name == "ind_gp_20":
        params["if_ind"] = True

    if method_name == "unified_gp":
        params["if_unified"] = True

    if method_name == "context_gp_plain":
        params["if_cluster"] = False

    if method_name == "forward_inverse_context_gp_plain":
        params["if_inner_cluster"] = True
        params["if_forward"] = True
        params["if_cluster"] = False
        params["if_inverse"] = True

    if method_name == "forward_inverse_fixed_context_gp_plain":
        params["if_inner_cluster"] = True
        params["if_forward"] = True
        params["if_cluster"] = False
        params["if_inverse"] = True
        params["if_fixed"] = True

    if method_name == "forward_inverse_context_gp_more_plain":
        params["if_inner_cluster"] = True
        params["if_forward"] = True
        params["if_cluster"] = False
        params["if_inverse"] = True

    if method_name == "inverse_context_gp_plain":
        params["if_inner_cluster"] = True
        params["if_cluster"] = False
        params["if_inverse"] = True

    if method_name == "forward_inverse_context_gp_inner_plain":
        params["if_forward"] = True
        params["if_cluster"] = False
        params["if_inverse"] = True

    if method_name == "inverse_context_gp_inner_plain":
        params["if_cluster"] = False
        params["if_inverse"] = True

    if method_name == "unified_gp_plain":
        params["if_unified"] = True
        params["if_cluster"] = False

    if method_name == "fixed_context_gp_smooth":
        params["if_fixed"] = True
        params["if_smooth"] = True

    if method_name == "fixed_context_gp":
        params["if_fixed"] = True

    if method_name == "fixed_switch_context_gp":
        params["if_fixed"] = True
        params["if_switch"] = True
        params["if_inverse"] = True

    if method_name == "fixed_context_inverse_cut_gp":
        params["if_fixed"] = True
        params["if_inverse"] = True
        params["if_cut"] = True

    if method_name == "fixed_context_inverse_random_cut_gp":
        params["if_fixed"] = True
        params["if_inverse"] = True
        params["if_cut"] = True
        params["if_random"] = True

    if method_name == "fixed_context_gp_20":
        params["if_fixed"] = True

    if method_name == "context_inverse_active_gp_plain":
        params["if_inverse"] = True
        params["if_active"] = True
        params["if_cluster"] = False

    return params


def solver(problem_params, method_params, trial):
    # Offer parameters of method and problem
    # Return the results for one trial

    # Fetch method parameters
    if_ind = method_params["if_ind"]
    if_cut = method_params["if_cut"]
    if_random = method_params["if_random"]
    if_smooth = method_params["if_smooth"]
    if_inverse = method_params["if_inverse"]
    if_forward = method_params["if_forward"]
    if_unified = method_params["if_unified"]
    if_fixed = method_params["if_fixed"]
    if_cluster = method_params["if_cluster"]
    if_active = method_params["if_active"]
    if_inner_cluster = method_params["if_inner_cluster"]
    if_switch = method_params["if_switch"]
    method_name = method_params["method_name"]

    # Fetch problem parameters
    ind_size = problem_params["ind_size"]
    tot_init = problem_params["tot_init"]
    tot_budget = problem_params["tot_budget"]
    aqf = problem_params["aqf"]
    train_iter = problem_params["train_iter"]
    test_iter = problem_params["test_iter"]
    problem_name = problem_params["problem_name"]
    n_obj = problem_params["n_obj"]
    n_dim = problem_params["n_dim"]
    n_task_params = problem_params["n_task_params"]
    switch_size = problem_params["switch_size"]

    # Fetch problem
    problem = get_problem(name=problem_name, problem_params=n_dim, task_params=n_task_params)

    # What intermediate result do I want?
    # Final training data
    # Final inverse model

    # Prepare solution placeholder
    bayesian_vector = torch.zeros(tot_budget, n_dim + n_task_params + n_obj)
    bayesian_best_results = torch.Tensor([])
    if if_ind or if_fixed:
        if if_ind:
            task_list = fetch_task_lhs(n_task_params, ind_size)
            # task_list = [torch.rand(n_task_params)
            #              for i in range(ind_size)]
        if if_fixed and not if_switch:
            task_list = fetch_task_lhs(n_task_params, ind_size)
            # if method_name == "fixed_context_gp":
            #     task_list = fetch_task_list(trial)
            # elif method_name == "fixed_context_gp_20":
            #     task_list = fetch_task_list(trial, method_name="ind_gp_20")
            # else:
            #     pass
        if if_fixed and if_switch:
            task_list = fetch_task_lhs(n_task_params, switch_size)
            switch_list = task_list
            switch_count_vec = torch.zeros(switch_size)

        budget_per_task = tot_budget // ind_size

        if not if_switch:
            bayesian_vector_list = [torch.zeros(budget_per_task, n_dim + n_task_params + n_obj)
                                    for i in range(ind_size)]
        else:
            bayesian_vector_list = [torch.zeros(budget_per_task, n_dim + n_task_params + n_obj)
                                    for i in range(switch_size)]

        if if_fixed and not if_switch:
            bayesian_best_results = torch.ones(ind_size, n_dim + n_task_params + n_obj) * 1e6
        if if_fixed and if_switch:
            bayesian_best_results = torch.ones(switch_size, n_dim + n_task_params + n_obj) * 1e6
        bayesian_cut_results = torch.Tensor([])

    # Prepare initialization
    if if_ind or if_fixed:
        if if_switch:
            for i in range(switch_size):
                sample_size = tot_init // switch_size
                bayesian_vector_list[i][:sample_size, :n_dim] = torch.rand(sample_size, n_dim)
                bayesian_vector_list[i][:sample_size, n_dim:(n_dim + n_task_params)] = task_list[i]
                for j in range(sample_size):
                    bayesian_vector_list[i][j, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                        problem.evaluate(bayesian_vector_list[i][j, :(n_dim + n_task_params)])

                    switch_count_vec[i] += 1

                    if bayesian_vector_list[i][j, -1] < bayesian_best_results[i, -1]:
                        bayesian_best_results[i, :] = bayesian_vector_list[i][j, :]
                        print("Task {} in Iteration {}: Best Obj {}".format(i + 1, j + 1,
                                                                            bayesian_vector_list[i][j, -1]))
        else:
            for i in range(ind_size):
                sample_size = tot_init // ind_size
                bayesian_vector_list[i][:sample_size, :n_dim] = torch.rand(sample_size, n_dim)
                bayesian_vector_list[i][:sample_size, n_dim:(n_dim + n_task_params)] = task_list[i]
                for j in range(sample_size):
                    bayesian_vector_list[i][j, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                        problem.evaluate(bayesian_vector_list[i][j, :(n_dim + n_task_params)])

                    if bayesian_vector_list[i][j, -1] < bayesian_best_results[i, -1]:
                        bayesian_best_results[i, :] = bayesian_vector_list[i][j, :]
                        print("Task {} in Iteration {}: Best Obj {}".format(i + 1, j + 1,
                                                                            bayesian_vector_list[i][j, -1]))
    else:
        # Randomly sample tasks
        task_list = torch.rand(tot_init, n_task_params)
        solution_list = torch.rand(tot_init, n_dim)
        bayesian_vector[:tot_init, :n_dim] = solution_list
        bayesian_vector[:tot_init, n_dim:(n_dim + n_task_params)] = task_list
        for i in range(tot_init):
            bayesian_vector[i, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                problem.evaluate(bayesian_vector[i, :(n_dim + n_task_params)])

    # Prepare iterations
    if if_ind or if_fixed:
        model_list = None
        sample_size = tot_init // ind_size
        tot_size = tot_budget // ind_size
        cut_size = int(tot_size * 0.80)

        if not if_cut:
            for j in range(sample_size, tot_size):
                inverse_model_list = None
                temp_vectors = None
                model_list_prepare = []
                likelihood_list_prepare = []

                if if_ind:
                    for i in range(ind_size):
                        # Build a forward GP model
                        likelihood = gpytorch.likelihoods.GaussianLikelihood()
                        model = VanillaGP(bayesian_vector_list[i][:j, :n_dim],
                                          bayesian_vector_list[i][:j,
                                          (n_dim + n_task_params):(n_dim + n_task_params + n_obj)].squeeze(1),
                                          likelihood)
                        # Insert it into a list
                        model_list_prepare.append(model)
                        likelihood_list_prepare.append(likelihood)

                if if_fixed:
                    temp_vectors = torch.Tensor([])
                    if not if_switch:
                        for i in range(ind_size):
                            # Stack all the records together
                            temp_vectors = torch.cat([temp_vectors, bayesian_vector_list[i][:j, :]])
                    else:
                        for i in range(switch_size):
                            # Stack all the records together
                            cur_count_vec = int(switch_count_vec[i].item())
                            temp_vectors = torch.cat([temp_vectors, bayesian_vector_list[i][:cur_count_vec, :]])

                    # Build a unified GP model
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    model = VanillaGP(temp_vectors[:, :(n_dim + n_task_params)],
                                      temp_vectors[:, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)].squeeze(1),
                                      likelihood)
                    # Insert it into a list
                    model_list_prepare.append(model)
                    likelihood_list_prepare.append(likelihood)

                # Train the models
                model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter)
                model_list.train()

                # Train the inverse models
                if if_fixed and if_inverse and j % 5 == 0:
                    temp_bayesian_best_results = bayesian_best_results.clone()

                    # Build a unified inverse model from theta -> x
                    inverse_model_list_prepare = []
                    inverse_likelihood_list_prepare = []
                    for d in range(n_dim):
                        # For each inverse model, we have a dataset for each dimension (D totally)
                        # We first build D separate models via model list
                        # Then we combine them into a single ModelList class
                        inverse_likelihood = gpytorch.likelihoods.GaussianLikelihood()
                        inverse_model = VanillaGP(
                            temp_bayesian_best_results[:, n_dim:(n_dim + n_task_params)],
                            temp_bayesian_best_results[:, d],
                            inverse_likelihood)

                        inverse_likelihood_list_prepare.append(inverse_likelihood)
                        inverse_model_list_prepare.append(inverse_model)

                    # Train the models
                    # We train each model collaboratively
                    inverse_model_list = ModelList(inverse_model_list_prepare,
                                                   inverse_likelihood_list_prepare,
                                                   train_iter * 3)
                    inverse_model_list.train()
                else:
                    pass

                # Use inverse models to retrieve the active tasks
                candidate_sort = None
                switch_list = None
                if if_switch and if_inverse and j % 5 == 0:
                    # Input switch list and return the according uncertainty list
                    candidate_mean, candidate_std = inverse_model_list.test(switch_list)
                    candidate_entropy = torch.sum(candidate_std, dim=1)
                    candidate_sort = torch.argsort(candidate_entropy, descending=True)
                    # Obtain the task list with the largest uncertainty
                    task_list = switch_list[candidate_sort[:ind_size], :]

                # Forward-Inverse Sampling
                if if_inverse and inverse_model_list and not if_switch:
                    for i in range(ind_size):
                        inverse_sample_size = 10000
                        candidate_mean, candidate_std = inverse_model_list.test(task_list[i].unsqueeze(0))
                        # print("j:{}, i:{}, candidate_mean:{}, candidate_std:{}".format(j,
                        #                                                                i,
                        #                                                                candidate_mean,
                        #                                                                candidate_std))
                        candidates = candidate_mean + torch.randn(inverse_sample_size, n_dim) * candidate_std
                        candidates = torch.clamp(candidates, 0, 1).float()
                        candidates = torch.cat([candidates,
                                                task_list[i].unsqueeze(0).repeat(inverse_sample_size, 1)], dim=1)

                        # select candidate
                        candidates_lcb = model_lcb(model_list.model.models[0],
                                                   model_list.likelihood.likelihoods[0],
                                                   candidates,
                                                   beta=beta_ucb)
                        candidate_min = torch.argmin(candidates_lcb).item()
                        ans = candidates[candidate_min, :n_dim]

                        param = ans.unsqueeze(0)
                        # attach the param and task_params
                        bayesian_vector_list[i][j, :n_dim] = param.clone()
                        bayesian_vector_list[i][j, n_dim:(n_dim + n_task_params)] = task_list[i]
                        # Evaluate the solution
                        bayesian_vector_list[i][j, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                            problem.evaluate(bayesian_vector_list[i][j, :(n_dim + n_task_params)])
                        # Examine the best
                        if bayesian_vector_list[i][j, -1] < bayesian_best_results[i, -1]:
                            bayesian_best_results[i, :] = bayesian_vector_list[i][j, :]
                            print("Task {} in Iteration {}: Best Obj {}".format(i+1, j+1,
                                                                                bayesian_vector_list[i][j, -1]))
                else:
                    # Sample the models
                    if not if_switch:
                        for i in range(ind_size):
                            if if_ind:
                                if_debug = False
                                if i == task_number - 1 or i == task_number - 2:
                                    if_debug = False

                                # Apply GP-UCB selection for each task
                                ans = next_sample([model_list.model.models[i]],
                                                  [model_list.likelihood.likelihoods[i]],
                                                  n_dim,
                                                  torch.tensor([1], dtype=torch.float32).to(device),
                                                  mode=1,
                                                  fixed_solution=task_list[i],
                                                  beta=-beta_ucb,
                                                  opt_iter=test_iter,
                                                  if_debug=if_debug)
                            if if_fixed:
                                if_debug = False
                                if i == task_number - 1 or i == task_number - 2:
                                    if_debug = False
                                # Apply GP-UCB selection for each task
                                ans = next_sample([model_list.model.models[0]],
                                                  [model_list.likelihood.likelihoods[0]],
                                                  n_dim,
                                                  torch.tensor([1], dtype=torch.float32).to(device),
                                                  mode=2,
                                                  fixed_solution=task_list[i],
                                                  beta=-beta_ucb,
                                                  opt_iter=test_iter,
                                                  if_debug=if_debug)

                            # ans should be in size of [n_dim, ]
                            param = ans.unsqueeze(0)
                            # attach the param and task_params
                            bayesian_vector_list[i][j, :n_dim] = param.clone()
                            bayesian_vector_list[i][j, n_dim:(n_dim + n_task_params)] = task_list[i]
                            # Evaluate the solution
                            bayesian_vector_list[i][j, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                                problem.evaluate(bayesian_vector_list[i][j, :(n_dim + n_task_params)])

                            if bayesian_vector_list[i][j, -1] < bayesian_best_results[i, -1]:
                                # if i == task_number - 1 or i == task_number - 2:
                                if i < 100:
                                    bayesian_best_results[i, :] = bayesian_vector_list[i][j, :]
                                    print("Task {} in Iteration {}: Best Obj {}".format(i+1, j+1, bayesian_vector_list[i][j, -1]))
                                    print("Task {} in Iteration {}: Best Sol {}".format(i+1, j+1, bayesian_vector_list[i][j, :n_dim]))
                    else:
                        for i in range(ind_size):
                            if if_ind:
                                if_debug = False
                                if i == task_number - 1 or i == task_number - 2:
                                    if_debug = False

                                # Apply GP-UCB selection for each task
                                ans = next_sample([model_list.model.models[i]],
                                                  [model_list.likelihood.likelihoods[i]],
                                                  n_dim,
                                                  torch.tensor([1], dtype=torch.float32).to(device),
                                                  mode=1,
                                                  fixed_solution=task_list[i],
                                                  beta=-beta_ucb,
                                                  opt_iter=test_iter,
                                                  if_debug=if_debug)
                            if if_fixed:
                                if_debug = False
                                # Apply GP-UCB selection for each task
                                ans = next_sample([model_list.model.models[0]],
                                                  [model_list.likelihood.likelihoods[0]],
                                                  n_dim,
                                                  torch.tensor([1], dtype=torch.float32).to(device),
                                                  mode=2,
                                                  fixed_solution=task_list[i],
                                                  beta=-beta_ucb,
                                                  opt_iter=test_iter,
                                                  if_debug=if_debug)

                            # ans should be in size of [n_dim, ]
                            param = ans.unsqueeze(0)
                            # attach the param and task_params

                            task_id = int(candidate_sort[i].item())
                            task_sum = int(switch_count_vec[task_id].item())
                            bayesian_vector_list[task_id][task_sum, :n_dim] = param.clone()
                            bayesian_vector_list[task_id][task_sum, n_dim:(n_dim + n_task_params)] = task_list[i]
                            # Evaluate the solution
                            bayesian_vector_list[task_id][task_sum, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                                problem.evaluate(bayesian_vector_list[task_id][task_sum, :(n_dim + n_task_params)])
                            switch_count_vec[task_id] += 1

                            if bayesian_vector_list[task_id][task_sum, -1] < bayesian_best_results[task_id, -1]:
                                # if i == task_number - 1 or i == task_number - 2:
                                if i < 100:
                                    bayesian_best_results[task_id, :] = bayesian_vector_list[task_id][task_sum, :]
                                    print("Task {} in Iteration {}: Best Obj {}".format(task_id + 1, task_sum + 1,
                                                                                        bayesian_vector_list[task_id][task_sum, -1]))
                                    print("Task {} in Iteration {}: Best Sol {}".format(i + 1, j + 1,
                                                                                        bayesian_vector_list[task_id][task_sum, :n_dim]))
        else:
            for j in range(sample_size, cut_size):
                model_list_prepare = []
                likelihood_list_prepare = []
                if if_ind:
                    for i in range(ind_size):
                        # Build a forward GP model
                        likelihood = gpytorch.likelihoods.GaussianLikelihood()
                        model = VanillaGP(bayesian_vector_list[i][:j, :n_dim],
                                          bayesian_vector_list[i][:j,
                                          (n_dim + n_task_params):(n_dim + n_task_params + n_obj)].squeeze(1),
                                          likelihood)
                        # Insert it into a list
                        model_list_prepare.append(model)
                        likelihood_list_prepare.append(likelihood)

                if if_fixed:
                    temp_vectors = torch.Tensor([])
                    for i in range(ind_size):
                        # Stack all the records together
                        temp_vectors = torch.cat([temp_vectors, bayesian_vector_list[i][:j, :]])
                    # Build a unified GP model
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    model = VanillaGP(temp_vectors[:, :(n_dim + n_task_params)],
                                      temp_vectors[:, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)].squeeze(
                                          1),
                                      likelihood)
                    # Insert it into a list
                    model_list_prepare.append(model)
                    likelihood_list_prepare.append(likelihood)

                # Train the models
                model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter)
                model_list.train()

                # Sample the models
                for i in range(ind_size):
                    if if_ind:
                        # Apply GP-UCB selection for each task
                        ans = next_sample([model_list.model.models[i]],
                                          [model_list.likelihood.likelihoods[i]],
                                          n_dim,
                                          torch.tensor([1], dtype=torch.float32).to(device),
                                          mode=1,
                                          fixed_solution=task_list[i],
                                          beta=-beta_ucb,
                                          opt_iter=test_iter,
                                          if_debug=False)
                    if if_fixed:
                        # Apply GP-UCB selection for each task
                        ans = next_sample([model_list.model.models[0]],
                                          [model_list.likelihood.likelihoods[0]],
                                          n_dim,
                                          torch.tensor([1], dtype=torch.float32).to(device),
                                          mode=2,
                                          fixed_solution=task_list[i],
                                          beta=-beta_ucb,
                                          opt_iter=test_iter,
                                          if_debug=False)
                    # ans should be in size of [n_dim, ]
                    param = ans.unsqueeze(0)
                    # attach the param and task_params
                    bayesian_vector_list[i][j, :n_dim] = param.clone()
                    bayesian_vector_list[i][j, n_dim:(n_dim + n_task_params)] = task_list[i]
                    # Evaluate the solution
                    bayesian_vector_list[i][j, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                        problem.evaluate(bayesian_vector_list[i][j, :(n_dim + n_task_params)])

                    if bayesian_vector_list[i][j, -1] < bayesian_best_results[i, -1]:
                        bayesian_best_results[i, :] = bayesian_vector_list[i][j, :]
                        print("Task {} in Iteration {}: Best Obj {}".format(i + 1, j + 1,
                                                                            bayesian_vector_list[i][j, -1]))
                        print("Task {} in Iteration {}: Best Sol {}".format(i + 1, j + 1,
                                                                            bayesian_vector_list[i][j, :n_dim]))

            for j in range(cut_size, tot_size):
                inverse_model_list = None
                model_list_prepare = []
                likelihood_list_prepare = []
                temp_vectors = torch.Tensor([])
                # Train the forward model
                if if_fixed:
                    for i in range(ind_size):
                        # Stack all the records together
                        temp_vectors = torch.cat([temp_vectors, bayesian_vector_list[i][:cut_size, :]])
                    if j > cut_size:
                        temp_vectors = torch.cat([temp_vectors, bayesian_cut_results])
                    # Build a unified GP model
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    model = VanillaGP(temp_vectors[:, :(n_dim + n_task_params)],
                                      temp_vectors[:, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)].squeeze(
                                          1),
                                      likelihood)
                    # Insert it into a list
                    model_list_prepare.append(model)
                    likelihood_list_prepare.append(likelihood)

                    # Train the models
                    model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter)
                    model_list.train()
                # Train the inverse model
                if if_inverse:
                    temp_bayesian_best_results = temp_vectors.clone()

                    # Build a unified inverse model from theta -> x
                    inverse_model_list_prepare = []
                    inverse_likelihood_list_prepare = []
                    for d in range(n_dim):
                        # For each inverse model, we have a dataset for each dimension (D totally)
                        # We first build D separate models via model list
                        # Then we combine them into a single ModelList class
                        inverse_likelihood = gpytorch.likelihoods.GaussianLikelihood()
                        inverse_model = VanillaGP(
                            temp_bayesian_best_results[:, n_dim:(n_dim + n_task_params)],
                            temp_bayesian_best_results[:, d],
                            inverse_likelihood)

                        inverse_likelihood_list_prepare.append(inverse_likelihood)
                        inverse_model_list_prepare.append(inverse_model)

                    # Train the models
                    # We train each model collaboratively
                    inverse_model_list = ModelList(inverse_model_list_prepare, inverse_likelihood_list_prepare,
                                                   train_iter)
                    inverse_model_list.train()

                print("iteration j is {}".format(j))
                # Use model_list and inverse_model_list
                # to conduct an ec_iter EC algorithm so that
                # one (n_task_params, ) vector can be obtained
                ec_iter = 50
                ec_gen = 100
                timest = time.time()
                if if_random:
                    candidate_task = torch.rand(ind_size, n_task_params)
                else:
                    candidate_task = ec_alg_moo(model_list, inverse_model_list, ec_gen, ec_iter, n_dim, n_task_params)
                timeen = time.time()
                print("Time cost for this iteration is {}.".format(timeen-timest))

                # Thereafter, use this task parameter to derive
                # the optimal solutions via forward GP model.
                candidate_sols = torch.zeros(ind_size, n_dim + n_task_params + n_obj)
                # Apply GP-UCB selection for each task
                for i in range(ind_size):
                    ans = next_sample([model_list.model.models[0]],
                                      [model_list.likelihood.likelihoods[0]],
                                      n_dim,
                                      torch.tensor([1], dtype=torch.float32).to(device),
                                      mode=2,
                                      fixed_solution=candidate_task[i],
                                      beta=-beta_ucb,
                                      opt_iter=test_iter,
                                      if_debug=False)

                    # Evaluate this solution to the vector of records.
                    param = ans.unsqueeze(0)
                    # attach the param and task_params
                    candidate_sols[i, :n_dim] = param.clone()
                    candidate_sols[i, n_dim:(n_dim + n_task_params)] = candidate_task[i]
                    # Evaluate the solution
                    candidate_sols[i, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                        problem.evaluate(candidate_sols[i, :(n_dim + n_task_params)])
                    # move the candidate sols to the tot
                    bayesian_cut_results = torch.cat([bayesian_cut_results, candidate_sols[i].unsqueeze(0)])

                # Sample tasks
                # With EC flavor
                # candidate_size = 10000
                # candidate_tasks = torch.rand(candidate_size, n_task_params)
                # candidate_mean, candidate_std = inverse_model_list.test(candidate_tasks)
                # candidate_std_sum = torch.sum(candidate_std, dim=1)
                # candidate_sort, candidate_ind = torch.sort(candidate_std_sum, descending=True)
                # candidate_finals = candidate_tasks[candidate_ind, :][:ind_size, :]
                # # Optimize the tasks
                # candidate_sols = torch.zeros(ind_size, n_dim + n_task_params + n_obj)
                # for i in range(ind_size):
                #     # Apply GP-UCB selection for each task
                #     ans = next_sample([model_list.model.models[0]],
                #                       [model_list.likelihood.likelihoods[0]],
                #                       n_dim,
                #                       torch.tensor([1], dtype=torch.float32).to(device),
                #                       mode=2,
                #                       fixed_solution=candidate_finals[i],
                #                       opt_iter=test_iter,
                #                       if_debug=False)
                #
                #     param = ans.unsqueeze(0)
                #     # attach the param and task_params
                #     candidate_sols[i, :n_dim] = param.clone()
                #     candidate_sols[i, n_dim:(n_dim + n_task_params)] = candidate_finals[i]
                #     # Evaluate the solution
                #     candidate_sols[i, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                #         problem.evaluate(candidate_sols[i, :(n_dim + n_task_params)])
                # # move the candidate sols to the tot
                # bayesian_cut_results = torch.cat([bayesian_cut_results, candidate_sols])

                if j == tot_size - 1:
                    # move the cut_results to best results
                    bayesian_best_results = torch.cat([bayesian_best_results, bayesian_cut_results])
    else:
        model_list = None
        inverse_model_list = None
        sample_size = tot_init // ind_size
        tot_size = tot_budget // ind_size
        for j in range(sample_size, tot_size):
            if_current_active = False

            if if_active:
                if_current_active = (j % 5 == 0)

            print("Trial {}: Iteration {}".format(trial+1, j+1))
            model_list_prepare = []
            likelihood_list_prepare = []

            # Train a unified GP model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            cur_tot = tot_init + ind_size * (j - sample_size)
            model = VanillaGP(bayesian_vector[:cur_tot, :(n_dim + n_task_params)],
                              bayesian_vector[:cur_tot, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)].squeeze(1),
                              likelihood)
            model_list_prepare.append(model)
            likelihood_list_prepare.append(likelihood)
            model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter)

            # Train !
            model_list.train()

            # Train an inverse model
            if if_inverse and if_active:
                temp_bayesian_best_results = torch.Tensor([])
                if if_inner_cluster:
                    # determine the cluster centers
                    num_clusters = 20
                    min_size = 1
                    clusters = dict()
                    centers, cvt_model = cvt(bayesian_vector[:cur_tot, n_dim:(n_dim + n_task_params)].numpy(),
                                             num_clusters)
                    # decompose tensors into clusters
                    cluster_results = cvt_model.predict(bayesian_vector[:cur_tot, n_dim:(n_dim + n_task_params)].numpy())
                    for c_id in range(num_clusters):
                        clusters[c_id] = bayesian_vector[:cur_tot, :][c_id == cluster_results, :]
                        if clusters[c_id].shape[0] <= min_size:
                            temp_bayesian_best_results = torch.cat([temp_bayesian_best_results, clusters[c_id]], dim=0)
                        else:
                            indices = torch.argsort(clusters[c_id], dim=0)[:, -1]
                            temp_bayesian_best_results = torch.cat(
                                [temp_bayesian_best_results, clusters[c_id][indices, :][:min_size, :]], dim=0)

                # Build a unified inverse model from theta -> x
                inverse_model_list_prepare = []
                inverse_likelihood_list_prepare = []
                for d in range(n_dim):
                    # For each inverse model, we have a dataset for each dimension (D totally)
                    # We first build D separate models via model list
                    # Then we combine them into a single ModelList class
                    inverse_likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    if not if_inner_cluster:
                        inverse_model = VanillaGP(
                            bayesian_vector[:cur_tot, n_dim:(n_dim + n_task_params)],
                            bayesian_vector[:cur_tot, d],
                            inverse_likelihood)
                    else:
                        inverse_model = VanillaGP(
                            temp_bayesian_best_results[:, n_dim:(n_dim + n_task_params)],
                            temp_bayesian_best_results[:, d],
                            inverse_likelihood)

                    inverse_likelihood_list_prepare.append(inverse_likelihood)
                    inverse_model_list_prepare.append(inverse_model)

                # Train the models
                # We train each model collaboratively
                inverse_model_list = ModelList(inverse_model_list_prepare, inverse_likelihood_list_prepare, train_iter)
                inverse_model_list.train()

                # if not if_forward:
                #     for t in range(ind_size):
                #         candidate_solution, _ = inverse_model_list.test(task_list[t, :].unsqueeze(0))
                #
                #         # Bi-level?
                #         ans = next_sample([model_list.model.models[0]],
                #                           [model_list.likelihood.likelihoods[0]],
                #                           n_task_params,
                #                           torch.tensor([1], dtype=torch.float32).to(device),
                #                           mode=3,
                #                           fixed_solution=candidate_solution.squeeze(0),
                #                           opt_iter=test_iter,
                #                           if_debug=False)
                #         # Change task list to ans
                #         task_list[t, :] = ans.clone()
                #     print("Task_list is : {}".format(task_list))

            # Randomly or Actively search task parameters
            print("iteration j is {}".format(j))
            # Use model_list and inverse_model_list
            # to conduct an ec_iter EC algorithm so that
            # one (n_task_params, ) vector can be obtained
            ec_iter = 50
            ec_gen = 100
            timest = time.time()
            if not if_current_active:
                task_list = torch.rand(ind_size, n_task_params)
            else:
                task_list = ec_alg_moo(model_list, inverse_model_list, ec_gen, ec_iter, n_dim, n_task_params)
            timeen = time.time()
            print("Time cost for this iteration is {}.".format(timeen - timest))

            # Sample the next sample to evaluate per task
            for i in range(ind_size):
                # Apply GP-UCB selection for each task
                if if_forward and inverse_model_list:
                    # if_forward always coupled with inverse model
                    # inverse_sample_size = 500
                    inverse_sample_size = 10000
                    candidate_mean, candidate_std = inverse_model_list.test(task_list[i, :].unsqueeze(0))
                    candidates = candidate_mean + torch.randn(inverse_sample_size, n_dim) * candidate_std
                    candidates = torch.clamp(candidates, 0, 1).float()
                    candidates = torch.cat([candidates,
                                            task_list[i, :].unsqueeze(0).repeat(inverse_sample_size, 1)], dim=1)

                    # select candidate
                    candidates_lcb = model_lcb(model_list.model.models[0],
                                               model_list.likelihood.likelihoods[0],
                                               candidates)
                    candidate_min = torch.argmin(candidates_lcb).item()
                    ans = candidates[candidate_min, :n_dim]

                    param = ans.unsqueeze(0)
                    # attach the param and task_params
                    bayesian_vector[cur_tot + i, :n_dim] = param.clone()
                    bayesian_vector[cur_tot + i, n_dim:(n_dim + n_task_params)] = task_list[i, :]
                    # Evaluate the solution
                    bayesian_vector[cur_tot + i, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                        problem.evaluate(bayesian_vector[cur_tot + i, :(n_dim + n_task_params)])

                elif not if_unified:
                    ans = next_sample([model_list.model.models[0]],
                                      [model_list.likelihood.likelihoods[0]],
                                      n_dim,
                                      torch.tensor([1], dtype=torch.float32).to(device),
                                      mode=2,
                                      fixed_solution=task_list[i, :],
                                      opt_iter=test_iter,
                                      if_debug=False)
                    # ans should be in size of [n_dim, ]
                    param = ans.unsqueeze(0)
                    # attach the param and task_params
                    bayesian_vector[cur_tot + i, :n_dim] = param.clone()
                    bayesian_vector[cur_tot + i, n_dim:(n_dim + n_task_params)] = task_list[i, :]
                    # Evaluate the solution
                    bayesian_vector[cur_tot + i, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                        problem.evaluate(bayesian_vector[cur_tot + i, :(n_dim + n_task_params)])
                else:
                    ans = next_sample([model_list.model.models[0]],
                                      [model_list.likelihood.likelihoods[0]],
                                      n_dim+n_task_params,
                                      torch.tensor([1], dtype=torch.float32).to(device),
                                      mode=1,
                                      fixed_solution=task_list[i, :],
                                      opt_iter=test_iter,
                                      if_debug=False)
                    # ans should be in size of [n_dim + n_task_params, ]
                    param = ans.unsqueeze(0)
                    # attach the param and task_params
                    bayesian_vector[cur_tot + i, :(n_dim + n_task_params)] = param.clone()
                    # Evaluate the solution
                    bayesian_vector[cur_tot + i, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                        problem.evaluate(bayesian_vector[cur_tot + i, :(n_dim + n_task_params)])

    # Build inverse model
    if if_ind or if_fixed:
        model_records = dict()
        model_records["ind"] = True
        model_records["dim"] = n_dim
        if not if_switch:
            model_records["tasks"] = task_list
        else:
            model_records["tasks"] = switch_list
        # For each task build an inverse model from y -> x
        for i in range(ind_size):
            model_list = None
            model_list_prepare = []
            likelihood_list_prepare = []
            for d in range(n_dim):
                # For each inverse model, we have a dataset for each dimension (D totally)
                # We first build D separate models via model list
                # Then we combine them into a single ModelList class
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                model = VanillaGP(bayesian_vector_list[i][:, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)],
                                  bayesian_vector_list[i][:, d],
                                  likelihood)
                likelihood_list_prepare.append(likelihood)
                model_list_prepare.append(model)

            # Train the models
            # We train each model collaboratively
            model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter)
            model_list.train()

            # Save the records
            model_records["record_{}".format(i+1)] = bayesian_vector_list[i]
            # Save the model state_dict
            model_records["model_{}".format(i+1)] = model_list.model.state_dict()
            # Save the likelihood state_dict
            model_records["likelihood_{}".format(i+1)] = model_list.likelihood.state_dict()
            # How to save the models
            # print("model state dict:", model_list.model.state_dict())
            # print("likelihood state dict:", model_list.likelihood.state_dict())

            # How to load the models
            # How to test the models
            # test_samples = torch.rand(5).unsqueeze(1)
            # test_results = model_list.test(test_samples.clone())

        # Build a unified inverse model from theta -> x
        model_list = None
        model_list_prepare = []
        likelihood_list_prepare = []

        for d in range(n_dim):
            # For each inverse model, we have a dataset for each dimension (D totally)
            # We first build D separate models via model list
            # Then we combine them into a single ModelList class
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = VanillaGP(
                bayesian_best_results[:, n_dim:(n_dim + n_task_params)],
                bayesian_best_results[:, d],
                likelihood)
            likelihood_list_prepare.append(likelihood)
            model_list_prepare.append(model)

        # Train the models
        # We train each model collaboratively
        model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter*3)
        model_list.train()

        # Save the records
        model_records["record_tasks"] = bayesian_best_results
        # Save the model state_dict
        model_records["model_tasks"] = model_list.model.state_dict()
        # Save the likelihood state_dict
        model_records["likelihood_tasks"] = model_list.likelihood.state_dict()

        torch.save(model_records, "./{}/{}_{}_{}_{}.pth".format(direct_name,
                                                                task_number,
                                                                beta_ucb,
                                                                method_name,
                                                                trial))
    else:
        model_records = dict()
        model_records["ind"] = False
        model_records["dim"] = n_dim

        # determine the cluster centers
        num_clusters = 20
        min_size = 5
        clusters = dict()
        centers, cvt_model = cvt(bayesian_vector[:, n_dim:(n_dim + n_task_params)].numpy(),
                                 num_clusters)
        # decompose tensors into clusters
        cluster_results = cvt_model.predict(bayesian_vector[:, n_dim:(n_dim + n_task_params)].numpy())
        print(cluster_results.__class__)
        for c_id in range(num_clusters):
            clusters[c_id] = bayesian_vector[c_id == cluster_results, :]
            if clusters[c_id].shape[0] <= min_size:
                print(bayesian_best_results.shape)
                print(clusters[c_id].shape)
                bayesian_best_results = torch.cat([bayesian_best_results, clusters[c_id]], dim=0)
            else:
                indices = torch.argsort(clusters[c_id], dim=0)[:, -1]
                bayesian_best_results = torch.cat([bayesian_best_results, clusters[c_id][indices, :][:min_size, :]], dim=0)
        # determine the minimum number M_k per center
        # maintain the best M_k individuals per center and store into the best dataset
        # train the inverse model using the best dataset

        # Build a unified inverse model from theta -> x
        model_list = None
        model_list_prepare = []
        likelihood_list_prepare = []
        for d in range(n_dim):
            # For each inverse model, we have a dataset for each dimension (D totally)
            # We first build D separate models via model list
            # Then we combine them into a single ModelList class
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            if if_cluster:
                model = VanillaGP(
                    bayesian_best_results[:, n_dim:(n_dim + n_task_params)],
                    bayesian_best_results[:, d],
                    likelihood)
            else:
                model = VanillaGP(
                    bayesian_vector[:, n_dim:(n_dim + n_task_params)],
                    bayesian_vector[:, d],
                    likelihood)
            likelihood_list_prepare.append(likelihood)
            model_list_prepare.append(model)

        # Train the models
        # We train each model collaboratively
        model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter*3)
        model_list.train()

        if if_cluster:
            model_records["tasks"] = bayesian_best_results[:, n_dim:(n_dim + n_task_params)]
        else:
            model_records["tasks"] = bayesian_vector[:, n_dim:(n_dim + n_task_params)]
        # Save the records
        model_records["record_tasks"] = bayesian_best_results
        # Save the model state_dict
        model_records["model_tasks"] = model_list.model.state_dict()
        # Save the likelihood state_dict
        model_records["likelihood_tasks"] = model_list.likelihood.state_dict()

        torch.save(model_records, "./{}/{}_{}_{}_{}.pth".format(direct_name,
                                                                task_number,
                                                                beta_ucb,
                                                                method_name,
                                                                trial))

    return None


def main_solver(trials, method_name="unified_gp"):
    problem_params = configure_problem(problem_name)
    method_params = configure_method(method_name)
    for trial in range(trials):
        solver(problem_params, method_params, trial)


def main_retrival(direct_name="sep_arm", method_name="ind_gp", problem_name="sep_arm", trial_number=0):
    # method_name = "ind_gp"
    # problem_name = "sep_arm"
    # trial_number = 0

    if True or method_name == "ind_gp" or "ind_gp_20" or "fixed_context_gp" or "fixed_context_gp_20":
        results = torch.load("./{}/{}_{}.pth".format(direct_name, method_name, trial_number))
        n_dim = results["dim"]
        n_task_param = len(results["tasks"][0])
        n_tasks = len(results["tasks"])
        problem = get_problem(problem_name, n_dim, task_params)
        result_model_list = []

        result_model_list_prepare = []
        result_likelihood_list_prepare = []
        for d in range(n_dim):
            result_model = None
            result_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            result_model = VanillaGP(
                results["record_tasks"][:, n_dim:(n_dim + task_params)],
                results["record_tasks"][:, d],
                result_likelihood
            )
            result_model_list_prepare.append(result_model)
            result_likelihood_list_prepare.append(result_likelihood)

        result_model_list = ModelList(result_model_list_prepare, result_likelihood_list_prepare, 600)
        result_model_list.model.load_state_dict(results["model_tasks"])
        result_model_list.likelihood.load_state_dict(results["likelihood_tasks"])

        # Test the data and showcase the result
        # trial = 10000
        # tasks = torch.rand(trial, n_task_param)
        # tasks_optimas, tasks_uncertainty = result_model_list.test(tasks)
        # tasks_sol = torch.cat([tasks_optimas, tasks], dim=1)
        #
        # tasks_ans = torch.zeros(trial)
        # for tr in range(trial):
        #     tasks_ans[tr] = problem.evaluate(tasks_sol[tr, :])
        #
        # plot_hist(tasks_ans.numpy())
        # plt.show()
        return result_model_list

    else:
        results = torch.load("./{}/{}_{}.pth".format(direct_name, method_name, trial_number))
        n_dim = results["dim"]
        n_task_param = results["tasks"].shape[1]
        n_tasks = results["tasks"].shape[0]
        problem = get_problem(problem_name, n_dim, task_params)
        result_model_list = []

        result_model_list_prepare = []
        result_likelihood_list_prepare = []
        for d in range(n_dim):
            result_model = None
            result_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            result_model = VanillaGP(
                results["record_tasks"][:, n_dim:(n_dim + task_params)],
                results["record_tasks"][:, d],
                result_likelihood
            )
            result_model_list_prepare.append(result_model)
            result_likelihood_list_prepare.append(result_likelihood)

        result_model_list = ModelList(result_model_list_prepare, result_likelihood_list_prepare, 200)
        result_model_list.model.load_state_dict(results["model_tasks"])
        result_model_list.likelihood.load_state_dict(results["likelihood_tasks"])

        # Test the data and showcase the result
        # trial = 10000
        # tasks = torch.rand(trial, n_task_param)
        # tasks_optimas, tasks_uncertainty = result_model_list.test(tasks)
        # tasks_sol = torch.cat([tasks_optimas, tasks], dim=1)
        #
        # tasks_ans = torch.zeros(trial)
        # for tr in range(trial):
        #     tasks_ans[tr] = problem.evaluate(tasks_sol[tr, :])
        #
        # plot_hist(tasks_ans.numpy())
        # plt.show()
        return result_model_list


def testing(tasks, result_model_list, problem):
    trial, dim = tasks.shape
    tasks_optimas, tasks_uncertainty = result_model_list.test(tasks)
    tasks_sol = torch.cat([tasks_optimas, tasks], dim=1)

    tasks_ans = torch.zeros(trial)
    for tr in range(trial):
        tasks_ans[tr] = problem.evaluate(tasks_sol[tr, :])

    return tasks_ans, tasks_sol


def sphere_testing(tasks, result_model_list, problem):
    trial, dim = tasks.shape
    tasks_optimas, tasks_uncertainty = result_model_list.test(tasks)
    tasks_sol = torch.cat([tasks_optimas, tasks], dim=1)

    tasks_ans = torch.zeros(trial)
    for tr in range(trial):
        tasks_ans[tr] = problem.divergence(tasks_sol[tr, :])

    return tasks_ans, tasks_sol


def uniform_sample(sample_width=100, task_num=2):
    # Prepare the np.linspace
    list_linspace = []
    list_ravel = []
    for i in range(task_num):
        list_linspace.append(np.linspace(0.05, 0.95, sample_width))

    # Grid creation
    grids = np.meshgrid(*list_linspace)

    # List of ravel grids
    for grid in grids:
        list_ravel.append(grid.ravel())

    points = np.vstack(list_ravel).T
    samples = torch.from_numpy(points).float()
    return samples

# def uniform_sample(sample_width=100):
#     x = np.linspace(0, 1, sample_width)
#     y = np.linspace(0, 1, sample_width)
#     xx, yy = np.meshgrid(x, y)
#     points = np.vstack([xx.ravel(), yy.ravel()]).T
#     samples = torch.from_numpy(points).float()
#
#     return samples


def compare_all(n_task_params=2, mode="canonical"):
    models_tot = dict()
    results_tot = dict()
    sols_tot = dict()

    # test_data
    # test_tot = 10000
    # test_data = torch.rand(test_tot, 2)
    sample_width = 100
    test_tot = sample_width ** n_task_params
    test_data = uniform_sample(sample_width, n_task_params)
    # sample_width = 100
    # test_tot = sample_width ** n_task_params
    # test_data = uniform_sample(sample_width, n_task_params)
    # test_tot = 20
    # test_data = torch.stack(fetch_task_lhs(5, 20))
    # test_data = torch.rand(test_tot, n_task_params)
    print(test_data)
    print(test_data.shape)

    # method_name = "context_gp"
    # problem_name = problem_name
    trial_number_tot = 1

    # problem
    problem = get_problem(problem_name, dim_size, n_task_params)

    model_dict = dict()
    result_lists = []
    std_lists = []
    sol_lists = []
    # Store the models for each method
    for m_id, method_name in enumerate(method_name_list):
        model_lists = []
        results_lists = torch.zeros(trial_number_tot, test_tot)
        # stds_lists = torch.zeros(trial_number_tot, test_tot)
        sols_lists = torch.zeros(trial_number_tot, test_tot, dim_size + n_task_params)
        for trial_number in range(trial_number_tot):
            temp_model = main_retrival(direct_name=direct_name,
                                       method_name=method_name,
                                       problem_name=problem_name,
                                       trial_number=trial_number)
            model_lists.append(temp_model)
            if mode == "canonical":
                temp_result, temp_sol = testing(test_data, temp_model, problem)
            else:
                temp_result, temp_sol = sphere_testing(test_data, temp_model, problem)
            results_lists[trial_number, :] = temp_result
            sols_lists[trial_number, :, :] = temp_sol

        models_tot[method_name] = model_lists
        results_tot[method_name] = results_lists
        sols_tot[method_name] = sols_lists
        result_lists.append(torch.mean(results_lists, dim=0).numpy())
        std_lists.append(torch.std(results_lists, dim=0).numpy())
        sol_lists.append(sols_lists)

        plot_details(test_data.numpy(), torch.mean(results_lists, dim=0).numpy(), method_name)
        # plot_tot(results_lists, m_id + 1, method_name_list)
    # plot_box(result_lists, method_name_list)
    # plot_box(result_lists, ["Strategy 1", "Strategy 2", "Strategy 3"])
    # plot_box(result_lists, ["Strategy 1", "Strategy 2"])
    plt.show()
    while True:
        x_ind = int(input())
        print("test_data is {}.".format(test_data[x_ind, :]))
        for ind_ind in range(len(result_lists)):
            print("Result {}: {} / {}.".format(ind_ind + 1,
                                               result_lists[ind_ind][x_ind],
                                               std_lists[ind_ind][x_ind]
                                               ))
        debug_each(sol_lists, ["Strategy 1", "Strategy 2"], x_ind)
        plt.show()


def compare_convergence():
    models_tot = dict()
    results_tot = dict()
    model_dict = dict()

    # method_name = "context_gp"
    # problem_name = "sep_arm"
    trial_number_tot = 9

    result_lists = []
    convergence_lists = []
    plot_lists = []
    # Both convergence lists and plot lists
    # have the size of [task_number, method_number]
    # and each element is a torch.Tensor
    for t_id in range(task_number):
        empty_list = []
        empty_list_plot = []
        for m_id in range(len(method_name_list)):
            empty_list.append(torch.Tensor([]))
            empty_list_plot.append(torch.Tensor([]))
        convergence_lists.append(empty_list)
        plot_lists.append(empty_list_plot)

    # Store the results for each method
    for m_id, method_name in enumerate(method_name_list):
        method_results = torch.zeros(task_number, trial_number_tot)
        for trial_number in range(trial_number_tot):
            # Retrieve all the results
            results = torch.load("./{}/{}_{}.pth".format(direct_name, method_name, trial_number))
            # Store all the results in the last iteration
            temp_results = results["record_tasks"][:task_number, -1]
            print(temp_results.shape)
            method_results[:, trial_number] = temp_results

            for t_id in range(task_number):
                temp_convergence, _ = torch.cummin(torch.log(results["record_{}".format(t_id+1)][:, -1]+1), dim=0)
                temp_tot = results["record_{}".format(t_id+1)]
                convergence_lists[t_id][m_id] = \
                    torch.cat([convergence_lists[t_id][m_id], temp_convergence.unsqueeze(1)], dim=1)
                plot_lists[t_id][m_id] = \
                    torch.cat([plot_lists[t_id][m_id], temp_tot], dim=0)

        result_lists.append(method_results)
        # plot_details(test_data.numpy(), torch.mean(results_lists, dim=0).numpy(), method_name)
        # plot_tot(results_lists, m_id + 1, method_name)
    # plot_box(result_lists, method_name_list)

    fig, axs = plt.subplots(4, 5, figsize=(12, 8))
    for t_id in range(task_number):
        x_id = t_id // 5
        y_id = t_id % 5
        ax_plot_iteration_convergence(axs[x_id][y_id], convergence_lists[t_id], method_name_list, t_id)

    # for t_id in range(task_number):
    #     # debug_tot(plot_lists[t_id], method_name_list, 200, 400, 4)
    #     debug_hist(plot_lists[t_id], method_name_list, 0, 200, -1)

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    # plot_convergence(result_lists, method_name_list)
    # fig.text('Iterations')
    # fig.text('Value')
    plt.suptitle("Comparison Results on the Convergence of Optimization Algorithms")
    plt.tight_layout()
    plt.show()


def fetch_task_list(trial_number, problem_name="sep_arm", method_name="ind_gp"):
    results = torch.load("./{}/{}_{}_{}.pth".format(direct_name, problem_name, method_name, trial_number))
    return results["tasks"]


def fetch_task_lhs(task_param=2, task_size=10):
    results = torch.load("./task_list_{}_{}.pth".format(task_size, task_param))
    return results


# method_name_list = ["ind_gp", "context_gp", "unified_gp",
#                     "context_gp_plain", "unified_gp_plain",
#                     "fixed_context_gp", "inverse_context_gp_plain",
#                     "inverse_context_gp_inner_plain",
#                     "forward_inverse_context_gp_inner_plain",
#                     "forward_inverse_context_gp_plain"]


if __name__ == "__main__":
    # main_solver(trials=10, method_name="ind_gp")
    main_solver(trials=10, method_name="fixed_context_gp")
    # main_solver(trials=10, method_name="context_inverse_active_gp_plain")
    # main_solver(trials=10, method_name="context_gp_plain")
    # main_solver(trials=10, method_name="fixed_context_inverse_cut_gp")
    # main_solver(trials=10, method_name="forward_inverse_fixed_context_gp_plain")
    # trial_num = 1
    # # tasks_1 = fetch_task_list(trial_num, method_name="unified_gp")
    # tasks_2 = fetch_task_list(trial_num, method_name="unified_gp_plain")
    # # tasks_3 = fetch_task_list(trial_num, method_name="context_gp")
    # tasks_4 = fetch_task_list(trial_num, method_name="context_gp_plain")
    # tasks_5 = fetch_task_list(trial_num, method_name="ind_gp")
    # tasks_5 = torch.stack(tasks_5)
    # tasks_6 = fetch_task_list(trial_num, method_name="fixed_context_gp")
    # tasks_6 = torch.stack(tasks_6)
    # tasks_7 = fetch_task_list(trial_num, method_name="forward_inverse_context_gp_plain")
    # sample_width = 100
    # tasks = uniform_sample(sample_width)
    # print(tasks.shape)
    # plt.scatter(tasks[:, 0], tasks[:, 1], alpha=0.5)
    # plt.title("Sampled tasks for testing data")
    # plt.show()
    #
    # # plt.scatter(tasks_1[:, 0], tasks_1[:, 1], alpha=0.2, label="unified_gp")
    # plt.scatter(tasks_2[:, 0], tasks_2[:, 1], alpha=0.2, label="unified_gp_plain")
    # # plt.scatter(tasks_3[:, 0], tasks_3[:, 1], alpha=0.2, label="context_gp")
    # plt.scatter(tasks_4[:, 0], tasks_4[:, 1], alpha=0.2, label="Random")
    # plt.scatter(tasks_5[:, 0], tasks_5[:, 1], alpha=0.2, label="Strategy 1")
    # plt.scatter(tasks_6[:, 0], tasks_6[:, 1], alpha=0.2, label="Strategy 2")
    # plt.scatter(tasks_7[:, 0], tasks_7[:, 1], alpha=0.2, label="Forward-Inverse")
    # plt.legend()
    # plt.show()
    # compare_all(n_task_params=task_params)
    # compare_convergence()

