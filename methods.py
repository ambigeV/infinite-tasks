import random

import gpytorch
import torch
import numpy as np
from models import VanillaGP, ModelList, next_sample, device, model_lcb
from problems import get_problem
from utils import plot_hist, cvt, plot_tot, plot_box, plot_details
import matplotlib.pyplot as plt

# method_name_list = ["ind_gp", "context_gp", "unified_gp",
#                     "context_gp_plain", "unified_gp_plain",
#                     "fixed_context_gp", "inverse_context_gp_plain",
#                     "inverse_context_gp_inner_plain",
#                     "forward_inverse_context_gp_inner_plain",
#                     "forward_inverse_context_gp_plain"]

method_name_list = ["ind_gp", "fixed_context_gp",
                    "context_gp_plain", "forward_inverse_context_gp_plain"]

problem_name = "sep_arm"
direct_name = "result_physics"


def configure_problem(problem_name):
    params = dict()
    params["ind_size"] = 10
    params["tot_init"] = 200
    params["tot_budget"] = 2000
    # params["tot_budget"] = 300
    params["aqf"] = "ucb"
    params["train_iter"] = 200
    params["test_iter"] = 30
    params["problem_name"] = problem_name
    params["n_obj"] = 1
    params["n_dim"] = 10
    params["n_task_params"] = 2
    return params


def configure_method(method_name):
    params = dict()
    params["if_ind"] = False
    params["if_inverse"] = False
    params["if_forward"] = False
    params["if_unified"] = False
    params["if_cluster"] = True
    params["if_inner_cluster"] = False
    params["if_fixed"] = False
    params["method_name"] = method_name

    if method_name == "ind_gp":
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

    if method_name == "fixed_context_gp":
        params["if_fixed"] = True

    return params


def solver(problem_params, method_params, trial):
    # Offer parameters of method and problem
    # Return the results for one trial

    # Fetch method parameters
    if_ind = method_params["if_ind"]
    if_inverse = method_params["if_inverse"]
    if_forward = method_params["if_forward"]
    if_unified = method_params["if_unified"]
    if_fixed = method_params["if_fixed"]
    if_cluster = method_params["if_cluster"]
    if_inner_cluster = method_params["if_inner_cluster"]
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

    # Fetch problem
    problem = get_problem(name=problem_name, problem_params=n_dim)

    # What intermediate result do I want?
    # Final training data
    # Final inverse model

    # Prepare solution placeholder
    bayesian_vector = torch.zeros(tot_budget, n_dim + n_task_params + n_obj)
    bayesian_best_results = torch.Tensor([])
    if if_ind or if_fixed:
        if if_ind:
            task_list = [torch.rand(n_task_params)
                         for i in range(ind_size)]
        if if_fixed:
            task_list = fetch_task_list(trial)

        budget_per_task = tot_budget//ind_size
        bayesian_vector_list = [torch.zeros(budget_per_task, n_dim + n_task_params + n_obj)
                                for i in range(ind_size)]
        bayesian_best_results = torch.ones(ind_size, n_dim + n_task_params + n_obj)

    # Prepare initialization
    if if_ind or if_fixed:
        for i in range(ind_size):
            sample_size = tot_init // ind_size
            bayesian_vector_list[i][:sample_size, :n_dim] = torch.rand(sample_size, n_dim)
            bayesian_vector_list[i][:sample_size, n_dim:(n_dim + n_task_params)] = task_list[i]
            for j in range(sample_size):
                bayesian_vector_list[i][j, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                    problem.evaluate(bayesian_vector_list[i][j, :(n_dim + n_task_params)])

                if bayesian_vector_list[i][j, -1] < bayesian_best_results[i, -1]:
                    bayesian_best_results[i, :] = bayesian_vector_list[i][j, :]
                    print("Task {} in Iteration {}: Best Obj {}".format(i+1, j+1, bayesian_vector_list[i][j, -1]))
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

        for j in range(sample_size, tot_size):
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
                                  temp_vectors[:, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)].squeeze(1),
                                  likelihood)
                # Insert it into a list
                model_list_prepare.append(model)
                likelihood_list_prepare.append(likelihood)

            # Train the models
            model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter)
            model_list.train()

            for i in range(ind_size):
                if if_ind:
                    # Apply GP-UCB selection for each task
                    ans = next_sample([model_list.model.models[i]],
                                      [model_list.likelihood.likelihoods[i]],
                                      n_dim,
                                      torch.tensor([1], dtype=torch.float32).to(device),
                                      mode=1,
                                      fixed_solution=task_list[i],
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
                    print("Task {} in Iteration {}: Best Obj {}".format(i+1, j+1, bayesian_vector_list[i][j, -1]))
                # DEBUG:
                # cur_obj = bayesian_vector_list[i][j, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)]
                # print("Iteration {}: Task #{} with fitness value {}.".format(j+1,
                #                                                              i+1,
                #                                                              cur_obj))
    else:
        model_list = None
        inverse_model_list = None
        sample_size = tot_init // ind_size
        tot_size = tot_budget // ind_size
        for j in range(sample_size, tot_size):
            print("Trial {}: Iteration {}".format(trial+1, j+1))
            model_list_prepare = []
            likelihood_list_prepare = []

            # Randomly sample tasks
            task_list = torch.rand(ind_size, n_task_params)

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
            if if_inverse and torch.rand(1).item() < 0.5:
                temp_bayesian_best_results = torch.Tensor([])
                if if_inner_cluster:
                    # determine the cluster centers
                    num_clusters = 20
                    min_size = 3
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

                if not if_forward:
                    for t in range(ind_size):
                        candidate_solution, _ = inverse_model_list.test(task_list[t, :].unsqueeze(0))

                        # Bi-level?
                        ans = next_sample([model_list.model.models[0]],
                                          [model_list.likelihood.likelihoods[0]],
                                          n_task_params,
                                          torch.tensor([1], dtype=torch.float32).to(device),
                                          mode=3,
                                          fixed_solution=candidate_solution.squeeze(0),
                                          opt_iter=test_iter,
                                          if_debug=False)
                        # Change task list to ans
                        task_list[t, :] = ans.clone()
                    print("Task_list is : {}".format(task_list))

            # Sample the next sample to evaluate per task
            for i in range(ind_size):
                # Apply GP-UCB selection for each task
                if if_forward and inverse_model_list:
                    # if_forward always coupled with inverse model
                    inverse_sample_size = 500
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
        model_records["tasks"] = task_list
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
        model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter)
        model_list.train()

        # Save the records
        model_records["record_tasks"] = bayesian_best_results
        # Save the model state_dict
        model_records["model_tasks"] = model_list.model.state_dict()
        # Save the likelihood state_dict
        model_records["likelihood_tasks"] = model_list.likelihood.state_dict()

        torch.save(model_records, "./{}/{}_{}_{}.pth".format(direct_name, problem_name, method_name, trial+3))
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
        model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter)
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

        torch.save(model_records, "./{}/{}_{}_{}.pth".format(direct_name, problem_name, method_name, trial+3))

    return None


def main_solver(trials, method_name="unified_gp"):
    problem_params = configure_problem("sep_arm")
    method_params = configure_method(method_name)
    for trial in range(trials):
        solver(problem_params, method_params, trial)


def main_retrival(method_name="ind_gp", problem_name="sep_arm", trial_number=0):
    # method_name = "ind_gp"
    # problem_name = "sep_arm"
    # trial_number = 0

    if method_name == "ind_gp" or "fixed_context_gp":
        results = torch.load("./{}/{}_{}_{}.pth".format(direct_name, problem_name, method_name, trial_number))
        n_dim = results["dim"]
        n_task_param = len(results["tasks"][0])
        n_tasks = len(results["tasks"])
        problem = get_problem(problem_name, n_task_param)
        result_model_list = []

        result_model_list_prepare = []
        result_likelihood_list_prepare = []
        for d in range(n_dim):
            result_model = None
            result_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            result_model = VanillaGP(
                results["record_tasks"][:, n_dim:(n_dim + n_task_param)],
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

    else:
        results = torch.load("./{}/{}_{}_{}.pth".format(direct_name, problem_name, method_name, trial_number))
        n_dim = results["dim"]
        n_task_param = results["tasks"].shape[1]
        n_tasks = results["tasks"].shape[0]
        problem = get_problem(problem_name, n_task_param)
        result_model_list = []

        result_model_list_prepare = []
        result_likelihood_list_prepare = []
        for d in range(n_dim):
            result_model = None
            result_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            result_model = VanillaGP(
                results["record_tasks"][:, n_dim:(n_dim + n_task_param)],
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

    return tasks_ans


def uniform_sample(sample_width=100):
    x = np.linspace(0.05, 1, sample_width)
    y = np.linspace(0.05, 1, sample_width)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T
    samples = torch.from_numpy(points).float()

    return samples


def compare_all():
    models_tot = dict()
    results_tot = dict()

    # test_data
    # test_tot = 10000
    # test_data = torch.rand(test_tot, 2)
    sample_width = 100
    test_tot = sample_width * sample_width
    test_data = uniform_sample(sample_width)

    # method_name = "context_gp"
    problem_name = "sep_arm"
    trial_number_tot = 10

    # problem
    problem = get_problem(problem_name, 2)

    model_dict = dict()
    result_lists = []
    # Store the models for each method
    for m_id, method_name in enumerate(method_name_list):
        model_lists = []
        results_lists = torch.zeros(trial_number_tot, test_tot)
        for trial_number in range(trial_number_tot):
            temp_model = main_retrival(method_name=method_name,
                                       problem_name=problem_name,
                                       trial_number=trial_number)
            model_lists.append(temp_model)
            temp_result = testing(test_data, temp_model, problem)
            results_lists[trial_number, :] = temp_result

        models_tot[method_name] = model_lists
        results_tot[method_name] = results_lists
        result_lists.append(torch.mean(results_lists, dim=0).numpy())

        # plot_details(test_data.numpy(), torch.mean(results_lists, dim=0).numpy(), method_name)
        # plot_tot(results_lists, m_id + 1, method_name)
    plot_box(result_lists, ["Strategy 1", "Strategy 2", "Random", "Forward-Inverse"])
    plt.show()


def fetch_task_list(trial_number, problem_name="sep_arm", method_name="ind_gp"):
    results = torch.load("./{}/{}_{}_{}.pth".format(direct_name, problem_name, method_name, trial_number))
    return results["tasks"]


# method_name_list = ["ind_gp", "context_gp", "unified_gp",
#                     "context_gp_plain", "unified_gp_plain",
#                     "fixed_context_gp", "inverse_context_gp_plain",
#                     "inverse_context_gp_inner_plain",
#                     "forward_inverse_context_gp_inner_plain",
#                     "forward_inverse_context_gp_plain"]


if __name__ == "__main__":
    main_solver(trials=7, method_name="fixed_context_gp")
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
    # compare_all()

