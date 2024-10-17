from models import ModelList, evaluation
import torch
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from models import compute_gradient_list
import time


def model_det(models, data):
    data_tot = torch.Tensor([])
    for cur_model in models:
        cur_model.eval()
        cur_data = torch.linalg.det(cur_model.covar_module.base_kernel(data).to_dense()).detach()[:, None]
        data_tot = torch.cat([data_tot, cur_data], dim=1)
    data_tot = torch.mean(data_tot, dim=1)
    return data_tot


class OPT(Problem):
    def __init__(self, input_func, n_var, n_obj, xl=0, xu=1):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.input_func = input_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.input_func.evaluate(x)


class ModelRank:
    def __init__(self, forward_model_list: ModelList, inverse_model_list: ModelList,
                 n_dim, n_task_params, avg_tasks):
        self.forward_model_list = forward_model_list
        self.inverse_model_list = inverse_model_list
        self.n_dim = n_dim
        self.n_obj = 2
        self.n_task_params = n_task_params
        self.avg_tasks = avg_tasks

    def evaluate(self, solution: torch.tensor = None):

        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, False returns Pareto front, True returns the evaluated solutions
        :return:
        """
        print("solution shape is {}.".format(solution.shape))
        n_sols, n_task_dim = solution.shape
        m_sols, n_dim = self.avg_tasks.shape
        assert n_task_dim == self.n_task_params

        time_start = time.time()
        ###################################################################################
        # Compute forward uncertainty
        task_num = n_sols
        context_num = m_sols
        candidates = solution
        context_vectors = self.avg_tasks

        # Expand dimensions to concatenate the context vectors with each candidate tensor
        # expanded_candidates = candidates.unsqueeze(1)  # Shape: (n_sols, 1, n_task_dim)
        expanded_candidates = np.expand_dims(candidates, axis=1)
        # expanded_contexts = context_vectors.unsqueeze(0)  # Shape: (1, m_sols, n_dim)
        expanded_contexts = np.expand_dims(context_vectors, axis=0)

        # Concatenate along the last dimension with candidates behind context vectors
        # Shape: (n_sols, m_sols, n_task_dim+n_dim)
        # Assuming expanded_contexts and expanded_candidates are numpy arrays
        expanded_contexts_expanded = np.tile(expanded_contexts, (task_num, 1, 1))
        expanded_candidates_expanded = np.tile(expanded_candidates, (1, context_num, 1))
        # Concatenating along the last axis
        concatenated_tensors = np.concatenate((expanded_contexts_expanded, expanded_candidates_expanded), axis=-1)
        # print(concatenated_tensors.shape)

        # Compute the objective values
        # Objective value computed along the last dimension of concatenated tensors
        # objective_values = concatenated_tensors.norm(dim=-1)  # Shape: (n_sols, m_sols)
        _, forward_uncertainty = evaluation(self.forward_model_list.model.models[0],
                                            self.forward_model_list.likelihood.likelihoods[0],
                                            concatenated_tensors,
                                            False)

        # Average over the m_sols context vectors to get the final n_sols objective values
        forward_uncertainty = forward_uncertainty.mean(dim=1)  # Shape: (n_sols,)
        ###################################################################################

        # Compute inverse uncertainty
        _, inverse_uncertainty = self.inverse_model_list.test(solution)
        inverse_uncertainty = -torch.sum(inverse_uncertainty, dim=1) # Shape: (n_sols)

        uncertainty = torch.cat([forward_uncertainty.unsqueeze(1), inverse_uncertainty.unsqueeze(1)], dim=1)

        time_end = time.time()
        print("Time cost is {}.".format(time_end - time_start))
        return uncertainty.numpy()


class ModelGradient:
    def __init__(self, inverse_model_list: ModelList, n_dim, n_task_params, n_tasks, mode):
        self.inverse_model_list = inverse_model_list
        self.n_dim = n_dim
        self.n_obj = 2
        self.n_task_params = n_task_params
        self.n_tasks = n_tasks
        self.mode = mode

    def evaluate(self, solution: torch.tensor = None):

        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, False returns Pareto front, True returns the evaluated solutions
        :return:
        """
        # print("solution shape is {}.".format(solution.shape))
        n_sols, n_task_dim = solution.shape
        assert n_task_dim == self.n_task_params * self.n_tasks

        ###################################################################################
        # Compute sum of the gradient information
        new_solution = solution.reshape(-1, self.n_task_params)
        new_solution_torch = torch.from_numpy(new_solution).float()

        grad_result = compute_gradient_list(self.inverse_model_list, new_solution_torch, self.mode)
        # grad_result: Tensor Shape: (n_sols * n_tasks, )
        grad_result = grad_result.view(n_sols, -1)
        grad_result = torch.mean(grad_result, dim=1)
        # grad_result: Tensor Shape: (n_sols, )

        ###################################################################################
        # Compute the det(K(\Theta))
        new_solution_torch = new_solution_torch.view(n_sols, self.n_tasks, self.n_task_params)
        data_result = model_det(self.inverse_model_list.model.models, new_solution_torch)
        # data_result: Tensor Shape: (n_sols, )

        result = - torch.cat([grad_result.unsqueeze(1), data_result.unsqueeze(1)], dim=1)

        return result.numpy()


class ModelGradientMyopic:
    def __init__(self, inverse_model_list: ModelList, n_dim, n_task_params, n_tasks, mode, ref_tasks):
        self.inverse_model_list = inverse_model_list
        self.n_dim = n_dim
        self.n_obj = 2
        self.n_task_params = n_task_params
        self.n_tasks = n_tasks
        self.mode = mode
        self.ref_tasks = ref_tasks

    def evaluate(self, solution: torch.tensor = None):

        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, False returns Pareto front, True returns the evaluated solutions
        :return:
        """
        # print("solution shape is {}.".format(solution.shape))
        n_sols, n_task_dim = solution.shape
        assert n_task_dim == self.n_task_params * self.n_tasks

        ###################################################################################
        # Compute sum of the gradient information
        new_solution = solution.reshape(-1, self.n_task_params)
        new_solution_torch = torch.from_numpy(new_solution).float()

        grad_result = compute_gradient_list(self.inverse_model_list, new_solution_torch, self.mode)
        # grad_result: Tensor Shape: (n_sols * n_tasks, )
        grad_result = grad_result.view(n_sols, -1)
        grad_result = torch.mean(grad_result, dim=1)
        # grad_result: Tensor Shape: (n_sols, )

        ###################################################################################
        # Compute the det(K(\Theta))
        new_solution_torch = new_solution_torch.view(n_sols, self.n_tasks, self.n_task_params)
        new_solution_torch = torch.cat([new_solution_torch,
                                        self.ref_tasks.unsqueeze(0).repeat(n_sols, 1, 1)], dim=1)
        data_result = model_det(self.inverse_model_list.model.models, new_solution_torch)
        # data_result: Tensor Shape: (n_sols, )

        result = - torch.cat([grad_result.unsqueeze(1), data_result.unsqueeze(1)], dim=1)

        return result.numpy()


def ec_alg_moo(model_list: ModelList, inverse_model_list: ModelList,
               ec_gen: int, ec_iter: int, n_dim: int, n_task_params: int):
    avg_size = 100
    sample_size = 20
    avg_tasks = torch.rand(avg_size, n_dim)
    problem_current = ModelRank(model_list, inverse_model_list, n_dim, n_task_params, avg_tasks)
    obj_problem = OPT(problem_current, n_var=problem_current.n_task_params, n_obj=problem_current.n_obj)
    algorithm = NSGA2(pop_size=ec_gen)
    res = minimize(obj_problem,
                   algorithm,
                   ('n_gen', ec_iter),
                   seed=1,
                   eliminate_duplicates=True,
                   verbose=False)

    tot_pf, _ = res.X.shape
    print("Current_shape is {}".format(res.X.shape))
    if sample_size <= tot_pf:
        sample_tasks = torch.from_numpy(res.X[:sample_size]).float()
    else:
        sample_tasks = torch.from_numpy(res.X).float()
        new_sample_tasks = torch.rand(sample_size - tot_pf, n_task_params)
        sample_tasks = torch.cat([sample_tasks, new_sample_tasks], dim=0)

    return sample_tasks


def ec_active_moo(inverse_model_list: ModelList, ec_gen: int, ec_iter: int, n_dim: int,
                  n_task_params: int, n_tasks: int, mode: int):
    sample_size = n_tasks
    problem_current = ModelGradient(inverse_model_list, n_dim, n_task_params, n_tasks, mode)
    obj_problem = OPT(problem_current, n_var=problem_current.n_task_params * n_tasks, n_obj=problem_current.n_obj)
    algorithm = NSGA2(pop_size=ec_gen)
    res = minimize(obj_problem,
                   algorithm,
                   ('n_gen', ec_iter),
                   seed=1,
                   eliminate_duplicates=True,
                   verbose=False)

    tot_pf, _ = res.X.shape
    print("Current_shape is {}".format(res.X.shape))
    sample_tasks = torch.from_numpy(res.X).float()
    # if sample_size <= tot_pf:
    #     sample_tasks = torch.from_numpy(res.X[:sample_size]).float()
    # else:
    #     sample_tasks = torch.from_numpy(res.X).float()
    #     new_sample_tasks = torch.rand(sample_size - tot_pf, n_task_params)
    #     sample_tasks = torch.cat([sample_tasks, new_sample_tasks], dim=0)

    return sample_tasks


def ec_active_myopic_moo(inverse_model_list: ModelList, ec_gen: int, ec_iter: int, n_dim: int,
                         n_task_params: int, n_tasks: int, mode: int, ref_tasks: torch.Tensor):
    sample_size = n_tasks
    assert n_tasks == 1
    problem_current = ModelGradientMyopic(inverse_model_list,
                                          n_dim,
                                          n_task_params,
                                          n_tasks,
                                          mode,
                                          ref_tasks)
    obj_problem = OPT(problem_current, n_var=problem_current.n_task_params * n_tasks, n_obj=problem_current.n_obj)
    algorithm = NSGA2(pop_size=ec_gen)
    res = minimize(obj_problem,
                   algorithm,
                   ('n_gen', ec_iter),
                   seed=1,
                   eliminate_duplicates=True,
                   verbose=False)

    tot_pf, _ = res.X.shape
    print("Current_shape is {}".format(res.X.shape))
    sample_tasks = torch.from_numpy(res.X).float()
    # if sample_size <= tot_pf:
    #     sample_tasks = torch.from_numpy(res.X[:sample_size]).float()
    # else:
    #     sample_tasks = torch.from_numpy(res.X).float()
    #     new_sample_tasks = torch.rand(sample_size - tot_pf, n_task_params)
    #     sample_tasks = torch.cat([sample_tasks, new_sample_tasks], dim=0)

    return sample_tasks