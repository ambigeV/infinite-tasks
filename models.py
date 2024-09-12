import gpytorch
import numpy
from gpytorch.mlls import SumMarginalLogLikelihood
import torch
import torch.distributions as dist
from utils import plot_grad

device = torch.device("cpu")


class VanillaGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.no_of_data, self.no_of_dim = train_x.shape
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.no_of_dim))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ModelList:
    def __init__(self, model_list, likelihood_list, train_iter):
        self.model = gpytorch.models.IndependentModelList(*model_list)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list)
        self.train_iter = train_iter

    def train(self):
        mll = SumMarginalLogLikelihood(self.likelihood, self.model)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        for i in range(self.train_iter):
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            loss = -mll(output, self.model.train_targets)
            loss.backward()
            optimizer.step()

    def test(self, test_x):
        # return tensors of size [sample_size, no_of_models(dimension)]
        if isinstance(test_x, numpy.ndarray):
            test_x = torch.from_numpy(test_x).float()

        self.model.eval()
        self.likelihood.eval()
        dimensions = len(self.model.models)
        test_x_list = [test_x for i in range(dimensions)]
        mean_ans, std_ans = torch.Tensor([]), torch.Tensor([])
        predictions = self.likelihood(*self.model(*test_x_list))

        for i in range(dimensions):
            mean_ans = torch.cat([mean_ans, predictions[i].mean.unsqueeze(1)], dim=1)
            std_ans = torch.cat([std_ans, predictions[i].stddev.unsqueeze(1)], dim=1)

        mean_ans = torch.clamp(mean_ans.detach(), 0, 1)
        std_ans = torch.clamp(std_ans.detach(), 0, 1)

        return mean_ans, std_ans


def evaluation(model, likelihood, test_x, if_grad=True):
    model.eval()
    likelihood.eval()

    if isinstance(test_x, numpy.ndarray):
        observed_pred = likelihood(model(torch.from_numpy(test_x).float()))
    elif isinstance(test_x, tuple):
        observed_pred = likelihood(model(*test_x))
    else:
        observed_pred = likelihood(model(test_x))
    if if_grad:
        return observed_pred.mean, torch.sqrt(observed_pred.variance)
    else:
        return observed_pred.mean.detach(), torch.sqrt(observed_pred.variance).detach()


def upper_confidence_bound(mean_tensor, std_tensor, beta=-0.5, beta_mean=1):
    return beta_mean * mean_tensor + beta * std_tensor


def lower_confidence_bound(mean_tensor, std_tensor, beta=0.5):
    return mean_tensor - beta * std_tensor


def model_lcb(model, likelihood, test_x, beta=0.5):
    means, std = evaluation(model, likelihood, test_x, if_grad=False)
    lcb = lower_confidence_bound(means, std, beta=beta)
    return lcb


def compute_ei(mean_vec: torch.tensor, std_vec: torch.tensor,
               cur_best: torch.tensor):
    normal_dist = dist.Normal(0, 1)
    eps = 1e-6
    gamma_solutions = (cur_best - mean_vec) / (std_vec + eps)
    ei_solutions = std_vec * (
            gamma_solutions * normal_dist.cdf(gamma_solutions) +
            normal_dist.log_prob(gamma_solutions).exp())
    return ei_solutions


def scalar_obj(x, a, alpha):
    # This scalarization can be referred to as so-called augmented
    # Tchebycheff approach, where the ideal points are assumed to be 0
    # and the epsilon is all zero as well.
    return 0 * torch.max(a * x, dim=1).values + torch.matmul(x, a) * alpha


def next_sample(model_list, likelihood_list, sol_dim, weights, mode, fixed_solution, alpha=0.05, beta=-0.5,
                beta_mean=1, opt_iter=20, num_restart=8, aq_func="UCB", if_cuda=False, if_debug=False, y_best=None):
    # mode
    # mode 1:
    # solutions are all you got and fixed solution should be None
    # evaluate([solutions])
    # mode 2:
    # solutions are the first part and concatenate the fixed solution after solutions
    # evaluate([solutions, fixed_solution])
    # mode 3:
    # solutions are the second part and concatenate  the fixed solution before solutions
    # evaluate([fixed_solution, solutions])

    # Initiate a set of starting points in size (num_start)
    obj_size = len(model_list)

    if if_debug:
        grad_list = torch.Tensor([])

    if if_cuda:
        solutions = torch.rand(num_restart, sol_dim).to(device)
    else:
        solutions = torch.rand(num_restart, sol_dim)
    solutions.requires_grad = True
    optimizer = torch.optim.Adam([solutions], lr=1e-2)

    for k in range(opt_iter):
        optimizer.zero_grad()
        # Start Searching next sample to evaluate
        means_list = []
        stds_list = []
        # Evaluate the possible results
        for i in range(len(model_list)):
            if mode == 1:
                mean_values, std_values = evaluation(model_list[i], likelihood_list[i], solutions)
                if if_debug:
                    print("mean_values: {}, std:values {}".format(mean_values.detach(),
                                                                  std_values.detach()))

            elif mode == 2:
                # fixed_solution.requires_grad = True
                mean_values, std_values = \
                    evaluation(model_list[i],
                               likelihood_list[i],
                               torch.cat([solutions, fixed_solution.unsqueeze(0).repeat(num_restart, 1)], dim=1))
                if if_debug:
                    print("mean_values: {}, std:values {}".format(mean_values.detach(),
                                                                  std_values.detach()))



            elif mode == 3:
                # fixed_solution.requires_grad = True
                mean_values, std_values = \
                    evaluation(model_list[i],
                               likelihood_list[i],
                               torch.cat([fixed_solution.unsqueeze(0).repeat(num_restart, 1), solutions], dim=1))
            else:
                assert 1 == 2
            means_list.append(mean_values.unsqueeze(1))
            stds_list.append(std_values.unsqueeze(1))

        means = torch.cat(means_list, dim=1)
        stds = torch.cat(stds_list, dim=1)


        # ucb is in size of [num_restart, obj_size]
        if aq_func == "UCB":
            ucb = upper_confidence_bound(means, stds, beta, beta_mean)
        else:
            ucb = -compute_ei(means, stds, y_best)
        # print("The outputs in iteration {} is {}.".format(k, ucb.detach()))
        outputs = scalar_obj(ucb, weights, alpha)
        # print("The outputs in iteration {} is {}.".format(k, outputs.detach()))

        # loss update
        if k < opt_iter - 1:
            loss = outputs.sum()
            loss.backward()
            optimizer.step()

            sorted_outputs, ind = torch.sort(outputs)
            ans = solutions[ind, :][0].detach()

            if if_debug:
                # print("Loss in iter {} is {} from answer {}.".format(k,
                #                                                      torch.min(outputs.detach()).item(),
                #                                                      ans))
                # print("Gradient in iter {} is {}.".format(
                #     k, torch.mean(torch.abs(solutions.grad), dim=1)
                # ))
                grad_list = torch.cat([grad_list, torch.mean(torch.abs(solutions.grad), dim=1).unsqueeze(0)], dim=0)

            # Bound the tensor in between 0 and 1
            with torch.no_grad():
                # print("Before clamp in iter {}: {}".format(i, x))
                solutions.clamp_(0, 1)
                # print("After clamp in iter {}: {}".format(i, solutions))
        else:
            sorted_outputs, ind = torch.sort(outputs)
            ans = solutions[ind, :][0].detach()
            # if if_debug:
            #     print("Best ans: {}, Best output: {}".format(ans.detach(), sorted_outputs.detach()))
            # print(outputs.detach())
            # print(solutions.detach())
            return ans
