import torch
import gpytorch
import numpy as np
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from models import VanillaGP


# class MultitaskGP(gpytorch.models.ExactGP):
#     """
#     Multitask Gaussian Process model for optimization tasks.
#     This implements the MTGP model described in the paper.
#     """
#
#     def __init__(self, train_x, train_y, likelihood, num_tasks):
#         super(MultitaskGP, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel()
#         )
#         # Task correlation matrix via Cholesky decomposition
#         self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=1)
#         self.num_tasks = num_tasks
#
#     def forward(self, x, i=None):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#
#         # If task indices are provided, incorporate task covariance
#         if i is not None:
#             # covar_x = self.task_covar_module(i).mul(covar_x)
#             covar_i = self.task_covar_module(i)
#             print(covar_i.shape)
#             covar_x = covar_x.mul(covar_i)
#             print(covar_x.shape)
#
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskGP(gpytorch.models.ExactGP):
    """
    Multitask Gaussian Process model using the Hadamard product pattern.
    This implementation allows for efficient handling of large test sets.
    """

    def __init__(self, train_inputs, train_targets, likelihood, num_tasks):
        # train_inputs should be a tuple of (x, task_indices)
        super(MultitaskGP, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        # Task correlation matrix via IndexKernel
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=1)
        self.num_tasks = num_tasks

    def forward(self, x, i):
        """
        Forward pass of the model.

        Args:
            x: Input features
            i: Task indices

        Returns:
            MultivariateNormal distribution
        """
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)

        # Get task-task covariance
        covar_i = self.task_covar_module(i)

        # Combine using Hadamard product (element-wise multiplication)
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


def build_and_train_mtgp(X_all, y_all, task_indices, ind_size, train_iter):
    """
    Build and train a multitask Gaussian process model

    Args:
        X_all: Combined input features from all tasks
        y_all: Combined target values from all tasks
        task_indices: Task indices for each data point
        ind_size: Number of tasks
        train_iter: Number of training iterations

    Returns:
        mtgp_model: Trained MTGP model
        likelihood: Gaussian likelihood
        correlation_matrix: Task correlation matrix
    """

    # Initialize model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mtgp_model = MultitaskGP((X_all, task_indices), y_all, likelihood, ind_size)

    # Train the model
    mtgp_model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(mtgp_model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, mtgp_model)

    for i in range(train_iter):
        optimizer.zero_grad()
        output = mtgp_model(X_all, task_indices)
        loss = -mll(output, y_all)
        loss.backward()
        optimizer.step()

    mtgp_model.eval()
    likelihood.eval()

    # Extract correlation matrix
    correlation_matrix = extract_task_correlation_matrix_cholesky(mtgp_model)

    return mtgp_model, likelihood, correlation_matrix


def derive_transfer_matrix(correlation_matrix):
    """
    Creates a unified transfer matrix from the correlation matrix.
    Takes advantage of the symmetry of the correlation matrix.

    Args:
        correlation_matrix (torch.Tensor): Matrix of correlations between tasks

    Returns:
        torch.Tensor: Transfer probability matrix
    """
    n_tasks = correlation_matrix.shape[0]
    transfer_matrix = torch.zeros(n_tasks, n_tasks)

    # Compute only upper triangular part (i < j)
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            # Compute normalized correlation as transfer probability
            k_ij = correlation_matrix[i, j]
            k_ii = correlation_matrix[i, i]
            k_jj = correlation_matrix[j, j]

            # Normalized correlation coefficient
            transfer_prob = k_ij / torch.sqrt(k_ii * k_jj)

            # Ensure valid probability range
            transfer_prob = torch.clamp(transfer_prob, 0, 1)

            # Set both (i,j) and (j,i) entries since matrix is symmetric
            transfer_matrix[i, j] = transfer_prob
            transfer_matrix[j, i] = transfer_prob

    # Diagonal remains zero (no self-transfer)
    return transfer_matrix


def adaptive_knowledge_transfer(problem, task_list, ind_size, tot_budget, budget_per_task, n_dim, n_task_params, n_obj,
                                mtgp_correlation_mat, improved_solutions):
    """
    Performs adaptive knowledge transfer between tasks based on a unified transfer matrix.

    Args:
        problem: Problem instance to evaluate solutions
        task_list: List of task parameters
        ind_size: Number of tasks
        tot_budget: Total function evaluation budget
        budget_per_task: Budget per task
        n_dim: Number of decision variables
        n_task_params: Number of task parameters
        n_obj: Number of objectives
        mtgp_correlation_mat: Correlation matrix between tasks
        improved_solutions: Current best solutions for each task

    Returns:
        Best solutions found for each task
    """
    # Create bayesian databases for each task
    bayesian_vector_list = [torch.zeros(budget_per_task, n_dim + n_task_params + n_obj) for _ in range(ind_size)]
    bayesian_budget_meter = torch.zeros(ind_size)
    bayesian_best_results = torch.zeros(ind_size, n_dim + n_task_params + n_obj)

    # Derive transfer matrix once
    transfer_matrix = derive_transfer_matrix(mtgp_correlation_mat)

    # Function evaluation counter
    fes = 0

    # Main optimization loop
    while fes < tot_budget:
        # Generate random matrix for stochastic transfers
        random_matrix = torch.rand(ind_size, ind_size)

        # Adaptive knowledge transfer based on transfer matrix and random matrix
        for m in range(ind_size):
            if fes >= tot_budget:
                break

            for m_donor in range(ind_size):
                if fes >= tot_budget:
                    break

                if m == m_donor:
                    continue

                # Only transfer if random value is smaller than transfer probability
                if random_matrix[m, m_donor] < transfer_matrix[m, m_donor]:
                    # Get current evaluation count for this task
                    current_count = int(bayesian_budget_meter[m].item())
                    next_index = current_count

                    # Ensure we have enough space in the tensor
                    if next_index >= bayesian_vector_list[m].shape[0]:
                        # Expand the tensor if needed
                        expansion = torch.zeros(budget_per_task, n_dim + n_task_params + n_obj)
                        bayesian_vector_list[m] = torch.cat([bayesian_vector_list[m], expansion], dim=0)

                    # Create new solution with task parameters of current task
                    transfer_solution = improved_solutions[m_donor].clone()
                    transfer_solution[n_dim:(n_dim + n_task_params)] = task_list[m]

                    # Evaluate transferred solution
                    transfer_obj = problem.evaluate(transfer_solution[:(n_dim + n_task_params)])
                    fes += 1
                    transfer_solution[(n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = transfer_obj

                    # Add to database
                    bayesian_vector_list[m][next_index] = transfer_solution.clone()
                    bayesian_budget_meter[m] += 1

                    # Update best solution if improved
                    if transfer_obj < bayesian_best_results[m, -1]:
                        bayesian_best_results[m] = transfer_solution.clone()
                        print(f"Task {m + 1} Transfer from Task {m_donor + 1} at FE {fes}: {transfer_obj}")

    return bayesian_best_results


def extract_task_correlation_matrix_cholesky(model):
    """Extract task correlation matrix using Cholesky decomposition"""
    # In IndexKernel, covar_factor is the Cholesky factor L
    L = model.task_covar_module.covar_factor

    # Compute K = LL^T
    K = L @ L.t()

    # Convert to correlation matrix
    diag_values = torch.sqrt(torch.diag(K))
    outer_product = torch.outer(diag_values, diag_values)

    correlation_matrix = K / outer_product
    return correlation_matrix


def extract_task_correlation_matrix(mtgp_model):
    """
    Extract the task correlation matrix K̄ from the trained MTGP model.
    This matrix represents the similarity between different tasks.

    Args:
        mtgp_model: Trained MultitaskGP model

    Returns:
        Task correlation matrix K̄
    """
    # Extract the task covariance matrix
    var_matrix = mtgp_model.task_covar_module.covar_matrix.evaluate()

    # Convert to correlation matrix
    task_stddevs = torch.sqrt(torch.diag(var_matrix))
    task_stddevs_outer = torch.outer(task_stddevs, task_stddevs)

    # Compute correlation matrix
    correlation_matrix = var_matrix / task_stddevs_outer

    return correlation_matrix


def compute_mtgp_lcb(model, likelihood, x, task_indices, beta=1.0):
    """
    Compute the Lower Confidence Bound (LCB) for points using the MTGP model.

    Args:
        model: Trained MultitaskGP model
        likelihood: Gaussian likelihood
        x: Points to evaluate
        task_indices: Task indices for each point
        beta: Exploration parameter (higher value = more exploration)

    Returns:
        LCB values for the provided points
    """
    # print("model: num of tasks {}".format(model.num_tasks))
    print("data size: {}".format(x.shape))
    print("task size: {}".format(task_indices.shape))
    # Get predictive distribution
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x, task_indices))
        mean = observed_pred.mean
        std = observed_pred.stddev

    # For minimization problems, LCB = mean - beta * std
    lcb = mean - beta * std

    return lcb


def build_ei_acquisition(gp_model, best_f, maximize=False):
    """
    Build Expected Improvement (EI) acquisition function.
    For minimization problems, we negate the outputs.

    Args:
        gp_model: Trained GP model (could be local GP for a specific task)
        best_f: Current best function value
        maximize: If True, we maximize the objective, otherwise minimize

    Returns:
        EI acquisition function
    """
    # For minimization, we negate the best_f
    if not maximize:
        best_f = -best_f

    # Create the EI acquisition function
    ei = ExpectedImprovement(gp_model, best_f, maximize=maximize)

    return ei


def optimize_ei(ei_acqf, bounds, num_restarts=10, raw_samples=100):
    """
    Optimize the EI acquisition function to find the next point to evaluate.

    Args:
        ei_acqf: EI acquisition function
        bounds: Bounds for optimization
        num_restarts: Number of restarts for optimization
        raw_samples: Number of initial samples

    Returns:
        The point that maximizes the EI acquisition function
    """
    candidate, _ = optimize_acqf(
        acq_function=ei_acqf,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return candidate


def compute_transfer_probability(task_correlation_matrix, task_i, task_j):
    """
    Compute the transfer probability between two tasks based on their correlation.

    Args:
        task_correlation_matrix: Task correlation matrix K̄
        task_i: Index of the first task
        task_j: Index of the second task

    Returns:
        Transfer probability between task_i and task_j
    """
    # According to equation (23) in the paper
    k_ij = task_correlation_matrix[task_i, task_j]
    k_ii = task_correlation_matrix[task_i, task_i]
    k_jj = task_correlation_matrix[task_j, task_j]

    # Compute transfer probability
    transfer_prob = k_ij / torch.sqrt(k_ii * k_jj)

    # Ensure probability is between 0 and 1
    transfer_prob = torch.clamp(transfer_prob, 0, 1)

    return transfer_prob


import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np


def extract_correlation_matrix(mtgp_model):
    """
    Extract the task correlation matrix from MTGP model
    """
    # Get the task covariance matrix
    task_covar = mtgp_model.task_covar_module.covar_matrix.evaluate()

    # Convert to correlation matrix
    diag_values = torch.sqrt(torch.diag(task_covar))
    outer_product = torch.outer(diag_values, diag_values)

    correlation_matrix = task_covar / outer_product
    return correlation_matrix


def compute_transfer_probability(correlation_matrix, task_i, task_j):
    """
    Compute the transfer probability between tasks using equation (23) in the paper
    """
    k_ij = correlation_matrix[task_i, task_j]
    k_ii = correlation_matrix[task_i, task_i]
    k_jj = correlation_matrix[task_j, task_j]

    # Compute transfer probability
    transfer_prob = k_ij / torch.sqrt(k_ii * k_jj)

    # Ensure probability is between 0 and 1
    transfer_prob = torch.clamp(transfer_prob, 0, 1)

    return transfer_prob


def generate_de_trials(population, task_params, n_dim, n_task_params, F=0.6, CR=0.7, num_trials=50):
    """
    Generate trial solutions using DE/rand/1/bin
    Creates num_trials trials for each member of the population (total: num_trials * NP)

    Args:
        population: Current population of solutions
        task_params: Task parameters to use for all trials
        n_dim: Number of decision variables
        n_task_params: Number of task parameters
        F: Differential weight (scaling factor)
        CR: Crossover probability
        num_trials: Number of trials to generate per population member

    Returns:
        Tensor of trial solutions with shape (num_trials * NP, n_dim + n_task_params)
    """
    NP = len(population)
    total_trials = num_trials * NP
    trials = torch.zeros(total_trials, n_dim + n_task_params)

    # Copy task parameters to all trials
    trials[:, n_dim:(n_dim + n_task_params)] = task_params.repeat(total_trials, 1)

    # DE/rand/1/bin
    for i in range(NP):
        base_index = i * num_trials  # Starting index for trials of the i-th population member

        for t in range(num_trials):
            trial_index = base_index + t

            # Select 3 random individuals from population (different from i)
            r1, r2, r3 = np.random.choice(NP, 3, replace=False)

            r1_vec = population[r1, :n_dim]
            r2_vec = population[r2, :n_dim]
            r3_vec = population[r3, :n_dim]

            # DE/rand/1 mutation
            mutant = r1_vec + F * (r2_vec - r3_vec)
            mutant = torch.clamp(mutant, 0, 1)

            # Get the individual from the population
            individual = population[i, :n_dim]

            # Binomial crossover
            for j in range(n_dim):
                if np.random.rand() <= CR or j == np.random.randint(n_dim):
                    trials[trial_index, j] = mutant[j]
                else:
                    trials[trial_index, j] = individual[j]

    return trials


def select_de(mtgp_model, likelihood, n_dim, trials, task_idx):
    # Evaluate trials using MTGP surrogate
    task_idx = torch.ones(len(trials), dtype=torch.long) * task_idx
    mtgp_values = compute_mtgp_lcb(mtgp_model, likelihood, trials[:, :n_dim].clone(), task_idx)

    # Select best candidate
    best_idx = torch.argmin(mtgp_values)
    best_candidate = trials[best_idx]

    return best_candidate


def build_local_gp_and_optimize(database, best_individual, n_dim, nearest_neighbors=50, F=0.6, CR=0.7):
    """
    Build local GP model and optimize EI acquisition function using DE
    """
    # Get n nearest solutions to best individual
    X_all = database[:, :n_dim]
    y_all = database[:, -1]
    pop_size = len(X_all)
    if nearest_neighbors > pop_size:
        nearest_neighbors = pop_size

    # Calculate distances
    distances = torch.cdist(best_individual[:n_dim].unsqueeze(0), X_all).squeeze()

    # Get indices of n nearest neighbors
    nearest_indices = torch.argsort(distances)[:nearest_neighbors]

    # Extract nearest neighbors
    X_neighbors = database[nearest_indices, :n_dim]
    y_neighbors = database[nearest_indices, -1]

    # Build local GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = VanillaGP(X_neighbors, y_neighbors, likelihood)

    # Train model
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(100):
        optimizer.zero_grad()
        output = model(X_neighbors)
        loss = -mll(output, y_neighbors)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    # Define bounded space from nearest neighbors
    min_bounds = torch.min(X_neighbors, dim=0).values
    max_bounds = torch.max(X_neighbors, dim=0).values

    # Add small margin
    margin = 0.01 * (max_bounds - min_bounds)
    min_bounds = torch.max(min_bounds - margin, torch.zeros_like(min_bounds))
    max_bounds = torch.min(max_bounds + margin, torch.ones_like(max_bounds))

    # Define EI acquisition function
    best_f = torch.min(y_neighbors)

    def expected_improvement(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).reshape(1, -1)

        with torch.no_grad():
            observed_pred = likelihood(model(x_tensor))
            mean = observed_pred.mean
            std = observed_pred.stddev

        # For minimization
        z = (best_f - mean) / (std + 1e-6)
        normal_cdf = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
        normal_pdf = (1 / np.sqrt(2 * np.pi)) * torch.exp(-0.5 * z ** 2)

        ei = (z * normal_cdf + normal_pdf) * std

        return -ei.item()  # Negative because we're minimizing

    # Optimize EI using DE
    de_pop_size = 20
    de_iterations = 50

    # Initialize DE population
    population = np.zeros((de_pop_size, n_dim))
    for i in range(de_pop_size):
        for j in range(n_dim):
            population[i, j] = np.random.uniform(min_bounds[j].item(), max_bounds[j].item())

    # Evaluate initial population
    fitness = np.array([expected_improvement(ind) for ind in population])

    # DE iterations
    for _ in range(de_iterations):
        for i in range(de_pop_size):
            # Select 3 random individuals, different from i
            idx_list = list(range(de_pop_size))
            idx_list.remove(i)
            r1, r2, r3 = np.random.choice(idx_list, 3, replace=False)

            # Mutation: DE/rand/1
            mutant = population[r1] + F * (population[r2] - population[r3])

            # Bound the mutant vector
            for j in range(n_dim):
                mutant[j] = max(min(mutant[j], max_bounds[j].item()), min_bounds[j].item())

            # Crossover: binomial
            trial = np.copy(population[i])
            j_rand = np.random.randint(n_dim)
            for j in range(n_dim):
                if np.random.random() <= CR or j == j_rand:
                    trial[j] = mutant[j]

            # Selection
            trial_fitness = expected_improvement(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

    # Get best solution from DE
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx]

    # Create improved solution
    improved_solution = best_individual.clone()
    improved_solution[:n_dim] = torch.tensor(best_solution, dtype=torch.float32)

    return improved_solution


import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np


# def compute_mtgp_lcb(model, likelihood, x, task_indices, beta=1.0):
#     """
#     Compute the Lower Confidence Bound (LCB) for points using the MTGP model.
#
#     Args:
#         model: Trained MultitaskGP model
#         likelihood: Gaussian likelihood
#         x: Points to evaluate
#         task_indices: Task indices for each point
#         beta: Exploration parameter (higher value = more exploration)
#
#     Returns:
#         LCB values for the provided points
#     """
#     # print("model: num of tasks {}".format(model.task_covar_module.num_tasks))
#     print("data size: {}".format(x.shape))
#     print("task size: {}".format(task_indices.shape))
#     # Get predictive distribution
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         observed_pred = likelihood(model(x, i=task_indices))
#         mean = observed_pred.mean
#         std = observed_pred.stddev
#
#     # For minimization problems, LCB = mean - beta * std
#     lcb = mean - beta * std
#
#     return lcb


def test_task_correlation_extraction():
    """
    Test the extraction of task correlation matrix from MTGP model
    with a controlled example where we know what to expect.
    """
    # Create a simple 2-task scenario with known correlation
    # For simplicity, let's create synthetic data where:
    # - Task 1 is f(x) = x^2
    # - Task 2 is f(x) = 2*x^2 + 1 (same shape, different scale)

    # These should be highly correlated

    # Generate synthetic data
    torch.manual_seed(0)  # For reproducibility

    # Task 1 data
    x1 = torch.linspace(0, 1, 10).reshape(-1, 1)
    y1 = x1.pow(2).reshape(-1)

    # Task 2 data
    x2 = torch.linspace(0, 1, 10).reshape(-1, 1)
    y2 = (2 * x2.pow(2) + 1).reshape(-1)

    # Combined data
    X_all = torch.cat([x1, x2], dim=0)
    y_all = torch.cat([y1, y2], dim=0)

    # Task indices (0 for task 1, 1 for task 2)
    task_indices = torch.cat([
        torch.zeros(10, dtype=torch.long),
        torch.ones(10, dtype=torch.long)
    ])

    # Initialize model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Create the multitask model

    model = MultitaskGP(X_all, y_all, likelihood, 2)

    # Train the model
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Train for a small number of iterations
    for i in range(50):
        optimizer.zero_grad()
        output = model(X_all, i=task_indices)
        loss = -mll(output, y_all)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    # Now, extract the task covariance matrix directly
    task_covar_raw = model.task_covar_module.covar_matrix.evaluate()
    print("Raw task covariance matrix:")
    print(task_covar_raw)

    # Method 1: Using task_covar_module directly
    def extract_task_correlation_matrix_direct(model):
        """Extract task correlation matrix directly from the MTGP model"""
        # Get the raw covariance matrix
        task_covar = model.task_covar_module.covar_matrix.evaluate()

        # Convert to correlation matrix (normalize by the diagonal values)
        diag_values = torch.sqrt(torch.diag(task_covar))
        outer_product = torch.outer(diag_values, diag_values)

        correlation_matrix = task_covar / outer_product
        return correlation_matrix

    # Method 2: Using covar_factor (Cholesky decomposition L where K = LL^T)
    def extract_task_correlation_matrix_cholesky(model):
        """Extract task correlation matrix using Cholesky decomposition"""
        # In IndexKernel, covar_factor is the Cholesky factor L
        L = model.task_covar_module.covar_factor

        # Compute K = LL^T
        K = L @ L.t()

        # Convert to correlation matrix
        diag_values = torch.sqrt(torch.diag(K))
        outer_product = torch.outer(diag_values, diag_values)

        correlation_matrix = K / outer_product
        return correlation_matrix

    # Compare both methods
    corr_matrix_direct = extract_task_correlation_matrix_direct(model)
    corr_matrix_cholesky = extract_task_correlation_matrix_cholesky(model)

    print("\nTask correlation matrix (direct method):")
    print(corr_matrix_direct)

    print("\nTask correlation matrix (Cholesky method):")
    print(corr_matrix_cholesky)

    # We expect a high correlation between tasks 1 and 2
    # The diagonal elements should be 1.0
    # The off-diagonal elements should be large positive values

    # Verify the diagonal elements are 1.0
    print("\nChecking diagonal elements (should be 1.0):")
    print(f"Direct method: {torch.diag(corr_matrix_direct)}")
    print(f"Cholesky method: {torch.diag(corr_matrix_cholesky)}")

    # Verify the correlation is positive and high
    print("\nChecking correlation between tasks (should be positive and high):")
    print(f"Direct method: {corr_matrix_direct[0, 1]}")
    print(f"Cholesky method: {corr_matrix_cholesky[0, 1]}")

    # For completeness, let's also create a negative correlation example
    # Task 3 is f(x) = -x^2 (negatively correlated with Task 1)
    x3 = torch.linspace(0, 1, 10).reshape(-1, 1)
    y3 = -x3.pow(2).reshape(-1)

    # Create a new model with 3 tasks
    X_all_3tasks = torch.cat([x1, x2, x3], dim=0)
    y_all_3tasks = torch.cat([y1, y2, y3], dim=0)

    task_indices_3tasks = torch.cat([
        torch.zeros(10, dtype=torch.long),
        torch.ones(10, dtype=torch.long),
        2 * torch.ones(10, dtype=torch.long)
    ])

    # Create a new model for 3 tasks

    model_3tasks = MultitaskGP(X_all_3tasks, y_all_3tasks, likelihood, 3)

    # Train the model
    model_3tasks.train()

    optimizer = torch.optim.Adam([
        {'params': model_3tasks.parameters()},
    ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_3tasks)

    for i in range(50):
        optimizer.zero_grad()
        output = model_3tasks(X_all_3tasks, i=task_indices_3tasks)
        loss = -mll(output, y_all_3tasks)
        loss.backward()
        optimizer.step()

    model_3tasks.eval()

    # Extract correlation matrix for the 3-task model
    corr_matrix_3tasks = extract_task_correlation_matrix_direct(model_3tasks)

    print("\nCorrelation matrix for 3-task model:")
    print(corr_matrix_3tasks)

    # We expect:
    # - High positive correlation between tasks 1 and 2
    # - Negative correlation between tasks 1 and 3
    # - Negative correlation between tasks 2 and 3

    print("\nVerifying correlations in 3-task model:")
    print(f"Task 1-2 correlation: {corr_matrix_3tasks[0, 1]}")
    print(f"Task 1-3 correlation: {corr_matrix_3tasks[0, 2]}")
    print(f"Task 2-3 correlation: {corr_matrix_3tasks[1, 2]}")

    # Visualize the correlation matrices
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(corr_matrix_direct.detach().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation Matrix (2 Tasks)')
    plt.xticks([0, 1], ['Task 1', 'Task 2'])
    plt.yticks([0, 1], ['Task 1', 'Task 2'])

    plt.subplot(1, 2, 2)
    plt.imshow(corr_matrix_3tasks.detach().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation Matrix (3 Tasks)')
    plt.xticks([0, 1, 2], ['Task 1', 'Task 2', 'Task 3'])
    plt.yticks([0, 1, 2], ['Task 1', 'Task 2', 'Task 3'])

    plt.tight_layout()
    plt.savefig('task_correlation_matrices.png')

    # Also check the transfer probabilities
    def compute_transfer_probability(task_correlation_matrix, task_i, task_j):
        """Compute transfer probability using equation (23) in the paper"""
        k_ij = task_correlation_matrix[task_i, task_j]
        k_ii = task_correlation_matrix[task_i, task_i]
        k_jj = task_correlation_matrix[task_j, task_j]

        # Compute transfer probability
        transfer_prob = k_ij / torch.sqrt(k_ii * k_jj)

        # Ensure probability is between 0 and 1
        transfer_prob = torch.clamp(transfer_prob, 0, 1)

        return transfer_prob

    print("\nComputing transfer probabilities for 3-task model:")
    print(f"p_T(1,2): {compute_transfer_probability(corr_matrix_3tasks, 0, 1)}")
    print(f"p_T(1,3): {compute_transfer_probability(corr_matrix_3tasks, 0, 2)}")
    print(f"p_T(2,3): {compute_transfer_probability(corr_matrix_3tasks, 1, 2)}")

    # Test the LCB computation
    test_mtgp_lcb(model, model_3tasks, likelihood)

    return model, model_3tasks


def test_mtgp_lcb(model_2tasks, model_3tasks, likelihood):
    """
    Test the computation of Lower Confidence Bound (LCB) for MTGP models.

    Args:
        model_2tasks: Trained 2-task GP model
        model_3tasks: Trained 3-task GP model
        likelihood: Gaussian likelihood
    """
    print("\n\n=== Testing MTGP Lower Confidence Bound (LCB) ===")

    # Create test points for each task
    test_x = torch.linspace(0, 1.5, 100).reshape(-1, 1)  # Extend beyond training range

    # === Test with 2-task model ===
    print("\n--- Testing LCB with 2-task model ---")

    # Test points for Task 1
    test_task_indices_1 = torch.zeros(test_x.size(0), dtype=torch.long)
    lcb_task1 = compute_mtgp_lcb(model_2tasks, likelihood, test_x, test_task_indices_1, beta=2.0)
    print(f"LCB shape for Task 1: {lcb_task1.shape}")

    # Test points for Task 2
    test_task_indices_2 = torch.ones(test_x.size(0), dtype=torch.long)
    lcb_task2 = compute_mtgp_lcb(model_2tasks, likelihood, test_x, test_task_indices_2, beta=2.0)
    print(f"LCB shape for Task 2: {lcb_task2.shape}")

    # Visualize LCB for both tasks
    plt.figure(figsize=(10, 6))

    # Get predictions for plotting
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Task 1 predictions
        observed_pred_task1 = likelihood(model_2tasks(test_x, i=test_task_indices_1))
        mean_task1 = observed_pred_task1.mean
        lower_task1, upper_task1 = observed_pred_task1.confidence_region()

        # Task 2 predictions
        observed_pred_task2 = likelihood(model_2tasks(test_x, i=test_task_indices_2))
        mean_task2 = observed_pred_task2.mean
        lower_task2, upper_task2 = observed_pred_task2.confidence_region()

    # Plot Task 1
    plt.subplot(2, 1, 1)
    plt.plot(test_x.numpy(), mean_task1.numpy(), 'b-', label='Mean')
    plt.fill_between(test_x.numpy().flatten(), lower_task1.numpy(), upper_task1.numpy(), alpha=0.3, color='b',
                     label='Confidence')
    plt.plot(test_x.numpy(), lcb_task1.numpy(), 'r--', label='LCB (β=2.0)')
    plt.title('Task 1 (f(x) = x²)')
    plt.legend()

    # Plot Task 2
    plt.subplot(2, 1, 2)
    plt.plot(test_x.numpy(), mean_task2.numpy(), 'g-', label='Mean')
    plt.fill_between(test_x.numpy().flatten(), lower_task2.numpy(), upper_task2.numpy(), alpha=0.3, color='g',
                     label='Confidence')
    plt.plot(test_x.numpy(), lcb_task2.numpy(), 'r--', label='LCB (β=2.0)')
    plt.title('Task 2 (f(x) = 2x² + 1)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('mtgp_lcb_2tasks.png')

    # === Test with 3-task model ===
    print("\n--- Testing LCB with 3-task model ---")

    # Test with different beta values for Task 3
    test_task_indices_3 = 2 * torch.ones(test_x.size(0), dtype=torch.long)

    beta_values = [0.5, 2.0, 4.0]
    lcb_results = []

    plt.figure(figsize=(12, 6))

    # Get predictions for Task 3
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_task3 = likelihood(model_3tasks(test_x, i=test_task_indices_3))
        mean_task3 = observed_pred_task3.mean
        lower_task3, upper_task3 = observed_pred_task3.confidence_region()

    # Plot mean and confidence region
    plt.plot(test_x.numpy(), mean_task3.numpy(), 'k-', label='Mean')
    plt.fill_between(test_x.numpy().flatten(), lower_task3.numpy(), upper_task3.numpy(), alpha=0.2, color='k',
                     label='Confidence')

    # Calculate and plot LCB for different beta values
    for beta in beta_values:
        lcb = compute_mtgp_lcb(model_3tasks, likelihood, test_x, test_task_indices_3, beta=beta)
        lcb_results.append(lcb)
        plt.plot(test_x.numpy(), lcb.numpy(), '--', label=f'LCB (β={beta})')

    plt.title('Task 3 (f(x) = -x²) with Different β Values')
    plt.legend()
    plt.savefig('mtgp_lcb_task3_beta_comparison.png')

    print("\nLCB computation test completed. Plots saved.")

    # Test acquisition function optimization based on LCB
    print("\n--- Testing acquisition function optimization using LCB ---")

    # Create a grid of points for each task
    grid_x = torch.linspace(0, 1.5, 100).reshape(-1, 1)

    # Compute LCB for each task
    grid_task_indices_1 = torch.zeros(grid_x.size(0), dtype=torch.long)
    grid_task_indices_2 = torch.ones(grid_x.size(0), dtype=torch.long)
    grid_task_indices_3 = 2 * torch.ones(grid_x.size(0), dtype=torch.long)

    lcb_grid_task1 = compute_mtgp_lcb(model_3tasks, likelihood, grid_x, grid_task_indices_1, beta=2.0)
    lcb_grid_task2 = compute_mtgp_lcb(model_3tasks, likelihood, grid_x, grid_task_indices_2, beta=2.0)
    lcb_grid_task3 = compute_mtgp_lcb(model_3tasks, likelihood, grid_x, grid_task_indices_3, beta=2.0)

    # Find the points with minimum LCB for each task
    min_idx_task1 = torch.argmin(lcb_grid_task1)
    min_idx_task2 = torch.argmin(lcb_grid_task2)
    min_idx_task3 = torch.argmin(lcb_grid_task3)

    min_point_task1 = grid_x[min_idx_task1].item()
    min_point_task2 = grid_x[min_idx_task2].item()
    min_point_task3 = grid_x[min_idx_task3].item()

    print(f"Next best point for Task 1: x = {min_point_task1}")
    print(f"Next best point for Task 2: x = {min_point_task2}")
    print(f"Next best point for Task 3: x = {min_point_task3}")

    # Visualize the acquisition function
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(grid_x.numpy(), lcb_grid_task1.numpy(), 'b-')
    plt.axvline(x=min_point_task1, color='r', linestyle='--')
    plt.title(f'LCB Acquisition for Task 1 (min at x = {min_point_task1:.4f})')

    plt.subplot(3, 1, 2)
    plt.plot(grid_x.numpy(), lcb_grid_task2.numpy(), 'g-')
    plt.axvline(x=min_point_task2, color='r', linestyle='--')
    plt.title(f'LCB Acquisition for Task 2 (min at x = {min_point_task2:.4f})')

    plt.subplot(3, 1, 3)
    plt.plot(grid_x.numpy(), lcb_grid_task3.numpy(), 'm-')
    plt.axvline(x=min_point_task3, color='r', linestyle='--')
    plt.title(f'LCB Acquisition for Task 3 (min at x = {min_point_task3:.4f})')

    plt.tight_layout()
    plt.savefig('mtgp_lcb_acquisition.png')


def test_hadamard_mtgp():
    """
    Test the Hadamard pattern MultitaskGP model with synthetic data.
    """
    # Create a simple 2-task scenario with known correlation
    # - Task 1 is f(x) = x^2
    # - Task 2 is f(x) = 2*x^2 + 1 (same shape, different scale)

    # Generate synthetic data
    torch.manual_seed(0)  # For reproducibility

    # Task 1 data
    x1 = torch.linspace(0, 1, 10).reshape(-1)
    y1 = x1.pow(2) + torch.randn(x1.size()) * 0.05

    # Task 2 data
    x2 = torch.linspace(0, 1, 10).reshape(-1)
    y2 = (2 * x2.pow(2) + 1) + torch.randn(x2.size()) * 0.05

    # Combined data
    train_x = torch.cat([x1, x2])
    train_y = torch.cat([y1, y2])

    # Task indices (0 for task 1, 1 for task 2)
    task_indices = torch.cat([
        torch.zeros(10, dtype=torch.long),
        torch.ones(10, dtype=torch.long)
    ])

    # Initialize model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Create the multitask model using Hadamard pattern
    # Note: we pass (train_x, task_indices) as a tuple for train_inputs
    model = MultitaskGP((train_x, task_indices), train_y, likelihood, num_tasks=2)

    # Train the model
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Train for a small number of iterations
    for i in range(50):
        optimizer.zero_grad()
        output = model(train_x, task_indices)
        loss = -mll(output, train_y)
        loss.backward()
        print(f"Iter {i + 1}/50 - Loss: {loss.item():.3f}")
        optimizer.step()

    model.eval()
    likelihood.eval()

    # Extract task correlation matrix
    corr_matrix = extract_task_correlation_matrix(model)
    print("\nTask correlation matrix:")
    print(corr_matrix)

    # Create test points - we can use many more points now
    test_x = torch.linspace(0, 1.5, 100)

    # Create test task indices for both tasks
    test_task1_indices = torch.zeros(test_x.size(0), dtype=torch.long)
    test_task2_indices = torch.ones(test_x.size(0), dtype=torch.long)

    # Get predictions for both tasks
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Task 1 predictions
        pred_task1 = likelihood(model(test_x, test_task1_indices))
        mean_task1 = pred_task1.mean
        lower_task1, upper_task1 = pred_task1.confidence_region()

        # Task 2 predictions
        pred_task2 = likelihood(model(test_x, test_task2_indices))
        mean_task2 = pred_task2.mean
        lower_task2, upper_task2 = pred_task2.confidence_region()

    # Compute LCB for both tasks
    lcb_task1 = compute_mtgp_lcb(model, likelihood, test_x, test_task1_indices, beta=2.0)
    lcb_task2 = compute_mtgp_lcb(model, likelihood, test_x, test_task2_indices, beta=2.0)

    # Visualize predictions and LCB
    plt.figure(figsize=(12, 8))

    # Plot Task 1
    plt.subplot(2, 1, 1)
    plt.plot(x1.numpy(), y1.numpy(), 'k*', label='Training data')
    plt.plot(test_x.numpy(), mean_task1.numpy(), 'b-', label='Mean')
    plt.fill_between(test_x.numpy(), lower_task1.numpy(), upper_task1.numpy(), alpha=0.3, color='b', label='Confidence')
    plt.plot(test_x.numpy(), lcb_task1.numpy(), 'r--', label='LCB (β=2.0)')
    plt.title('Task 1 (f(x) = x²)')
    plt.legend()

    # Plot Task 2
    plt.subplot(2, 1, 2)
    plt.plot(x2.numpy(), y2.numpy(), 'k*', label='Training data')
    plt.plot(test_x.numpy(), mean_task2.numpy(), 'g-', label='Mean')
    plt.fill_between(test_x.numpy(), lower_task2.numpy(), upper_task2.numpy(), alpha=0.3, color='g', label='Confidence')
    plt.plot(test_x.numpy(), lcb_task2.numpy(), 'r--', label='LCB (β=2.0)')
    plt.title('Task 2 (f(x) = 2x² + 1)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('hadamard_mtgp_predictions.png')

    # Find the points with minimum LCB for each task
    min_idx_task1 = torch.argmin(lcb_task1)
    min_idx_task2 = torch.argmin(lcb_task2)

    min_point_task1 = test_x[min_idx_task1].item()
    min_point_task2 = test_x[min_idx_task2].item()

    print(f"Next best point for Task 1: x = {min_point_task1:.4f}")
    print(f"Next best point for Task 2: x = {min_point_task2:.4f}")

    # === Create a 3-task model ===
    # Task 3 is f(x) = -x^2 (negatively correlated with Task 1)
    x3 = torch.linspace(0, 1, 10).reshape(-1)
    y3 = -x3.pow(2) + torch.randn(x3.size()) * 0.05

    # Combined data for 3 tasks
    train_x_3tasks = torch.cat([x1, x2, x3])
    train_y_3tasks = torch.cat([y1, y2, y3])

    task_indices_3tasks = torch.cat([
        torch.zeros(10, dtype=torch.long),
        torch.ones(10, dtype=torch.long),
        2 * torch.ones(10, dtype=torch.long)
    ])

    # Create the 3-task model
    model_3tasks = MultitaskGP(
        (train_x_3tasks, task_indices_3tasks),
        train_y_3tasks,
        likelihood,
        num_tasks=3
    )

    # Train the model
    model_3tasks.train()
    optimizer = torch.optim.Adam(model_3tasks.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_3tasks)

    for i in range(50):
        optimizer.zero_grad()
        output = model_3tasks(train_x_3tasks, task_indices_3tasks)
        loss = -mll(output, train_y_3tasks)
        loss.backward()
        if i % 10 == 0:
            print(f"Iter {i + 1}/50 - Loss: {loss.item():.3f}")
        optimizer.step()

    model_3tasks.eval()

    # Extract correlation matrix for the 3-task model
    corr_matrix_3tasks = extract_task_correlation_matrix(model_3tasks)
    print("\nCorrelation matrix for 3-task model:")
    print(corr_matrix_3tasks)

    # Visualize the correlation matrices
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(corr_matrix.detach().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation Matrix (2 Tasks)')
    plt.xticks([0, 1], ['Task 1', 'Task 2'])
    plt.yticks([0, 1], ['Task 1', 'Task 2'])

    plt.subplot(1, 2, 2)
    plt.imshow(corr_matrix_3tasks.detach().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation Matrix (3 Tasks)')
    plt.xticks([0, 1, 2], ['Task 1', 'Task 2', 'Task 3'])
    plt.yticks([0, 1, 2], ['Task 1', 'Task 2', 'Task 3'])

    plt.tight_layout()
    plt.savefig('hadamard_mtgp_correlation_matrices.png')

    # Test with task 3
    test_task3_indices = 2 * torch.ones(test_x.size(0), dtype=torch.long)

    # Compute LCB for task 3 with different beta values
    beta_values = [0.5, 2.0, 4.0]

    plt.figure(figsize=(10, 6))

    # Get predictions for Task 3
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_task3 = likelihood(model_3tasks(test_x, test_task3_indices))
        mean_task3 = pred_task3.mean
        lower_task3, upper_task3 = pred_task3.confidence_region()

    # Plot mean and confidence region
    plt.plot(test_x.numpy(), mean_task3.numpy(), 'k-', label='Mean')
    plt.fill_between(test_x.numpy(), lower_task3.numpy(), upper_task3.numpy(), alpha=0.2, color='k', label='Confidence')
    plt.plot(x3.numpy(), y3.numpy(), 'k*', label='Training data')

    # Calculate and plot LCB for different beta values
    for beta in beta_values:
        lcb = compute_mtgp_lcb(model_3tasks, likelihood, test_x, test_task3_indices, beta=beta)
        plt.plot(test_x.numpy(), lcb.numpy(), '--', label=f'LCB (β={beta})')

    plt.title('Task 3 (f(x) = -x²) with Different β Values')
    plt.legend()
    plt.savefig('hadamard_mtgp_task3_lcb.png')

    return model, model_3tasks


# If this file is run directly, run the test
if __name__ == "__main__":
    model, model_3tasks = test_hadamard_mtgp()
