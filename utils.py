import math

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import numpy as np


def cvt(x, k, sample_weight=None, verbose=False):
    k_means = KMeans(init='k-means++', n_clusters=k, n_init=1, verbose=False)
    # print(sample_weight)
    k_means.fit(x, sample_weight=sample_weight)
    X = k_means.cluster_centers_
    return X, k_means


def plot_box(tensors, methods, if_log=False):
    if if_log:
        plt.boxplot(tensors,  whis=[0, 100], labels=methods)
    else:
        plt.boxplot(tensors, labels=methods)
    # plt.title('')
    plt.xlabel('Methods')
    plt.ylabel('Objective Values')
    plt.grid(True)
    # plt.legend()


def plot_tot(values_tot, figure_id, keys, if_mean=True):
    if if_mean:
        values_mean = torch.mean(values_tot, dim=0)
        print(torch.mean(values_mean))
    else:
        values_mean = values_tot
    plot_hist(values_mean.numpy(), figure_id, keys[figure_id-1])


def debug_hist(targets, methods, start=0, end=200, dim=-1):
    for m_id in range(len(methods)):
        target = targets[m_id]
        method = methods[m_id]
        plot_hist(target[start:end, dim], m_id, method)


def debug_tot(targets, methods, start=0, end=200, dim=4):
    for target, method in zip(targets, methods):
        debug_plot(target[start:end, :dim], method)


def debug_each(targets, methods, ind, dim=4):
    for target, method in zip(targets, methods):
        debug_plot(target[:, ind, :dim], method)


def debug_plot(target: torch.tensor, title=None):
    a, b = target.shape
    print("The shape is {}.".format(target.shape))
    # a for sample size, b for sol dim
    fig, ax = plt.subplots()
    ind = torch.arange(1, b+1)
    new_target = target.T
    for i in ind:
        ax.axvline(x=i, color='k', alpha=0.2)
    ax.plot(ind, new_target, 'b-', alpha=0.3)
    if title is not None:
        ax.set_title(title)
    # plt.show()


def plot_grad(tensors):
    plt.figure()

    M, N = tensors.shape

    # Iterate over the 4 dimensions and plot each
    for i in range(N):
        plt.plot(np.arange(M), tensors.numpy()[:, i], label=f'Dimension {i + 1}')

    # Setting labels and title
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Trend of Each Dimension Over 10 Steps')
    plt.legend()

    plt.show()


def plot_iteration_convergence(results, method_list, task_id):

    # Combine tensors into one tensor of shape [K, N, M]
    combined_results = torch.stack(results)

    # Calculate mean and standard deviation for each method
    means = combined_results.mean(dim=2)  # Shape [K, N]
    stds = combined_results.std(dim=2)  # Shape [K, N]

    K, N = means.shape

    # Plotting
    methods = method_list
    x = np.arange(N)  # x-axis: iterations

    plt.figure()
    # Plot mean curves with standard deviation bands
    for i in range(K):
        plt.plot(x, means[i], label=methods[i])
        plt.fill_between(x, means[i] - stds[i], means[i] + stds[i], alpha=0.2)

    # Setting labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.title('Performance Comparison of Different Methods for task {}'.format(task_id + 1))
    plt.legend()


def ax_plot_iteration_convergence(ax, results, method_list, task_id):

    # Combine tensors into one tensor of shape [K, N, M]
    combined_results = torch.stack(results)

    # Calculate mean and standard deviation for each method
    means = combined_results.mean(dim=2)  # Shape [K, N]
    stds = combined_results.std(dim=2)  # Shape [K, N]

    K, N = means.shape

    # Plotting
    methods = method_list
    x = np.arange(N)  # x-axis: iterations

    # plt.figure()
    # Plot mean curves with standard deviation bands
    for i in range(K):
        ax.plot(x, means[i], label=methods[i])
        ax.fill_between(x, means[i] - stds[i], means[i] + stds[i], alpha=0.2)

    # Setting labels and title
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Value')
    ax.set_title('Task {}'.format(task_id + 1))
    # if task_id == 5:
    #     ax.legend()


def plot_convergence(results, method_list):
    # Parameterize K from the length of the list
    K = len(results)

    # Combine tensors into one tensor of shape [K, M, N]
    combined_results = torch.stack(results)
    _, M, N = combined_results.shape

    # Calculate mean and standard deviation for each method
    means = combined_results.mean(dim=2)  # Shape [K, M]
    stds = combined_results.std(dim=2)  # Shape [K, M]

    # Plotting
    x = np.arange(M)  # x-axis

    plt.figure(figsize=(10, 6))

    # for i in range(K):
    #     plt.errorbar(x, means[i], yerr=stds[i], label=method_list[i], capsize=5)
    bar_width = 0.2
    # Plot bars with error bars
    for i in range(K):
        plt.bar(x + i * bar_width, means[i], yerr=stds[i], width=bar_width, label=method_list[i], capsize=5)

    plt.xlabel('Result Index')
    plt.ylabel('Value')
    plt.title('Comparison Results on Each Task')
    plt.xticks(x + bar_width * (K - 1) / 2, [f'Task {i+1}' for i in range(M)])
    plt.legend()
    plt.show()


def plot_details(test_input, test_output, name, if_constrain=False):
    if if_constrain:
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(test_input[:, 0], test_input[:, 1], c=1-test_output,
                              cmap='inferno', s=10)
    else:
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        maxmax = 0.5
        ind = test_output < maxmax
        scatter = plt.scatter(test_input[ind, 0], test_input[ind, 1], c=test_output[ind],
                              cmap='inferno', vmin=0, vmax=maxmax, s=10)
        # scatter = plt.contour(test_input[:, 0], test_input[:, 1], c=1-test_output, levels=10, cmap='viridis')
    # plt.figure(figsize=(8, 6))
    # num = len(test_output)
    # num = int(math.sqrt(num))
    # X = np.reshape(test_input[:, 0], (num, num))
    # Y = np.reshape(test_input[:, 1], (num, num))
    # Z = np.reshape(test_output[:], (num, num))
    # levels = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
    # contour = plt.contour(X, Y, Z, levels=levels, colors='black')  # Contour lines
    # contour_filled = plt.contourf(X, Y, Z, levels=levels, cmap='viridis')  # Filled contours
    # plt.clabel(contour, inline=True, fontsize=10)

    # # Add color bar to show the color scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Output Value')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Output Value of {}'.format(name))
    plt.xlabel('Task Dimension 1')
    plt.ylabel('Task Dimension 2')
    # plt.grid(True)
    # plt.show()


def plot_hist(values_np, figure_id, key):
    plt.figure()
    plt.hist(values_np, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Value Distribution of the {}'.format(key))
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.show()


if __name__ == "__main__":

    # Example tensor of size K
    K = 10
    original_tensor = torch.randn(K)  # Replace with your actual data

    # Compute the cumulative minimum using torch.cummin
    new_tensor, _ = torch.cummin(original_tensor, dim=0)

    print("Original Tensor:", original_tensor)
    print("New Tensor with cumulative minimums:", new_tensor)