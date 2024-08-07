import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import numpy as np


def cvt(x, k, verbose=False):
    k_means = KMeans(init='k-means++', n_clusters=k, n_init=1, verbose=False)
    k_means.fit(x)
    X = k_means.cluster_centers_
    return X, k_means


def plot_box(tensors, methods):
    plt.boxplot(tensors, labels=methods)
    plt.title('Box Plot of Tensors for Each Method')
    plt.xlabel('Methods')
    plt.ylabel('Values')
    plt.grid(True)
    # plt.legend()
    plt.show()


def plot_details(test_input, test_output, name):
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(test_input[:, 0], test_input[:, 1], c=test_output,
                          cmap='inferno', vmin=0, vmax=1, s=10)

    # Add color bar to show the color scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Output Value')

    plt.title('Output Value of {}'.format(name))
    plt.xlabel('Task Dimension 1')
    plt.ylabel('Task Dimension 2')
    # plt.grid(True)
    # plt.show()


def plot_tot(values_tot, figure_id, key):
    values_mean = torch.mean(values_tot, dim=0)
    print(torch.mean(values_mean))
    plot_hist(values_mean.numpy(), figure_id, key)


def plot_hist(values_np, figure_id, key):
    plt.figure(figure_id)
    plt.hist(values_np, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Value Distribution of the {}'.format(key))
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.show()