from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def cvt(k, dim, coef=10, verbose=False):
    x = np.random.rand(k*coef, dim)
    k_means = KMeans(init='k-means++', n_clusters=k, n_init=1, verbose=False)
    k_means.fit(x)
    X = k_means.cluster_centers_
    return X, x, k_means


def plot_func(min_range, max_range, no_of_samples):
    inputs = np.linspace(min_range, max_range, no_of_samples)
    outputs = 1 - np.exp(-inputs)
    plt.plot(inputs, outputs)
    plt.show()


if __name__ == "__main__":
    plot_func(0, 2, 1000)
    # centers, results, model = cvt(10, 2)
    # print("DEBUG")
    # plt.plot(results[:, 0], results[:, 1], 'bo', alpha=0.1)
    # plt.plot(centers[:, 0], centers[:, 1], 'rx', alpha=1)
    # plt.show()