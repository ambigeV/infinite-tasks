from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import torch


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
    # plot_func(0, 1, 1000)
    # print(np.linalg.norm(np.ones(2)))
    # centers, results, model = cvt(10, 2)
    # print("DEBUG")
    # plt.plot(results[:, 0], results[:, 1], 'bo', alpha=0.1)
    # plt.plot(centers[:, 0], centers[:, 1], 'rx', alpha=1)
    # plt.show()
    # ind_size = 10
    # dim_size = 2
    # name = "task_list_{}_{}.pth".format(ind_size, dim_size)
    # # sampler = qmc.LatinHypercube(dim_size)
    # # samples = sampler.random(ind_size)
    # # ind_size_list = []
    # ind_size_list = torch.load(name)
    # samples = torch.stack(ind_size_list).numpy()
    # # for i in samples:
    # #     ind_size_list.append(torch.from_numpy(i).float())
    # # torch.save(ind_size_list, name)
    # print(samples.__class__)
    # print(samples.shape)
    # plt.scatter(samples[:, 0], samples[:, 1])
    # for i in range(len(ind_size_list)):
    #     plt.text(samples[i, 0], samples[i, 1], '{}'.format(i+1), ha='right')
    # plt.show()
    # results = torch.load("./{}/{}_{}.pth".format("sep_arm_result_4",
    #                                              "10_50_fixed_context_gp_smooth",
    #                                              0))
    # print("DEBUG")
    # print("ENDING... ...")
    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib.cm as cm

    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X ** 2 - Y ** 2)
    Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
