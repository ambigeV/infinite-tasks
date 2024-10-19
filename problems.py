import numpy as np
from math import cos, sin
import math
import torch
import time


def get_linear_mat(k, task_params=2):
    mat_main_placeholder = np.zeros((task_params, k))
    mat_placeholder = np.zeros((2, k))

    assert k > 1
    mat_placeholder[1, :] = np.arange(0, k) / (k - 1)
    mat_placeholder[0, :] = np.arange(k - 1, -1, -1) / (k - 1)

    if task_params > 2:
        task_limit_num = task_params - 1
        task_indices = np.arange(k)
        for i in range(task_limit_num):
            temp_anchor = task_indices % task_limit_num
            mat_main_placeholder[i:(i + 2), temp_anchor == i] = mat_placeholder[:, temp_anchor == i]

        return mat_main_placeholder.T
    else:
        return mat_placeholder.T


def nonlinear_map(x):
    return (np.sin(5 * (x + 0.5)) + 1) / 2


def middle_nonlinear_map(x):
    return 0.3 * (1 + np.sin(5 * np.pi * x - np.pi / 2)) + 0.3 * np.square(x - 0.2)


def super_nonlinear_map(x):
    return 0.3 * (1 + np.sin(10 * np.pi * x - np.pi / 2)) + 0.3 * np.square(x - 0.2)


def get_constant_mat(k, task_params=2):
    mat_main_placeholder = np.zeros((task_params, k))
    mat_placeholder = np.zeros((2, k))

    assert k > 1
    mid_val = k // 2
    mat_placeholder[1, mid_val:] = 1
    mat_placeholder[0, :mid_val] = 1

    if task_params > 2:
        task_limit_num = task_params - 1
        task_indices = np.arange(k)
        for i in range(task_limit_num):
            temp_anchor = task_indices % task_limit_num
            mat_main_placeholder[i:(i + 2), temp_anchor == i] = mat_placeholder[:, temp_anchor == i]

        return mat_main_placeholder.T
    else:
        return mat_placeholder.T


def get_problem(name, problem_params=None, task_params=2):
    name = name.lower()

    # key: 'perfect_sphere'
    # key: 'linear_sphere'
    # key: 'nonlinear_sphere'

    # key: 'perfect_ackley'
    # key: 'linear_ackley'
    # key: 'nonlinear_ackley'

    # key: 'perfect_rastrigin'
    # key: 'linear_rastrigin'
    # key: 'nonlinear_rastrigin'

    PROBLEM = {
        'sep_arm': ActualArm(n_dim=problem_params),
        'perfect_sphere': Sphere(n_dim=problem_params, mode="perfect"),
        'linear_sphere': Sphere(n_dim=problem_params, mode="linear"),
        'nonlinear_sphere': Sphere(n_dim=problem_params, mode="nonlinear"),
        'middle_nonlinear_sphere': Sphere(n_dim=problem_params, mode="middle_nonlinear"),
        'super_nonlinear_sphere': Sphere(n_dim=problem_params, mode="super_nonlinear"),
        'linear_sphere_high': Sphere(n_dim=problem_params, mode="linear", task_param=task_params),
        'nonlinear_sphere_high': Sphere(n_dim=problem_params, mode="nonlinear", task_param=task_params),
        'middle_nonlinear_sphere_high': Sphere(n_dim=problem_params, mode="middle_nonlinear", task_param=task_params),
        'super_nonlinear_sphere_high': Sphere(n_dim=problem_params, mode="super_nonlinear", task_param=task_params),
        'perfect_ackley': Ackley(n_dim=problem_params, mode="perfect"),
        'linear_ackley': Ackley(n_dim=problem_params, mode="linear"),
        'nonlinear_ackley': Ackley(n_dim=problem_params, mode="nonlinear"),
        'middle_nonlinear_ackley': Ackley(n_dim=problem_params, mode="middle_nonlinear"),
        'super_nonlinear_ackley': Ackley(n_dim=problem_params, mode="super_nonlinear"),
        'linear_ackley_high': Ackley(n_dim=problem_params, mode="linear", task_param=task_params),
        'nonlinear_ackley_high': Ackley(n_dim=problem_params, mode="nonlinear", task_param=task_params),
        'middle_nonlinear_ackley_high': Ackley(n_dim=problem_params, mode="middle_nonlinear", task_param=task_params),
        'super_nonlinear_ackley_high': Ackley(n_dim=problem_params, mode="super_nonlinear", task_param=task_params),
        'perfect_rastrigin': Rastrigin(n_dim=problem_params, mode="perfect"),
        'linear_rastrigin': Rastrigin(n_dim=problem_params, mode="linear"),
        'perfect_rastrigin_10': Rastrigin(n_dim=problem_params, mode="perfect", factor=10),
        'linear_rastrigin_10': Rastrigin(n_dim=problem_params, mode="linear", factor=10),
        'perfect_rastrigin_20': Rastrigin(n_dim=problem_params, mode="perfect", factor=20),
        'linear_rastrigin_20': Rastrigin(n_dim=problem_params, mode="linear", factor=20),
        'nonlinear_rastrigin_20_high': Rastrigin(n_dim=problem_params, mode="nonlinear",
                                                 task_param=task_params, factor=20),
        'middle_nonlinear_rastrigin_20_high': Rastrigin(n_dim=problem_params, mode="middle_nonlinear",
                                                        task_param=task_params, factor=20),
    }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name]


class Arm:
    def __init__(self, lengths):
        # [l1, l2, l3, ..., lN] => n_dofs(N), lengths([0, l1, l2, ..., lN]), joint_xy([])
        self.n_dofs = len(lengths)
        # Give a vector "lengths"
        # 10-DoF has 10 values indicating each length per DoF
        self.lengths = np.concatenate(([0], lengths))
        # Why concatenate a 0 here???
        self.joint_xy = []

    def fw_kinematics(self, p):
        assert(len(p) == self.n_dofs)
        p = np.append(p, 0)
        self.joint_xy = []
        mat = np.matrix(np.identity(4))
        for i in range(0, self.n_dofs + 1):
            m = [[cos(p[i]), -sin(p[i]), 0, self.lengths[i]],
                 [sin(p[i]),  cos(p[i]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            mat = mat * np.matrix(m)
            v = mat * np.matrix([0, 0, 0, 1]).transpose()
            self.joint_xy += [v[0:2].A.flatten()]
        return self.joint_xy[self.n_dofs], self.joint_xy


class Sphere:
    def __init__(self, n_dim=10, mode="perfect", task_param=2, factor=4):
        self.n_dim = n_dim
        self.n_obj = 1
        self.mode = mode
        shift_mat = None
        if self.mode == "perfect":
            shift_mat = get_constant_mat(self.n_dim, task_param)
        elif self.mode == "linear" or \
                self.mode == "nonlinear" or \
                self.mode == "super_nonlinear" or \
                self.mode == "middle_nonlinear":
            shift_mat = get_linear_mat(self.n_dim, task_param)
        else:
            pass
        self.shift_mat = shift_mat
        self.factor = factor

    def evaluate(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        sol = x[:self.n_dim]
        hook = x[self.n_dim:]
        shift = np.squeeze(np.matmul(self.shift_mat, np.expand_dims(hook, axis=1)), axis=1)
        if self.mode == "nonlinear":
            shift = nonlinear_map(shift)
        if self.mode == "middle_nonlinear":
            shift = middle_nonlinear_map(shift)
        if self.mode == "super_nonlinear":
            shift = super_nonlinear_map(shift)

        return np.sum(np.power(sol * self.factor - shift * self.factor, 2))


class Ackley:
    def __init__(self, n_dim=10, mode="perfect", task_param=2, factor=4):
        self.n_dim = n_dim
        self.n_obj = 1
        self.mode = mode
        shift_mat = None
        if self.mode == "perfect":
            shift_mat = get_constant_mat(self.n_dim, task_param)
        elif self.mode == "linear" or \
                self.mode == "nonlinear" or \
                self.mode == "super_nonlinear" or \
                self.mode == "middle_nonlinear":
            shift_mat = get_linear_mat(self.n_dim, task_param)
        else:
            pass
        self.shift_mat = shift_mat
        self.factor = factor

    def divergence(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        sol = x[:self.n_dim]
        hook = x[self.n_dim:]
        shift = np.squeeze(np.matmul(self.shift_mat, np.expand_dims(hook, axis=1)), axis=1)

        return np.sum(np.power(sol * self.factor - shift * self.factor, 2))

    def evaluate(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        sol = x[:self.n_dim]
        hook = x[self.n_dim:]
        shift = np.squeeze(np.matmul(self.shift_mat, np.expand_dims(hook, axis=1)), axis=1)
        if self.mode == "nonlinear":
            shift = nonlinear_map(shift)
        if self.mode == "middle_nonlinear":
            shift = middle_nonlinear_map(shift)
        if self.mode == "super_nonlinear":
            shift = super_nonlinear_map(shift)

        return -20 * np.exp(-0.2 * np.sqrt(np.mean(np.power(sol * self.factor - shift * self.factor, 2)))) - \
            np.exp(np.mean(np.cos(2 * np.pi * self.factor * (sol - shift)))) + 20 + np.exp(1)


class Rastrigin:
    def __init__(self, n_dim=10, mode="perfect", task_param=2, factor=4):
        self.n_dim = n_dim
        self.n_obj = 1
        self.mode = mode
        shift_mat = None
        if self.mode == "perfect":
            shift_mat = get_constant_mat(self.n_dim, task_param)
        elif self.mode == "linear" or \
                self.mode == "nonlinear" or \
                self.mode == "super_nonlinear" or \
                self.mode == "middle_nonlinear":
            shift_mat = get_linear_mat(self.n_dim, task_param)
        else:
            pass
        self.shift_mat = shift_mat
        self.factor = factor

    def divergence(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        sol = x[:self.n_dim]
        hook = x[self.n_dim:]
        shift = np.squeeze(np.matmul(self.shift_mat, np.expand_dims(hook, axis=1)), axis=1)

        return np.sum(np.power(sol * self.factor - shift * self.factor, 2))

    def evaluate(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        sol = x[:self.n_dim]
        hook = x[self.n_dim:]
        shift = np.squeeze(np.matmul(self.shift_mat, np.expand_dims(hook, axis=1)), axis=1)

        if self.mode == "nonlinear":
            shift = nonlinear_map(shift)
        if self.mode == "middle_nonlinear":
            shift = middle_nonlinear_map(shift)
        if self.mode == "super_nonlinear":
            shift = super_nonlinear_map(shift)

        return np.sum(np.power((sol - shift) * self.factor, 2) -
                      10 * np.cos(2 * np.pi * self.factor * (sol - shift)) + 10)


class ActualArm:
    def __init__(self, n_dim=10):
        self.n_dim = n_dim
        self.n_obj = 1

    def evaluate(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        angles = x[:self.n_dim]
        angular_range = (x[self.n_dim] * 0.5 + 0.5) / len(angles)
        lengths = np.ones(len(angles)) * (x[self.n_dim + 1] * 0.5 + 0.5) / len(angles)
        target = 0.5 * np.ones(2)
        a = Arm(lengths)
        command = (angles - 0.5) * angular_range * math.pi * 2
        ef, _ = a.fw_kinematics(command)
        f = 1 - np.exp(-np.linalg.norm(ef - target))

        return f


if __name__ == "__main__":
    # problem_name = ["sep_arm"]
    # problem_params = 10
    # task_dim = 2
    # problem = get_problem(problem_name[0], problem_params=problem_params)
    # st = time.time()
    # for i in range(10000):
    #     solution_params = torch.rand(problem_params + task_dim)
    #     print("solution in iter {} is {}".format(i + 1, problem.evaluate(solution_params)))
    # en = time.time()
    # print("lasting time is {}s".format(en - st))

    print(get_linear_mat(4, 5))

