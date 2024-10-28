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
        're21_1': RE21_1(),
        're21_2': RE21_2(),
        're25_1': RE25_1(),
        'truss': Truss(n_dim=3),
        'recontrol': ReControl(n_dim=3),
        'recontrol_env': ReControlEnv(n_dim=3),
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
                                                 task_param=task_params, factor=20, mean=300, std=200),
        'middle_nonlinear_rastrigin_20_high': Rastrigin(n_dim=problem_params, mode="middle_nonlinear",
                                                        task_param=task_params, factor=20, mean=300, std=200),
        'super_nonlinear_rastrigin_20_high': Rastrigin(n_dim=problem_params, mode="super_nonlinear",
                                                       task_param=task_params, factor=20),
        'nonlinear_griewank_high': Griewank(n_dim=problem_params, mode="nonlinear",
                                            task_param=task_params, factor=600, mean=50, std=30),
        'middle_nonlinear_griewank_high': Griewank(n_dim=problem_params, mode="middle_nonlinear",
                                                   task_param=task_params, factor=600, mean=50, std=30),
        'super_nonlinear_griewank_high': Griewank(n_dim=problem_params, mode="super_nonlinear",
                                                  task_param=task_params, factor=600),
        'nonlinear_rosenbrock_high': Rosenbrock(n_dim=problem_params, mode="nonlinear",
                                                task_param=task_params, factor=5),
        'middle_nonlinear_rosenbrock_high': Rosenbrock(n_dim=problem_params, mode="middle_nonlinear",
                                                       task_param=task_params, factor=5),
        'super_nonlinear_rosenbrock_high': Rosenbrock(n_dim=problem_params, mode="super_nonlinear",
                                                      task_param=task_params, factor=5),
        'nonlinear_tang_high': Tang(n_dim=problem_params, mode="nonlinear",
                                    task_param=task_params, factor=2),
        'middle_nonlinear_tang_high': Tang(n_dim=problem_params, mode="middle_nonlinear",
                                           task_param=task_params, factor=2),
        'super_nonlinear_tang_high': Tang(n_dim=problem_params, mode="super_nonlinear",
                                          task_param=task_params, factor=2),
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


class Griewank:
    def __init__(self, n_dim=10, mode="perfect", task_param=2, factor=4, mean=0, std=1):
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
        self.mean = mean
        self.std = std

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

        return (np.sum(np.power(sol * self.factor - shift * self.factor, 2) / 4000) - \
            np.prod(np.cos(sol * self.factor - shift * self.factor / np.sqrt(np.linspace(1, self.n_dim, self.n_dim)))) \
            + 1 - self.mean)/self.std


class Rastrigin:
    def __init__(self, n_dim=10, mode="perfect", task_param=2, factor=4, mean=0, std=1):
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
        self.mean = mean
        self.std = std

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

        return (np.sum(np.power((sol - shift) * self.factor, 2) -
                      10 * np.cos(2 * np.pi * self.factor * (sol - shift)) + 10) - self.mean)/self.std


class Rosenbrock:
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

        new_sol = 1 + self.factor * (sol - shift)

        return np.sum(100 * np.power(np.power(new_sol[:self.n_dim-1], 2) - new_sol[1:], 2) +
                      np.power(new_sol[:self.n_dim-1] - 1, 2))


class Tang:
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

        new_sol = self.factor * (sol - shift) - 2.903534

        return 0.5 * np.sum(np.power(new_sol, 4) - 16 * np.power(new_sol, 2) + 5 * new_sol)


class Truss:
    def __init__(self, n_dim=3):
        self.n_dim = n_dim
        self.sol_low = torch.Tensor([1e-5, 1e-5, 1])
        self.sol_high = torch.Tensor([100, 100, 3])
        self.sol_delta_low = torch.ones(3) * -0.05
        self.sol_delta_high = torch.ones(3) * 0.05

    def evaluate(self, t):
        x = t.clone()

        # Model the noise
        x[:self.n_dim] = x[:self.n_dim] * (self.sol_delta_high - self.sol_delta_low) + self.sol_delta_low
        x[:self.n_dim] = x[:self.n_dim] + x[self.n_dim:]
        x[:self.n_dim] = torch.clamp(x[:self.n_dim], 0, 1)
        x[:self.n_dim] = x[:self.n_dim] * (self.sol_high - self.sol_low) + self.sol_low
        # Compute f1
        f1 = x[0] * torch.sqrt(torch.square(x[0]) + 16) + x[1] * torch.sqrt(1 + torch.square(x[2])) * 10
        # f1 = torch.where(f1 < 1, f1, 1)
        # Compute f2
        f2 = (20 * torch.sqrt(16 + torch.square(x[2]))) / (x[0] * x[2]) * 1e-5
        # f2 = torch.where(f2 < 1, f2, 1)
        # Compute f3
        # f3 = (80 * torch.sqrt(1 + torch.square(x[2]))) / (x[1] * x[2]) * 1e-5
        # f3 = torch.where(f3 < 1, f3, 1)

        return -((f1 + f2).item() - 1 - 1e3)/1e3


class ReControl:
    def __init__(self, n_dim=3):
        self.m_2 = 0
        self.m_2_min = 8000
        self.m_2_max = 12000
        self.m_1 = 42000 * torch.ones(1)
        self.v = 0.7 * torch.ones(1)
        self.n_dim = n_dim
        self.g = 9.8
        self.l = None
        self.l_min = 5
        self.l_max = 8
        self.W_coef_min = 0.005
        self.W_coef_max = 0.015
        self.W_coef = None
        self.W = 0.01 * self.g * (self.m_1 + self.m_2)
        self.w = 1e6 * torch.ones(1)
        self.delta = 0.01 * torch.ones(1)
        self.F_min = 0 * torch.ones(1)
        self.F_max = 24107 * torch.ones(1)
        self.omega = None
        self.omega_0 = None

    def compute_TE(self, t):

        TE_2 = (self.m_1 * self.v * torch.pow(self.omega, 3) - self.omega * torch.square(self.omega_0) *
         (self.F_min * t[1] + self.F_max * (t[0] + t[2]) - torch.sum(t[:self.n_dim]) * self.W) +
         torch.square(self.omega_0) * ((self.F_max - self.F_min)*(torch.sin(t[2]*self.omega) - torch.sin((t[1] + t[2])*self.omega)) +
         (self.F_max - self.W) * torch.sin(torch.sum(t[:self.n_dim]) * self.omega)))

        TE = (self.m_2) / (2 * torch.square(self.m_1)*torch.pow(self.omega, 6)) * \
             (torch.square(self.omega) * torch.pow(self.omega_0, 4) *
              (self.F_max - self.W - (self.F_max - self.F_min) *
               (torch.cos(self.omega * t[2]) - torch.cos((t[1] + t[2])*self.omega))) +
              (self.W - self.F_max) * torch.cos(torch.square(torch.sum(t[:self.n_dim]) * self.omega)) +
              torch.pow(self.omega, 4) * ((self.F_max - self.F_min)*(torch.sin(t[2]*self.omega) - torch.sin((t[1] + t[2])*self.omega))) +
              (self.F_max - self.W) * torch.sin(torch.square(torch.sum(t[:self.n_dim]) * self.omega)) + torch.square(TE_2)
              )

        if TE >= self.delta:
            return self.w * TE
        else:
            return 0

    def compute_TE_new(self, t):

        TE_2 = (self.m_1 * self.v * torch.pow(self.omega, 3) - self.omega * torch.square(self.omega_0) *
         (self.F_min * t[1] + self.F_max * (t[0] + t[2]) - torch.sum(t[:self.n_dim]) * self.W) +
         torch.square(self.omega_0) * ((self.F_max - self.F_min)*(torch.sin(t[2]*self.omega) - torch.sin((t[1] + t[2])*self.omega)) +
         (self.F_max - self.W) * torch.sin(torch.sum(t[:self.n_dim]) * self.omega)))

        TE = (self.m_2) / (2 * torch.square(self.m_1)*torch.pow(self.omega, 6)) * \
             (torch.square(self.omega) * torch.pow(self.omega_0, 4) *
              (self.F_max - self.W - (self.F_max - self.F_min) *
               (torch.cos(self.omega * t[2]) - torch.cos((t[1] + t[2])*self.omega))) +
              (self.W - self.F_max) * torch.square(torch.cos(torch.sum(t[:self.n_dim]) * self.omega)) +
              torch.pow(self.omega, 4) * ((self.F_max - self.F_min)*(torch.sin(t[2]*self.omega) - torch.sin((t[1] + t[2])*self.omega))) +
              (self.F_max - self.W) * torch.square(torch.sin(torch.sum(t[:self.n_dim]) * self.omega)) + torch.square(TE_2)
              )

        if TE >= self.delta:
            return self.w * TE
        else:
            return 0

    def evaluate(self, x):
        t = x.clone()
        t[:self.n_dim] = t[:self.n_dim] * 3

        # Re-define some of the task parameters
        # Re-define m_2
        self.m_2 = self.m_2_min + (self.m_2_max - self.m_2_min) * t[self.n_dim]
        # Re-define l
        self.l = self.l_min + (self.l_max - self.l_min) * t[self.n_dim + 1]
        # Re-define W_coef and W
        self.W_coef = self.W_coef_min + (self.W_coef_max - self.W_coef_min) * t[self.n_dim + 2]
        self.W = self.W_coef * self.g * (self.m_1 + self.m_2)

        # Compute Omega
        omega = torch.sqrt(((self.m_1 + self.m_2)/self.m_1) * self.g/self.l)
        self.omega = omega
        omega_0 = torch.sqrt(self.g/self.l)
        self.omega_0 = omega_0

        E = self.compute_TE(t)

        return 1e-6 * (((2*E)/(self.m_2*torch.square(self.v))) +
                       (torch.sum(t[:self.n_dim]*self.omega/(2*torch.pi))) - 1e5).item()


class ReControlEnv:
    def __init__(self, n_dim=3):
        self.m_2 = 0
        self.m_2_min = 8000 * torch.ones(1)
        self.m_2_max = 12000 * torch.ones(1)
        self.m_1 = 42000 * torch.ones(1)
        self.v = 0.7 * torch.ones(1)
        self.n_dim = n_dim
        self.g = 9.8 * torch.ones(1)
        self.l = None
        self.l_min = 5 * torch.ones(1)
        self.l_max = 8 * torch.ones(1)
        self.W_coef_min = 0.005 * torch.ones(1)
        self.W_coef_max = 0.015 * torch.ones(1)
        self.W_coef = None
        self.W = 0.01 * self.g * (self.m_1 + self.m_2)
        self.w = 1e6 * torch.ones(1)
        self.delta = 0.01 * torch.ones(1)
        self.F_min = 0 * torch.ones(1)
        self.F_max = 24107 * torch.ones(1)
        self.omega = None
        self.omega_0 = None

    def compute_TE(self, t):

        TE_2 = (self.m_1 * self.v * torch.pow(self.omega, 3) - self.omega * torch.square(self.omega_0) *
         (self.F_min * t[1] + self.F_max * (t[0] + t[2]) - torch.sum(t[:self.n_dim]) * self.W) +
         torch.square(self.omega_0) * ((self.F_max - self.F_min)*(torch.sin(t[2]*self.omega) - torch.sin((t[1] + t[2])*self.omega)) +
         (self.F_max - self.W) * torch.sin(torch.sum(t[:self.n_dim]) * self.omega)))

        TE = (self.m_2) / (2 * torch.square(self.m_1)*torch.pow(self.omega, 6)) * \
             (torch.square(self.omega) * torch.pow(self.omega_0, 4) *
              (self.F_max - self.W - (self.F_max - self.F_min) *
               (torch.cos(self.omega * t[2]) - torch.cos((t[1] + t[2])*self.omega))) +
              (self.W - self.F_max) * torch.cos(torch.square(torch.sum(t[:self.n_dim]) * self.omega)) +
              torch.pow(self.omega, 4) * ((self.F_max - self.F_min)*(torch.sin(t[2]*self.omega) - torch.sin((t[1] + t[2])*self.omega))) +
              (self.F_max - self.W) * torch.sin(torch.square(torch.sum(t[:self.n_dim]) * self.omega)) + torch.square(TE_2)
              )

        if TE >= self.delta:
            return self.w * TE
        else:
            return 0

    def compute_TE_new(self, t):

        TE_2 = (self.m_1 * self.v * torch.pow(self.omega, 3) - self.omega * torch.square(self.omega_0) *
         (self.F_min * t[1] + self.F_max * (t[0] + t[2]) - torch.sum(t[:self.n_dim]) * self.W) +
         torch.square(self.omega_0) * ((self.F_max - self.F_min)*(torch.sin(t[2]*self.omega) - torch.sin((t[1] + t[2])*self.omega)) +
         (self.F_max - self.W) * torch.sin(torch.sum(t[:self.n_dim]) * self.omega)))

        TE = (self.m_2) / (2 * torch.square(self.m_1)*torch.pow(self.omega, 6)) * \
             (torch.square(self.omega) * torch.pow(self.omega_0, 4) *
              (self.F_max - self.W - (self.F_max - self.F_min) *
               (torch.cos(self.omega * t[2]) - torch.cos((t[1] + t[2])*self.omega))) +
              (self.W - self.F_max) * torch.square(torch.cos(torch.sum(t[:self.n_dim]) * self.omega)) +
              torch.pow(self.omega, 4) * ((self.F_max - self.F_min)*(torch.sin(t[2]*self.omega) - torch.sin((t[1] + t[2])*self.omega))) +
              (self.F_max - self.W) * torch.square(torch.sin(torch.sum(t[:self.n_dim]) * self.omega)) + torch.square(TE_2)
              )

        if TE >= self.delta:
            return self.w * TE
        else:
            return 0

    def evaluate(self, x):
        t = x.clone()
        t[:self.n_dim] = t[:self.n_dim] * 2
        t[:self.n_dim] = t[:self.n_dim] + t[self.n_dim:]

        # Re-define some of the task parameters
        # Re-define m_2
        self.m_2 = self.m_2_min + (self.m_2_max - self.m_2_min) * 0.5
        # Re-define l
        self.l = self.l_min + (self.l_max - self.l_min) * 0.5
        # Re-define W_coef and W
        self.W_coef = self.W_coef_min + (self.W_coef_max - self.W_coef_min) * 0.5
        self.W = self.W_coef * self.g * (self.m_1 + self.m_2)

        # Compute Omega
        omega = torch.sqrt(((self.m_1 + self.m_2)/self.m_1) * self.g/self.l)
        self.omega = omega
        omega_0 = torch.sqrt(self.g/self.l)
        self.omega_0 = omega_0

        E = self.compute_TE(t)

        return 1e-6 * (((2*E)/(self.m_2*torch.square(self.v))) +
                       (torch.sum(t[:self.n_dim]*self.omega/(2*torch.pi))) - 1e5).item()


class RE21_1():
    def __init__(self, n_dim=4, F=10.0, sigma=10.0, L=200.0, E=2e5):
        tmp_val = F / sigma

        self.F = F
        self.E = E
        self.L = L

        self.current_name = "real_one"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]).float()
        self.ubound = torch.ones(n_dim).float() * 3 * tmp_val
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.factor = 0.90

    def evaluate(self, y):
        F = self.F
        E = self.E
        L = self.L

        x = y.clone()

        x[self.n_dim:] = x[self.n_dim:] * (self.ubound - self.lbound) * (1 - self.factor)
        x[:self.n_dim] = x[:self.n_dim] * (self.ubound - self.lbound) * self.factor + self.lbound
        x[:self.n_dim] = x[:self.n_dim] + x[self.n_dim:]

        f1 = L * ((2 * x[0]) + np.sqrt(2.0) * x[1] + torch.sqrt(x[2]) + x[3])
        f1_max = L * ((2 * self.ubound[0]) + np.sqrt(2.0) * self.ubound[1] +
                      torch.sqrt(self.ubound[2]) + self.ubound[3])

        f2 = ((F * L) / E) * (
                    (2.0 / x[0]) + (2.0 * np.sqrt(2.0) / x[1]) - (2.0 * np.sqrt(2.0) / x[2]) + (2.0 / x[3]))
        f2_max = ((F * L) / E) * (
                    (2.0 / self.lbound[0]) + (2.0 * np.sqrt(2.0) / self.lbound[1]) -
                    (2.0 * np.sqrt(2.0) / self.ubound[2]) + (2.0 / self.lbound[3]))

        f1 = f1 / f1_max
        # f2 = f2 / f2_max
        # f_tot = (1.5*f1 + 0.5*f2) / 2
        # objs = torch.stack([f1, f2]).T
        #
        # return objs
        return f1.item()


class RE21_2():
    def __init__(self, n_dim=4, F=10.0, sigma=10.0, L=200.0, E=2e5):
        tmp_val = F / sigma

        self.F = F
        self.E = E
        self.L = L

        self.current_name = "real_one"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]).float()
        self.ubound = torch.ones(n_dim).float() * 3 * tmp_val
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.factor = 0.90

    def evaluate(self, y):
        F = self.F
        E = self.E
        L = self.L

        x = y.clone()

        x[self.n_dim:] = x[self.n_dim:] * (self.ubound - self.lbound) * (1 - self.factor)
        x[:self.n_dim] = x[:self.n_dim] * (self.ubound - self.lbound) * self.factor + self.lbound
        x[:self.n_dim] = x[:self.n_dim] + x[self.n_dim:]

        # f1 = L * ((2 * x[0]) + np.sqrt(2.0) * x[1] + torch.sqrt(x[2]) + x[3])
        # f1_max = L * ((2 * self.ubound[0]) + np.sqrt(2.0) * self.ubound[1] +
        #               torch.sqrt(self.ubound[2]) + self.ubound[3])

        f2 = ((F * L) / E) * (
                    (2.0 / x[0]) + (2.0 * np.sqrt(2.0) / x[1]) - (2.0 * np.sqrt(2.0) / x[2]) + (2.0 / x[3]))
        f2_max = ((F * L) / E) * (
                    (2.0 / self.lbound[0]) + (2.0 * np.sqrt(2.0) / self.lbound[1]) -
                    (2.0 * np.sqrt(2.0) / self.ubound[2]) + (2.0 / self.lbound[3]))

        # f1 = f1 / f1_max
        f2 = f2 / f2_max

        # objs = torch.stack([f1, f2]).T
        #
        # return objs
        return f2.item()


class RE25_1():
    def __init__(self, n_dim=3, F_max=1000, l_max=14, sigma_pm=6):
        self.F_max = F_max
        self.l_max = l_max
        self.sigma_pm = sigma_pm

        self.current_name = "real_three"

        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([1, 0.6, 0.009]).float()
        self.ubound = torch.tensor([70, 30, 0.5]).float()
        self.nadir_point = [5852.05896876, 1288669.78054]
        self.factor = 0.90

    def evaluate(self, y):

        x = y.clone()

        x[self.n_dim:] = x[self.n_dim:] * (self.ubound - self.lbound) * (1 - self.factor)
        x[:self.n_dim] = x[:self.n_dim] * (self.ubound - self.lbound) * self.factor + self.lbound
        x[:self.n_dim] = x[:self.n_dim] + x[self.n_dim:]

        x1 = torch.round(x[0])
        x2 = x[1]
        x3 = x[2]

        # First original objective function
        # f1 = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)
        f1 = (torch.pi * torch.pi * x2 * x3 * x3 * (x1 + 2)) / \
             (torch.pi * torch.pi * self.ubound[1] * self.ubound[2] * self.ubound[2] * (self.ubound[0] + 2))
        f1 = f1.float()

        # Constraint variables
        F_max = self.F_max
        l_max = self.l_max
        sigma_pm = self.sigma_pm

        S = 1.89 * 1e5
        C_f = ((4 * (x2 / x3) - 1) / (4 * (x2 / x3) - 4)) + (0.615 * x3) / x2
        G = 11.5 * 1e6
        K = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
        F_p = 300
        sigma_p = F_p / K
        l_f = (F_max / K) + (1.05 * (x1 + 2) * x3)
        sigma_w = 1.25

        # Original constraint functions
        g1 = S - (8 * C_f * F_max * x2) / (torch.pi * x3 * x3 * x3)
        g2 = l_max - l_f
        g3 = x2 / x3 - 3
        g4 = sigma_pm - sigma_p
        g5 = - sigma_p - (F_max - F_p) / K - 1.05 * (x1 + 2) * x3 + l_f
        g6 = - sigma_w + (F_max - F_p) / K
        g = torch.stack([g1, g2, g3, g4, g5, g6])
        if x.device.type == 'cuda':
            z = torch.zeros(g.shape).cuda().to(torch.float64)
        else:
            z = torch.zeros(g.shape).to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f2 = torch.sum(g, axis=0).to(torch.float64)
        f2 = f2 / 1e10

        f_tot = f2
        #
        # objs = torch.stack([f1, f2]).T

        return f_tot.item()


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

    # print(get_linear_mat(4, 5))
    problem = ReControl
    solution_vector = []
    solution_size = 30000
    solutions = torch.rand(solution_size, 6)



