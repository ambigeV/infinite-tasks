import numpy as np
from math import cos, sin
import math
import torch
import time


def get_problem(name, problem_params=None):
    name = name.lower()

    PROBLEM = {
        'sep_arm': ActualArm(n_dim=problem_params),
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
    problem_name = ["sep_arm"]
    problem_params = 10
    task_dim = 2
    problem = get_problem(problem_name[0], problem_params=problem_params)
    st = time.time()
    for i in range(10000):
        solution_params = torch.rand(problem_params + task_dim)
        print("solution in iter {} is {}".format(i + 1, problem.evaluate(solution_params)))
    en = time.time()
    print("lasting time is {}s".format(en - st))

