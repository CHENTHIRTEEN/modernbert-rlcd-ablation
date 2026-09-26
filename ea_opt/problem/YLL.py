import numpy as np
from numpy import pi, sin, sqrt, cos, exp, e, floor
import random
from pymoo.core.problem import Problem


class YLLF01(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-100, xu=100)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            for j in range(d):
                ys[i] += xi[j] * xi[j]
        out["F"] = ys


class YLLF02(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-10, xu=10)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            re1, re2 = 0, 1
            for j in range(d):
                re1 += abs(xi[j])
                re2 *= abs(xi[j])
            ys[i] = re1 + re2
        out["F"] = ys


class YLLF03(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-100, xu=100)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            for j in range(d):
                re1 = sum(xi[:j + 1])
                ys[i] += re1 * re1
        out["F"] = ys


class YLLF04(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-100, xu=100)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.max(np.abs(x), axis=1)


class YLLF05(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-30, xu=30)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            for j in range(d - 1):
                ys[i] += 100 * pow((xi[j + 1] - xi[j] * xi[j]), 2) + pow((xi[j] - 1), 2)
        out["F"] = ys


class YLLF06(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-100, xu=100)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(np.square(np.floor(x + 0.5)), axis=1)


class YLLF07(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-1.28, xu=1.28)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            result = 0
            for j in range(d):
                result += (j + 1) * pow(xi[j], 4)
            ys[i] = result + random.random()
        out["F"] = ys


class YLLF08(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-500, xu=500)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            re1 = sum(xi[j] * sin(sqrt(abs(xi[j]))) for j in range(d))
            ys[i] = -re1 + 418.9828872724338 * d
        out["F"] = ys


class YLLF09(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-5.12, xu=5.12)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            result = 0
            for j in range(d):
                result += (xi[j] * xi[j] - 10 * cos(2 * pi * xi[j]) + 10)
            ys[i] = result
        out["F"] = ys


class YLLF10(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-5.12, xu=5.12)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            re1, re2, result = 0, 0, 0
            for j in range(d):
                re1 += xi[j] * xi[j]
                re2 += cos(2 * pi * xi[j])

            result = -20 * exp(-0.2 * sqrt(re1 / d)) - exp(re2 / d) + 20 + e

            ys[i] = result
        out["F"] = ys


# class YLLF10(Problem):
#     def __init__(self, n_var=30):
#         super().__init__(n_var=n_var, n_obj=1, xl=-32, xu=32)
#
#     def _evaluate(self, x, out, *args, **kwargs):
#         n, d = x.shape
#         ys = np.zeros(shape=(n,))
#         for i, xi in enumerate(x):
#             re1, re2, result = 0, 0, 0
#             for j in range(d):
#                 re1 += xi[j] * xi[j]
#                 re2 += cos(2 * pi * xi[j])
#
#             result = -20 * exp(-0.2 * sqrt(re1 / d)) - exp(re2 / d) + 20 + e
#
#             ys[i] = result
#         out["F"] = ys


class YLLF11(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-600, xu=600)

    def _evaluate(self, X, out, *args, **kwargs):
        n, d = X.shape
        result = np.zeros(shape=(n,))
        for i, xi in enumerate(X):
            result1, result2 = 0, 1
            for j in range(d):
                result1 += xi[j] * xi[j]
                result2 *= np.cos(xi[j] / np.sqrt(j + 1))
            result[i] = (1 / 4000) * result1 - result2 + 1
        out["F"] = result


import numpy as np


class YLLF12(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-50, xu=50)

    def _evaluate(self, X, out, *args, **kwargs):
        n, d = X.shape
        result = np.zeros(shape=(n,))
        for i, xi in enumerate(X):
            re1, re2 = 0, 0
            yn = 1 + 1 / 4 * (xi[-1] + 1)
            y1 = 1 + 1 / 4 * (xi[0] + 1)
            for j in range(d - 1):
                yi = 1 + 1 / 4 * (xi[j] + 1)
                yii = 1 + 1 / 4 * (xi[j + 1] + 1)
                re1 += (yi - 1) * (yi - 1) * (1 + 10 * np.power(np.sin(np.pi * yii), 2))
            for j in range(d):
                re2 += self.u(xi[j], 10, 100, 4)
            result[i] = np.pi / d * (10 * np.power(np.sin(np.pi * y1), 2) + re1 + (yn - 1) * (yn - 1)) + re2
        out["F"] = result

    @staticmethod
    def u(x_i, a, k, m):
        if x_i > a:
            value = k * np.power((x_i - a), m)
        elif x_i < -a:
            value = k * np.power((-x_i - a), m)
        else:
            value = 0
        return value


class YLLF13(Problem):
    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=1, xl=-600, xu=600)

    def _evaluate(self, X, out, *args, **kwargs):
        n, d = X.shape
        result = np.zeros(shape=(n,))
        for i, xi in enumerate(X):
            re1, re2 = 0, 0
            for j in range(d - 1):
                re1 += (xi[j] - 1) * (xi[j] - 1) * (1 + np.power(np.sin(3 * np.pi * xi[j + 1]), 2))
            for j in range(d):
                re2 += self.u(xi[j], 5, 100, 4)
            result[i] = 0.1 * (10 * np.power(np.sin(3 * np.pi * xi[0]), 2) + re1 + (xi[-1] - 1) * (xi[-1] - 1) * (
                    1 + np.power(np.sin(2 * np.pi * xi[-1]), 2))) + re2
        out["F"] = result

    @staticmethod
    def u(x_i, a, k, m):
        if x_i > a:
            value = k * np.power((x_i - a), m)
        elif x_i < -a:
            value = k * np.power((-x_i - a), m)
        else:
            value = 0
        return value
