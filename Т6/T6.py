"""

Случайная величина имеет распределение Парето:

p(x) = (θ - 1) / x^θ , x >= 1
     =       0       , x <  1;   θ > 1.


a) По выборке объема n найти оценку параметра θ методом максимального правдоподобия.
b) Построить доверительный интервал для медианы.
c) Построить асимптотический доверительный интервал для параметра θ.
d) Сгенерируйте выборку объема n = 100 для некоторого значения параметра θ. 
   Вычислите указанные выше доверительные интервалы для доверительной вероятности 0.95.
e) Численно постройте бутстраповский доверительный интервал двумя способами, 
   используя параметрический бутстрап и непараметрический бутстрап.
f) Сравнить все интервалы.

"""

import numpy as np
from scipy import stats
from typing import Callable


class SolutionT6:
    def __init__(self, n: int, theta: float):
        self._n: int = n                    # Объем выборки
        self._selection: np.ndarray = None  # Массив для выборки
        self._theta: float = theta          # Параметр θ

        self.generate_selection()

    def generate_selection(self) -> None:
        """
        Генерация выборки.

        Так как распределение непрерывное и строго монотонно возрастающее, то вместо 
        inf(t: F(t) >= p) достаточно решить x = F^-1(y), y ~ R(0, 1)

        F(x) = 1 - 1 / x^(θ - 1) = y
        1 / (1 - y) = x^(θ - 1)
        x = (1 / (1 - y))^(1 / (θ - 1))
        """
        y_uniform = np.random.random_sample(size=self._n)
        self._selection = (1 / (1 - y_uniform))**(1 / (self._theta - 1))

    def calculate_median_interval(self) -> tuple[float, float]:
        """
        Вычисление доверительного интервала для медианы.
        """
        s = np.sum(np.log(self._selection))
        beta = 0.95

        t1 = stats.chi2.ppf((1 - beta) / 2, 2 * self._n)
        t2 = stats.chi2.ppf((1 + beta) / 2, 2 * self._n)

        t_min = 2**((2 * s) / t2)
        t_max = 2**((2 * s) / t1)

        return t_min, t_max
    
    def calculate_asymp_theta_interval(self) -> tuple[float, float]:
        """
        Вычисление асимптотического доверительного интервала для параметра θ.
        """
        s = np.sum(np.log(self._selection))
        theta_estimate = 1 + (self._n / s)
        beta = 0.95

        t1 = stats.norm.ppf((1 - beta) / 2, loc=0, scale=1)
        t2 = stats.norm.ppf((1 + beta) / 2, loc=0, scale=1)     

        t_min = theta_estimate - ((t2 * (theta_estimate - 1)) / np.sqrt(self._n))
        t_max = theta_estimate - ((t1 * (theta_estimate - 1)) / np.sqrt(self._n))

        return t_min, t_max

    def non_param_bootstrap_theta(self) -> tuple[float, float]:
        """
        Непараметрический бутстрап для параметра θ.

        B качестве оценки θ берем 1 + n / sum(ln(x_i)).
        """
        s = np.sum(np.log(self._selection))
        theta_estimate = 1 + (self._n / s)
        beta = 0.95

        bootstrap_array = []

        for _ in range(1000):
            subselection = np.random.choice(self._selection, size=self._n, replace=True)
            s_subselection = np.sum(np.log(subselection))
            theta_subselection = 1 + (self._n / s_subselection)
            delta = theta_subselection - theta_estimate
            bootstrap_array.append(delta)

        bootstrap = np.array(bootstrap_array)
        sorted_bootstrap = np.sort(bootstrap)

        k1 = int(((1 - beta) / 2) * 1000)
        k2 = int(((1 + beta) / 2) * 1000)

        delta_k1 = sorted_bootstrap[k1]
        delta_k2 = sorted_bootstrap[k2]

        t_min = theta_estimate - delta_k2
        t_max = theta_estimate - delta_k1

        return t_min, t_max

    def non_param_bootstrap_median(self) -> tuple[float, float]:
        """
        Непараметрический бутстрап для медианы.

        B качестве оценки медианы берем корень степени θ - 1 из 2, где вместо θ берем её оценку.
        """
        s = np.sum(np.log(self._selection))
        theta_estimate = 1 + (self._n / s)

        median_estimate = 2**(1 / (theta_estimate - 1))

        beta = 0.95

        bootstrap_array = []

        for _ in range(1000):
            subselection = np.random.choice(self._selection, size=self._n, replace=True)
            s_subselection = np.sum(np.log(subselection))
            theta_subselection = 1 + (self._n / s_subselection)
            median_subselection = 2**(1 / (theta_subselection - 1))
            delta = median_subselection - median_estimate
            bootstrap_array.append(delta)

        bootstrap = np.array(bootstrap_array)
        sorted_bootstrap = np.sort(bootstrap)

        k1 = int(((1 - beta) / 2) * 1000)
        k2 = int(((1 + beta) / 2) * 1000)

        delta_k1 = sorted_bootstrap[k1]
        delta_k2 = sorted_bootstrap[k2]

        t_min = median_estimate - delta_k2
        t_max = median_estimate - delta_k1

        return t_min, t_max
    
    def param_bootstrap_median(self) -> tuple[float, float]:
        """
        Параметрический бутстрап для медианы.
        """
        s = np.sum(np.log(self._selection))
        theta_estimate = 1 + (self._n / s)

        median_estimate = 2**(1 / (theta_estimate - 1))

        beta = 0.95

        bootstrap_array = []

        for _ in range(50000):
            x_uniform = np.random.random_sample(size=self._n)
            new_selection = (1 / (1 - x_uniform))**(1 / (theta_estimate - 1))
            new_s = np.sum(np.log(new_selection))
            new_theta = 1 + (self._n / new_s)
            new_median = 2**(1 / (new_theta - 1))
            delta = new_median - median_estimate
            bootstrap_array.append(delta)

        bootstrap = np.array(bootstrap_array)
        sorted_bootstrap = np.sort(bootstrap)

        k1 = int(((1 - beta) / 2) * 50000)
        k2 = int(((1 + beta) / 2) * 50000)

        delta_k1 = sorted_bootstrap[k1]
        delta_k2 = sorted_bootstrap[k2]

        t_min = median_estimate - delta_k2
        t_max = median_estimate - delta_k1

        return t_min, t_max
    
    def param_bootstrap_theta(self) -> tuple[float, float]:
        """
        Параметрический бутстрап для параметра θ.
        """
        s = np.sum(np.log(self._selection))
        theta_estimate = 1 + (self._n / s)

        beta = 0.95

        bootstrap_array = []

        for _ in range(50000):
            x_uniform = np.random.random_sample(size=self._n)
            new_selection = (1 / (1 - x_uniform))**(1 / (theta_estimate - 1))
            new_s = np.sum(np.log(new_selection))
            new_theta = 1 + (self._n / new_s)
            delta = new_theta - theta_estimate
            bootstrap_array.append(delta)

        bootstrap = np.array(bootstrap_array)
        sorted_bootstrap = np.sort(bootstrap)

        k1 = int(((1 - beta) / 2) * 50000)
        k2 = int(((1 + beta) / 2) * 50000)

        delta_k1 = sorted_bootstrap[k1]
        delta_k2 = sorted_bootstrap[k2]

        t_min = theta_estimate - delta_k2
        t_max = theta_estimate - delta_k1

        return t_min, t_max
    
    def print_intervals(self) -> None:
        """
        Сравнение интервалов.
        """
        t_med_min, t_med_max = self.calculate_median_interval()
        t_theta_min, t_theta_max = self.calculate_asymp_theta_interval()

        n_boot_theta_min, n_boot_theta_max = self.non_param_bootstrap_theta()
        p_boot_theta_min, p_boot_theta_max = self.param_bootstrap_theta()

        n_boot_med_min, n_boot_med_max = self.non_param_bootstrap_median()
        p_boot_med_min, p_boot_med_max = self.param_bootstrap_median()

        l_theta = t_theta_max - t_theta_min         # Длины интервалов
        l_med = t_med_max - t_med_min

        l_n_boot_theta = n_boot_theta_max - n_boot_theta_min
        l_p_boot_theta = p_boot_theta_max - p_boot_theta_min

        l_n_boot_med = n_boot_med_max - n_boot_med_min
        l_p_boot_med = p_boot_med_max - p_boot_med_min

        print("Сравнение интервалов: ")
        print(f"θ = {self._theta}")
        print(f"Доверительный интервал для медианы     : {t_med_min:.6f} < median < {t_med_max:.6f}, длина = {l_med:.6f}")
        print(f"Непараметрический бутстрап для медианы : {n_boot_med_min:.6f} < median < {n_boot_med_max:.6f}, длина = {l_n_boot_med:.6f}")
        print(f"Параметрический бутстрап для медианы   : {p_boot_med_min:.6f} < median < {p_boot_med_max:.6f}, длина = {l_p_boot_med:.6f}")
        print(f"Асимптотический доверительный интервал для параметра θ : {t_theta_min:.6f} < θ < {t_theta_max:.6f}, длина = {l_theta:.6f}")
        print(f"Непараметрический бутстрап для параметра θ             : {n_boot_theta_min:.6f} < θ < {n_boot_theta_max:.6f}, длина = {l_n_boot_theta:.6f}")
        print(f"Параметрический бутстрап для параметра θ               : {p_boot_theta_min:.6f} < θ < {p_boot_theta_max:.6f}, длина = {l_p_boot_theta:.6f}")

if __name__ == "__main__":
    solution = SolutionT6(n=100, theta=2)
    solution.print_intervals()