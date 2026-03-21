"""

Случайная величина распределена равномерно на отрезке [θ,2θ].

a) По выборке объема n найти оценки параметра θ методом моментов и методом максимального правдоподобия.
b) Проверить оценки на несмещенность и состоятельность. Исправить эти оценки, если необходимо.
c) Сравнить эффективность исправленных оценок.
d) Построить точный доверительный интервал для параметра θ.
e) Построить асимптотический доверительный интервал для параметра θ.
f) Сгенерируйте выборку объема n=100 для некоторого значения параметра θ. 
   Вычислите указанные выше доверительные интервалы для доверительной вероятности 0.95.
g) Численно постройте бутстраповский доверительный интервал.
h) Сравнить все интервалы.

"""

import numpy as np
from scipy import stats


class SolutionT5:
    def __init__(self, n: int, theta: float) -> None:
        self._n: int = n                            # Объем выборки
        self._selection: np.ndarray = np.zeros(n)   # Массив для выборки
        self._theta: float = theta                  # Параметр θ

        self.generate_selection()

    def generate_selection(self) -> None:
        """
        Генерация выборки.

        Так как распределение непрерывное и строго монотонно возрастающее, то вместо 
        inf(t: F(t) >= p) достаточно решить x = F^-1(y), y ~ R(0, 1)

        F(X) = (x - θ) / θ = y
        x = θy + θ
        """
        y_uniform = np.random.random_sample(size=self._n)
        self._selection = self._theta * y_uniform + self._theta

    def calculate_exact_confidence_interval(self) -> tuple[float, float]:
        """
        Вычисление точного доверительного интервала.
        """
        x_max = np.max(self._selection)
        beta = 0.95                                 # Доверительная вероятность
        t1 = ((1 - beta) / 2)**(1/self._n)
        t2 = ((1 + beta) / 2)**(1/self._n)

        t_min = x_max / (1 + t2)
        t_max = x_max / (1 + t1)

        return t_min, t_max
    
    def calculate_asymptotic_confidence_interval(self) -> tuple[float, float]:
        """
        Вычисление асимптотического доверительного интервала.
        """
        x_mean = np.mean(self._selection)           # Выборочное среднее
        beta = 0.95                                 # Доверительная вероятность
        
        p1 = (1 - beta) / 2
        p2 = (1 + beta) / 2

        q1 = stats.norm.ppf(p1, loc=0, scale=1)     # Квантиль порядка t1
        q2 = stats.norm.ppf(p2, loc=0, scale=1)     # Квантиль порядка t2

        x_2_mean = np.mean(self._selection**2)      # Среднее квадрата

        a = np.sqrt(x_2_mean - (x_mean**2))

        t_min = (2/3) * (x_mean - ((q2 * a) / np.sqrt(self._n)))
        t_max = (2/3) * (x_mean - ((q1 * a) / np.sqrt(self._n)))

        return t_min, t_max
    
    def non_param_bootstrap_interval(self) -> tuple[float, float]:
        """
        Построение доверительного интервала c помощью непараметрического бутстрапа.

        B качестве оценки θ возьмем x_max / 2.
        """
        theta_estimate = np.max(self._selection) / 2    # Оценка θ
        beta = 0.95                                     # Доверительная вероятность

        bootstrap_array = []

        for _ in range(1000):
            subselection = np.random.choice(self._selection, size=self._n, replace=True)
            theta_subselection = np.max(subselection) / 2
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

    def param_bootstrap_interval(self) -> tuple[float, float]:
        """
        Построение доверительного интервала c помощью параметрического бутстрапа.

        B качестве оценки θ возьмем x_max / 2.
        """
        theta_estimate = np.max(self._selection) / 2    # Оценка θ
        beta = 0.95                                     # Доверительная вероятность

        bootstrap_array = []

        for _ in range(50000):
            x_uniform = np.random.random_sample(size=self._n)
            new_selection = theta_estimate * x_uniform + theta_estimate
            new_theta = np.max(new_selection) / 2
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
        t_exact_min, t_exact_max = self.calculate_exact_confidence_interval()
        t_asymp_min, t_asymp_max = self.calculate_asymptotic_confidence_interval()
        t_non_param_boot_min, t_non_param_boot_max = self.non_param_bootstrap_interval()
        t_param_boot_min, t_param_boot_max = self.param_bootstrap_interval()

        l_exact = t_exact_max - t_exact_min         # Длины интервалов
        l_asymp = t_asymp_max - t_asymp_min
        l_non_param_bootstrap = t_non_param_boot_max - t_non_param_boot_min
        l_param_bootstrap = t_param_boot_max - t_param_boot_min

        print("Сравнение интервалов: ")
        print(f"θ = {self._theta}")
        print(f"Точный доверительный интервал          : {t_exact_min:.6f} < θ < {t_exact_max:.6f}, длина = {l_exact:.6f}")
        print(f"Асимптотический доверительный интервал : {t_asymp_min:.6f} < θ < {t_asymp_max:.6f}, длина = {l_asymp:.6f}")
        print(f"Непараметрический бутстрап             : {t_non_param_boot_min:.6f} < θ < {t_non_param_boot_max:.6f}, длина = {l_non_param_bootstrap:.6f}")
        print(f"Параметрический бутстрап               : {t_param_boot_min:.6f} < θ < {t_param_boot_max:.6f}, длина = {l_param_bootstrap:.6f}")


if __name__ == "__main__":
    solution = SolutionT5(n=100, theta=10)
    solution.print_intervals()