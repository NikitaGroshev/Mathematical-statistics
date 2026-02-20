"""
Случайная величина имеет экспоненциальный закон распределения

p(x) =  exp(-x) , x >= 0
        0       , x < 0

Сгенерируйте выборку объема n = 25.

a) Определить по выборке моду, медиану, размах, оценку коэффициента асимметрии.
b) Построить эмпирическую функцию распределения, гистограмму и boxplot.
c) Сравнить оценку плотности распределения среднего арифметического
   элементов выборки, полученную c помощью ЦПТ, c бутстраповской оценкой
   этой плотности.
d) Найти бутстраповскую оценку плотности распределения коэффициента асимметрии
   и оценить вероятность того, что коэффициент асимметрии будет меньше 1.
e) Сравнить плотность распределения медианы выборки c бутстраповской оценкой 
   этой плотности.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Callable

from math import comb


def coeff_gamma(selection: np.ndarray) -> float:
    """
    Функция для вычисления оценки коэффициента асимметрии.
    """
    mean = np.mean(selection)
    mu_2 = np.var(selection)
    mu_3 = np.sum((selection - mean)**3) / selection.size
    gamma = mu_3 / (np.sqrt(mu_2**3))

    return gamma

class SolutionT2:
    def __init__(self, n: int) -> None:
        self._n: int = n                            # объём выборки
        self._selection: np.ndarray = np.zeros(n)   # массив для выборки

    @property
    def n(self) -> int:
        return self._n
    
    @property
    def selection(self) -> np.ndarray:
        return self._selection

    def generate_selection(self) -> None:
        """
        Генерация выборки.

        Так как распределение непрерывное и строго монотонно возрастающее, то вместо 
        inf(t: F(t) >= p) достаточно решить x = F^-1(y), y ~ R(0, 1)

        F(x) = 1 - exp(-x)
        x = - ln(1 - y)
        """
        y_uniform = np.random.random_sample(size=self.n)
        self._selection = - np.log(1 - y_uniform)
        
    def print_selection(self) -> None:
        """
        Вывод выборки.
        """
        series = pd.Series(
            data=self.selection,
            index=np.array(['[' + str(i) + ']' for i in range(1, self.n + 1)])
        )
        print(series)

    def print_num_characteristics(self) -> None:
        """
        Вычисление числовых характеристик.
        """
        unique_elements, counts = np.unique(self.selection, return_counts=True)
        max_count = np.max(counts)
        modes = unique_elements[counts == max_count]    # Все моды выборки

        med = np.median(self.selection)                 # Медиана выборки

        l = np.ptp(self.selection)                      # Размах выборки

        mean = np.mean(self.selection)
        mu_2 = np.var(self.selection)
        mu_3 = np.sum((self.selection - mean)**3) / self.n
        gamma = mu_3 / (np.sqrt(mu_2**3))                  # Коэффициент асимметрии

        print(f"Моды:                    {modes}")
        print(f"Медиана:                 {med}")
        print(f"Размах:                  {l}")
        print(f"Оценка коэф. асимметрии: {gamma}")

    def visualize_empirical_distribution_function(self) -> None:
        """
        Построение эмпирической функции распределения.
        """
        x = np.sort(np.append(self.selection, np.array([0, np.max(self.selection) + 0.5])))
        y = np.array([i / self.n for i in range(0, self.n + 1)] + [1])

        figure, axis = plt.subplots(figsize=(10, 6))
        axis: plt.Axes

        for i in range(self.n + 1):
            if i == 0:
                axis.hlines(y[i], xmin=x[i], xmax=x[i+1], color='green', linewidth=2, label='Эмпирическая функция')
            else:
                axis.hlines(y[i], xmin=x[i], xmax=x[i+1], color='green', linewidth=2)

        x_teor = np.linspace(x[0], x[self.n + 1], 100)
        y_teor = 1 - np.exp(-x_teor)                    # Теоретическая плотность

        axis.plot(x_teor, y_teor, c="red", label='Теоретическая плотность')

        axis.set_title('Эмпирическая функция распределения')
        axis.set_xlabel('X')
        axis.set_ylabel('F(X)')

        plt.legend(loc='lower right')
        plt.show()

    def visualize_histogram(self) -> None:
        """
        Построение гистограммы.
        """
        k = int(1 + np.log2(self.n))    # Количество интервалов

        figure, axis = plt.subplots(figsize=(10, 6))
        axis: plt.Axes

        axis.hist(
            self.selection,
            bins=k,
            color="green",
            edgecolor="darkgreen",
            density=True,
            label='Выборка'
        )

        x_teor = np.linspace(0, np.max(self.selection))
        y_teor = np.exp(-x_teor)

        axis.plot(
            x_teor,
            y_teor,
            c='red',
            label='Теоретическая плотность'
        )

        axis.set_title("Гистограмма")

        plt.legend()
        plt.show()
        
    def visualize_boxplot(self) -> None:
        """
        Построение boxplot.
        """
        figure, axis = plt.subplots(figsize=(10, 6))
        axis: plt.Axes

        axis.boxplot(
            self.selection,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor='lightgreen'),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black", linestyle="--"),
            capprops=dict(color="black"),
            flierprops=dict(marker="*", color="black") 
        )

        axis.set_yticks([])
        axis.set_title('Boxplot')

        plt.show()

    def bootstrap(self, g: Callable[[np.ndarray], float]) -> np.ndarray:
        """
        Реализация bootstrap для произвольной статистики g.
        """
        bootstrap_array = []

        for _ in range(1000):
            subselection = np.random.choice(self.selection, size=self.n, replace=True)
            x = g(subselection)
            bootstrap_array.append(x)

        return np.array(bootstrap_array)

    def visualize_bootstrap_mean(self) -> None:
        """
        Гистограмма bootstrap для среднего арифметического.

        Для оценки плотности по ЦПТ использую параметры эмпирического 
        распределения. (Так как bootstrap зависит от выборки)
        """
        data = self.bootstrap(np.mean)

        k = int(1 + np.log2(data.size))

        figure, axis = plt.subplots(figsize=(10, 6))
        axis: plt.Axes

        axis.hist(
            data,
            bins=k,
            color="green",
            edgecolor="darkgreen",
            density=True,
            label="bootstrap"
        )

        a = np.mean(self.selection)                         # Оценка среднего выборки
        sigma = np.sqrt(np.var(self.selection) / self.n)    # Оценка sigma = sqrt((оценка дисперсии) / n)

        x_clt = np.linspace(0, np.max(data), 100)
        y_clt = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp( -((x_clt - a)**2) / (2 * sigma**2) ) # clt - central limit threorem

        axis.plot(
            x_clt,
            y_clt,
            color='red',
            label='ЦПТ'
        )

        axis.set_title("Бутстраповская оценка плотности среднего арифметического")

        plt.legend()
        plt.show()

    def probability_estimate(self) -> float:
        """
        Функция для вычисления вероятности того,
        что коэффициент асимметрии будет меньше 1.
        """
        data = self.bootstrap(coeff_gamma)
        m = data[data < 1].size
        n = data.size

        return m / n

    def visualize_bootstrap_gamma(self) -> None:
        """
        Гистограмма bootstrap для коэффициента асимметрии.
        """

        data = self.bootstrap(coeff_gamma)

        k = int(1 + np.log2(data.size))

        figure, axis = plt.subplots(figsize=(10, 6))
        axis: plt.Axes

        axis.hist(
            data,
            bins=k,
            color="green",
            edgecolor="darkgreen",
            density=True
        )

        axis.set_title("Бутстраповская оценка плотности коэффициента асимметрии")

        plt.show()

    def visualize_bootstrap_median(self) -> None:
        """
        Гистограмма bootstrap для медианы.
        """
        data = self.bootstrap(np.median)

        k = int(1 + np.log2(data.size))

        figure, axis = plt.subplots(figsize=(10, 6))
        axis: plt.Axes

        axis.hist(
            data,
            bins=k,
            color="green",
            edgecolor="darkgreen",
            density=True,
            label="bootstrap"
        )

        x_teor = np.linspace(0, np.max(data), 100)
        y_teor = 25 * comb(24, 12) * ((1 - np.exp(-x_teor))**12) * np.exp(-13 * x_teor) # Теоретическая плотность
                                                                                        # медианы
        axis.plot(
            x_teor,
            y_teor,
            color='red',
            label='Плотность медианы'
        )

        axis.set_title("Бутстраповская оценка плотности медианы")

        plt.legend()
        plt.show()

        
if __name__ == "__main__":
    solution = SolutionT2(25)
    print('==   Выборка   ==')
    solution.generate_selection()
    solution.print_selection()
    solution.print_num_characteristics()
    solution.visualize_empirical_distribution_function()
    solution.visualize_histogram()
    solution.visualize_boxplot()
    solution.visualize_bootstrap_mean()
    print(f"Вероятность того, что коэффициент асимметрии меньше 1 = {solution.probability_estimate()}")
    solution.visualize_bootstrap_gamma()
    solution.visualize_bootstrap_median()