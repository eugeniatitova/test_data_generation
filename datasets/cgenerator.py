import torch
import numpy as np


def corr2cov(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Ковариационная матрица от корреляции и стандартных отклонений"""

    d = np.diag(s)
    return d @ p @ d


class CGenerator:
    """Генератор нескольких непрерывных столбцов"""

    def __init__(self, n, mu, sigma, corr):
        """
        Args:
            n (int): длина столбцов
            mu (array): мат. ожидания столбцов
            sigma (array): среднеквадратичное отклонения столбцов
            corr (array): матрица корреляции
        """
        # вычисляем матрицу ковариации
        cov = corr2cov(corr, sigma)

        # генерируем многомерное распределение с заданными параметрами
        self.data = torch.tensor(np.random.multivariate_normal(mean=mu, cov=cov, size=n))
        self.data_len = n

    def __getitem__(self, index):
        return self.data[index]
