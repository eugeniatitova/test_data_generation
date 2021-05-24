import torch
import random


def int_r(num):
    num = int(num + (0.5 if num > 0 else -0.5))
    return num


class DGenerator:
    """Генератор нескольких дискретных столбцов"""

    def __init__(self, n, p):
        """
        Args:
            n (int): длина столбцов
            p (array): матрица процентного распределния между категориями
        """

        self.data = []
        for i in range(len(p)):
            col = []
            for j in range(len(p[i])):
                col += ([j] * int_r(p[i][j] * n))
            random.shuffle(col)
            self.data.append(col)

        self.data = torch.tensor(self.data).transpose(0, 1)
        self.data_len = n

    def __getitem__(self, index):
        return self.data[index]
