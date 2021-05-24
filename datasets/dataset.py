import csv
import torch

from torch.utils.data.dataset import Dataset

from datasets.cgenerator import CGenerator
from datasets.dgenerator import DGenerator


class Dataset(Dataset):
    """Датасет из заданных типов столбцов"""

    def __init__(self, generate,
                 data_path=None, is_train=None,
                 data_type=None, n=None, p=None, mu=None, sigma=None, corr=None):
        """
        Args:
            generate (bool): флаг, генерировать или читать из файла

            # конструктор для чтения из файла
            data_path (str): путь к датасету
            is_train (bool): флаг, train/validation

            # конструктор для генерации
            data_type (list): тип данных, например ['d', 'd', 'd', 'c', 'c', 'c']
            n (int): длина столбцов
            p (array): матрица процентного распределния между категориями
            mu (array): мат. ожидания столбцов
            sigma (array): среднеквадратичное отклонения столбцов
            corr (array): матрица корреляции
        """

        self.data = []

        if generate:
            discrete_data = DGenerator(n, p)
            continuous_data = CGenerator(n, mu, sigma, corr)
            data = []
            for i in range(0, n):
                discrete_index = 0
                continuous_index = 0
                row = []
                for j in data_type:
                    if j == 'c':
                        row.append(continuous_data[i][continuous_index])
                        continuous_index += 1
                    elif j == 'd':
                        row.append(discrete_data[i][discrete_index])
                        discrete_index += 1
                data.append(row)
            self.data = torch.tensor(data, dtype=torch.float32)

            # размер датасета
            self.data_len = n

        else:
            with open(data_path, "r", newline="") as file:
                reader = csv.reader(file)
                next(reader, None)
                data = []
                for row in reader:
                    data.append(list(map(float, row)))
            data = torch.tensor(data, dtype=torch.float32)

            train_size = round(data.size(0) * 0.66)

            if is_train:
                self.data = data[:train_size]
            else:
                self.data = data[train_size:]

            # размер датасета
            self.data_len = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data_len
