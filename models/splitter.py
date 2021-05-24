import numpy
import torch


class Splitter:
    """Класс для работы с непрерывными и дискретными данными по отдельности"""

    def __init__(self, data_type: list, card: list, con_min: list, con_max: list,
                 data: torch.tensor, is_normalized: bool, is_ohencoded: bool):
        """
        Args:
            data_type (list): тип данных, например ['d', 'd', 'd', 'c', 'c', 'c']
            card (list): кардинальность дискретных столбцов
            con_min (list): минимумы непрерывных столбцов
            con_max (list): максимумы непрерывных столбцов
            data (torch.tensor): тензор для разделения
            is_normalized (bool): флаг того, что данные уже нормализованы
            is_ohencoded (bool): флаг того, что данные преобразованы
        """

        self.data_type = data_type
        self.card = card

        self.ohe_data_type = self.data_type.copy()
        for i, j in zip(numpy.nonzero([c == 'd' for c in self.data_type])[0], self.card):
            self.ohe_data_type[i] = ['d'] * j
        self.ohe_data_type = [item for sublist in self.ohe_data_type for item in sublist]

        self.continuous_min = con_min
        self.continuous_max = con_max
        self.data = data

        self.normalized = is_normalized
        self.ohencoded = is_ohencoded

        [self.discrete_part, self.continuous_part] = self.split()

    def split(self):
        if self.ohencoded:
            disc_indices = numpy.nonzero([c == 'd' for c in self.ohe_data_type])[0]
            cont_indices = numpy.nonzero([c == 'c' for c in self.ohe_data_type])[0]
        else:
            disc_indices = numpy.nonzero([c == 'd' for c in self.data_type])[0]
            cont_indices = numpy.nonzero([c == 'c' for c in self.data_type])[0]

        discrete_part = torch.index_select(self.data, 1, torch.tensor(disc_indices).to(self.data.device))
        continuous_part = torch.index_select(self.data, 1, torch.tensor(cont_indices).to(self.data.device))

        return [discrete_part, continuous_part]

    def join(self):
        res = []
        cont_index = 0
        disc_index = 0

        if self.ohencoded:
            for i in self.ohe_data_type:
                if i == 'c':
                    res.append(self.continuous_part[:, cont_index])
                    cont_index += 1
                elif i == 'd':
                    res.append(self.discrete_part[:, disc_index])
                    disc_index += 1
        else:
            for i in self.data_type:
                if i == 'c':
                    res.append(self.continuous_part[:, cont_index])
                    cont_index += 1
                elif i == 'd':
                    res.append(self.discrete_part[:, disc_index])
                    disc_index += 1

        self.data = torch.stack(res).transpose(0, 1)
        return self.data

    def normalize(self):
        for i in range(self.continuous_part.size(1)):
            num = self.continuous_part[:, i] - self.continuous_min[i]
            den = self.continuous_max[i] - self.continuous_min[i]
            self.continuous_part[:, i] = num / den
        self.normalized = True

    def inverse_normalize(self):
        for i in range(self.continuous_part.size(1)):
            mul = self.continuous_max[i] - self.continuous_min[i]
            self.continuous_part[:, i] = self.continuous_part[:, i] * mul + self.continuous_min[i]
        self.normalized = False

    def ohe(self):
        x = self.discrete_part.transpose(0, 1).type(torch.int64)
        x_list = []
        for i, j in zip(x, self.card):
            x_list.append(torch.nn.functional.one_hot(i, j))
        self.discrete_part = torch.cat(x_list, dim=1)
        self.ohencoded = True

    def inverse_ohe(self):
        x = torch.split(self.discrete_part, split_size_or_sections=self.card, dim=1)
        x_list = []
        for i in x:
            x_list.append(i.argmax(-1))
        self.discrete_part = torch.stack(x_list).transpose(0, 1)
        self.ohencoded = False



