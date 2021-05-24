import io
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch

from sklearn.manifold import TSNE


class ImageLogger:
    """Отрисовка картинок для логоривания"""

    def __init__(self):
        plt.style.use('seaborn')

    def get_diagram(self, x: torch.tensor):
        x = torch.round(x.type(torch.float32))
        figure, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.yaxis.grid(True, zorder=1)
        columns = [f"{i + 1}" for i in range(x.size(1))]
        max_value = int(np.max(x.clone().detach().numpy())) + 1
        x_tricks = np.arange(len(columns))
        width = 1 / max_value - 0.1
        for i in range(max_value):
            data = []
            for j in range(x.size(1)):
                data.append(np.count_nonzero(x[:, j] == i) / len(x))
            ax.bar(x_tricks + i * width, data, width, label=str(i))

        ax.set_xticks(x_tricks)
        ax.set_xticklabels(columns)
        ax.legend()
        return self.get_image(figure)

    def get_qq_plot(self, x):
        cols = 2
        rows = math.ceil(x.size(1) / cols)
        figure = plt.figure(figsize=(6.4 * cols, 4.8 * rows))
        for i in range(x.size(1)):
            figure.add_subplot(rows, cols, i + 1)
            stats.probplot(x[:, i], plot=plt, fit=True)
        return self.get_image(figure)

    def get_corr_matrix(self, x):
        figure = plt.figure(figsize=(8, 8))
        corr = np.corrcoef(x.clone().detach().numpy(), rowvar=False)
        sns.heatmap(corr, square=True, linewidths=.5, annot=True)
        plt.xticks(rotation='0')
        plt.yticks(rotation='0')
        plt.subplots_adjust(top=0.95, bottom=0.25, left=0.25, right=1)
        plt.tight_layout()
        return self.get_image(figure)

    def get_latent_space(self, data):
        tsne = TSNE(n_components=2, random_state=0).fit_transform(data)
        x = tsne[:, 0]
        y = tsne[:, 1]
        figure = plt.figure(figsize=(10, 10))
        sns.scatterplot(
            data=tsne,
            x=x,
            y=y,
            legend="full",
            alpha=0.3
        )
        return self.get_image(figure)

    def get_image(self, figure: plt.figure):
        buf = io.BytesIO()
        plt.savefig(buf, format='raw')
        buf.seek(0)
        image = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
                           newshape=(int(figure.bbox.bounds[3]),
                                     int(figure.bbox.bounds[2]), -1))
        buf.close()
        plt.close(figure)
        return image
