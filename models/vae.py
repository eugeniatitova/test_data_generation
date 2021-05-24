import csv
import pandas as pd
import pytorch_lightning as pl
import torch

from datasets.dataset import Dataset
from models.logger import ImageLogger
from models.splitter import Splitter
from models.discriminator import Discriminator

from torch import nn
from torch.nn import functional as F


class VAE(pl.LightningModule):
    """Класс вариационного автоэнкодера"""

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.image_logger = ImageLogger()

        self.encoder = nn.Sequential(nn.Linear(self.hparams.input_dim, self.hparams.hidden_dim),
                                     nn.ReLU())

        self.fc_mu = nn.Linear(self.hparams.hidden_dim, self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.hidden_dim, self.hparams.latent_dim)

        self.decoder = nn.Sequential(nn.Linear(self.hparams.latent_dim, self.hparams.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hparams.hidden_dim, self.hparams.input_dim),
                                     nn.Sigmoid())

    def encode(self, x: torch.tensor):
        """
        Кодирует вход через сеть энкодера и возвращает латентные коды.
        :param x: (Tensor) Входной тензор [batch_size x input_dim]
        :return: (Tensor) Лист латентных кодов [batch_size x latent_dim], [batch_size x latent_dim]
        """
        z = self.encoder(x)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)
        return [mu, log_var]

    def decode(self, z: torch.tensor) -> torch.tensor:
        """
        Переводит латентные коды обратно в исходное пространство.
        :param z: (Tensor) [batch_size x latent_dim]
        :return: (Tensor) [batch_size x input_dim]
        """
        return self.decoder(z)

    def reparameterize(self, mu: torch.tensor, log_var: torch.tensor) -> torch.tensor:
        """
        Трюк с репараметризацией.
        :param mu: (Tensor) Среднее латентного пространства [batch_size x latent_dim]
        :param log_var: (Tensor) Стандартное отклонение [batch_size x latent_dim]
        :return: (Tensor) [batch_size x latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def loss_function(self, dis_recon, con_recon, dis_x, con_x, mu, log_var):
        discrete_loss = F.mse_loss(dis_recon, dis_x)
        continuous_loss = nn.BCELoss()(con_recon, con_x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return [kld_loss, discrete_loss, continuous_loss]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def forward(self, x: torch.tensor):
        [mu, log_var] = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return [recon_x, z, mu, log_var]

    def sample(self, num_samples: int):
        """
        Генерирует объект исходного пространства из латентного
        :param num_samples: (Int) Число объектов
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.hparams.latent_dim)
        x = self.decode(z)

        x_splitter = Splitter(data_type=self.hparams.data_type,
                              card=self.hparams.card,
                              con_min=self.hparams.continuous_min,
                              con_max=self.hparams.continuous_max,
                              data=x,
                              is_normalized=True,
                              is_ohencoded=True)

        x_splitter.inverse_normalize()
        x_splitter.inverse_ohe()
        x = x_splitter.join()

        self.logger.experiment.add_image('дискретная генерация',
                                         self.image_logger.get_diagram(x_splitter.discrete_part),
                                         dataformats='HWC')

        for i in range(0, self.hparams.continuous_dim):
            self.logger.experiment.add_histogram('непрерывная генерация', x_splitter.continuous_part[:, i])

        self.logger.experiment.add_image('коллеляция генерации',
                                         self.image_logger.get_corr_matrix(x_splitter.continuous_part),
                                         dataformats='HWC')

        if not self.hparams.generate:
            with open('test/generated.csv', "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(x.detach().numpy())

            original_df = pd.read_csv(self.hparams.data_path)
            generated_df = pd.DataFrame(x.detach().numpy())
            generated_df.columns = original_df.columns

            discriminator = Discriminator(original_df, generated_df)

            print(discriminator.score())
            print('Взаимная информация:', discriminator.mi(), '<', discriminator.entropy())

        return x

    def training_step(self, batch, batch_idx):

        x_splitter = Splitter(data_type=self.hparams.data_type,
                              card=self.hparams.card,
                              con_min=self.hparams.continuous_min,
                              con_max=self.hparams.continuous_max,
                              data=batch,
                              is_normalized=True,
                              is_ohencoded=True)

        [recon_x, z, mu, log_var] = self.forward(batch)

        recon_splitter = Splitter(data_type=self.hparams.data_type,
                                  card=self.hparams.card,
                                  con_min=self.hparams.continuous_min,
                                  con_max=self.hparams.continuous_max,
                                  data=recon_x,
                                  is_normalized=True,
                                  is_ohencoded=True)

        [kld_loss, discrete_loss, continuous_loss] = self.loss_function(dis_recon=recon_splitter.discrete_part,
                                                                        con_recon=recon_splitter.continuous_part,
                                                                        dis_x=x_splitter.discrete_part,
                                                                        con_x=x_splitter.continuous_part,
                                                                        mu=mu,
                                                                        log_var=log_var)

        self.log('training kld loss', kld_loss, logger=True)
        self.log('training discrete loss', discrete_loss, logger=True)
        self.log('training continuous loss', continuous_loss, logger=True)
        return (discrete_loss + continuous_loss) + (kld_loss / self.hparams.kld_divider)

    def validation_step(self, batch, batch_idx):

        x_splitter = Splitter(data_type=self.hparams.data_type,
                              card=self.hparams.card,
                              con_min=self.hparams.continuous_min,
                              con_max=self.hparams.continuous_max,
                              data=batch,
                              is_normalized=True,
                              is_ohencoded=True)

        [recon_x, z, mu, log_var] = self.forward(batch)

        recon_splitter = Splitter(data_type=self.hparams.data_type,
                                  card=self.hparams.card,
                                  con_min=self.hparams.continuous_min,
                                  con_max=self.hparams.continuous_max,
                                  data=recon_x,
                                  is_normalized=True,
                                  is_ohencoded=True)

        [kld_loss, discrete_loss, continuous_loss] = self.loss_function(dis_recon=recon_splitter.discrete_part,
                                                                        con_recon=recon_splitter.continuous_part,
                                                                        dis_x=x_splitter.discrete_part,
                                                                        con_x=x_splitter.continuous_part,
                                                                        mu=mu,
                                                                        log_var=log_var)

        validation_loss = (discrete_loss + continuous_loss) + (kld_loss / self.hparams.kld_divider)

        self.log('validation kld loss', kld_loss, logger=True)
        self.log('validation discrete loss', discrete_loss, logger=True)
        self.log('validation continuous loss', continuous_loss, logger=True)
        self.log('validation loss', validation_loss, logger=True)

        if self.current_epoch in self.hparams.epochs_to_log:
            self.logger.experiment.add_image('скрытое пространство, эпоха ' + str(self.current_epoch),
                                             self.image_logger.get_latent_space(z),
                                             dataformats='HWC')

            recon_splitter.inverse_normalize()
            recon_splitter.inverse_ohe()

            self.logger.experiment.add_image('дискретная реконструкция, эпоха ' + str(self.current_epoch),
                                             self.image_logger.get_diagram(recon_splitter.discrete_part),
                                             dataformats='HWC')

            for i in range(0, self.hparams.continuous_dim):
                x_i = recon_splitter.continuous_part[:, i]
                self.logger.experiment.add_histogram('непрерывная реконструкция, эпоха ' + str(self.current_epoch), x_i)

            self.logger.experiment.add_image('qq график, эпоха ' + str(self.current_epoch),
                                             self.image_logger.get_qq_plot(recon_splitter.continuous_part),
                                             dataformats='HWC')

            self.logger.experiment.add_image('корелляционная матрица, эпоха ' + str(self.current_epoch),
                                             self.image_logger.get_corr_matrix(recon_splitter.continuous_part),
                                             dataformats='HWC')
        return validation_loss

    def train_dataloader(self):
        if self.hparams.generate:
            train_data = Dataset(generate=self.hparams.generate,
                                 data_type=self.hparams.data_type,
                                 n=self.hparams.train_samples,
                                 p=self.hparams.p,
                                 mu=self.hparams.mu,
                                 sigma=self.hparams.sigma,
                                 corr=self.hparams.corr)
        else:
            train_data = Dataset(generate=self.hparams.generate,
                                 data_path=self.hparams.data_path,
                                 is_train=True)

        train_splitter = Splitter(data_type=self.hparams.data_type,
                                  card=self.hparams.card,
                                  con_min=self.hparams.continuous_min,
                                  con_max=self.hparams.continuous_max,
                                  data=train_data.data,
                                  is_normalized=False,
                                  is_ohencoded=False)

        train_splitter.normalize()
        train_splitter.ohe()
        train_splitter.join()
        train_data.data = train_splitter.data

        return torch.utils.data.DataLoader(train_data,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=True, num_workers=0)

    def val_dataloader(self):
        if self.hparams.generate:
            validation_data = Dataset(generate=self.hparams.generate,
                                      data_type=self.hparams.data_type,
                                      n=self.hparams.validation_samples,
                                      p=self.hparams.p,
                                      mu=self.hparams.mu,
                                      sigma=self.hparams.sigma,
                                      corr=self.hparams.corr)
        else:
            validation_data = Dataset(generate=self.hparams.generate,
                                      data_path=self.hparams.data_path,
                                      is_train=False)

        validation_splitter = Splitter(data_type=self.hparams.data_type,
                                       card=self.hparams.card,
                                       con_min=self.hparams.continuous_min,
                                       con_max=self.hparams.continuous_max,
                                       data=validation_data.data,
                                       is_normalized=False,
                                       is_ohencoded=False)

        validation_splitter.normalize()
        validation_splitter.ohe()
        validation_splitter.join()
        validation_data.data = validation_splitter.data

        return torch.utils.data.DataLoader(validation_data,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=0)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
