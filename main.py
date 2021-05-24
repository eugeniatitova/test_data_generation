import yaml

from argparse import ArgumentParser
from models.vae import VAE
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get_hyperparameters():
    parser = ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        #default='configs/gen.yaml'
                        default='configs/heart.yaml'
                        )

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            params = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return params


if __name__ == "__main__":

    tb_logger = pl_loggers.TensorBoardLogger(save_dir='tensorboard_logs/',
                                             name='test_data_generation')

    params = get_hyperparameters()

    vae = VAE(hparams=params)

    early_stop_callback = EarlyStopping(
        monitor='validation kld loss',
        min_delta=0.000,
        patience=5,
        verbose=False,
        mode='min')

    trainer = Trainer(#callbacks=[early_stop_callback],
                      max_epochs=params['epochs'],
                      logger=tb_logger,
                      accumulate_grad_batches=4,
                      gpus=1)
    trainer.fit(vae)

    generation = vae.sample(params['generated_samples'])




