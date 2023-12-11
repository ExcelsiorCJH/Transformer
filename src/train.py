import os

import torch
import omegaconf

from .module import Trainer
from .module.utils import fix_seed
from .dataset import ETTDataModule
from .model import Transformer

import warnings

warnings.filterwarnings(action="ignore")


def main(config) -> None:
    fix_seed(config.train.seed)

    # dm(datamodule)
    dm = ETTDataModule(**config.dm)

    # model
    model = Transformer(**config.model)

    # trainer
    trainer = Trainer(config=config, model=model, dm=dm)
    version = trainer.fit()

    return None


if __name__ == "__main__":
    config_path = "src/config/transformer_config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    main(config)
