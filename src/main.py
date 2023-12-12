import os
import logging

import torch
import omegaconf
import numpy as np

from tqdm.auto import tqdm
from .dataset import ETTDataModule
from .model import Transformer
from .module import Trainer
from .module.utils import fix_seed, load_trained_model, plot

import warnings

warnings.filterwarnings(action="ignore")


def main(config) -> None:
    fix_seed(config.train.seed)

    # dm(datamodule)
    dm = ETTDataModule(**config.dm)
    # model
    model = Transformer(**config.model)

    # train
    version = train(config, dm, model)

    # test
    config.version = version
    test(config, dm, model)

    return None


def train(config, dm, model) -> int:
    # trainer & train
    trainer = Trainer(config=config, model=model, dm=dm)
    version = trainer.fit()

    return version


def test(config, dm, model) -> None:
    pred_len = config.dm.pred_len
    label_len = config.dm.label_len
    task_type = config.dm.task_type

    # model load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = load_trained_model(config)
    model = Transformer(**config.model)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # dataloader
    test_dataloader = dm.test_dataloader()

    # save results
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(
        save_dir,
        f"version-{config.version}-{config.model.task_name}-{config.dm.dataset_name.lower()}",
    )
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    inputs = []
    predicts, targets = [], []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(
            tqdm(test_dataloader, total=len(test_dataloader), desc="test")
        ):
            batch = {k: v.to(device) for k, v in batch.items()}

            # decoder input
            dec_inp = torch.zeros_like(batch["future_values"][:, -pred_len:, :]).float()
            dec_inp = torch.cat(
                [batch["future_values"][:, :label_len, :], dec_inp], dim=1
            ).float()

            output = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                future_values=dec_inp.to(device),
                future_time_features=batch["future_time_features"],
            )
            output = output["last_hidden_states"]

            f_dim = -1 if task_type == "MS" else 0
            predict = output[:, -pred_len:, f_dim:]
            target = batch["future_values"][:, -pred_len:, f_dim:]

            predict = predict.cpu().numpy()
            target = target.cpu().numpy()

            past_values = batch["past_values"]
            past_values = past_values.cpu().numpy()

            if dm.scaler is not None:
                shape = predict.shape
                predict = dm.scaler.inverse_transform(predict.squeeze(0)).reshape(shape)
                target = dm.scaler.inverse_transform(target.squeeze(0)).reshape(shape)
                past_values = dm.scaler.inverse_transform(
                    past_values.squeeze(0)
                ).reshape(shape)

            predict = predict[:, :, f_dim:]
            target = target[:, :, f_dim:]

            inputs.append(past_values)
            predicts.append(predict)
            targets.append(target)

            if idx % 500 == 0:
                gt = np.concatenate(
                    (past_values[0, :, f_dim], target[0, :, f_dim]), axis=0
                )
                pr = np.concatenate(
                    (past_values[0, :, f_dim], predict[0, :, f_dim]), axis=0
                )

                fname = os.path.join(save_path, f"{idx}.png")
                plot(gt, pr, fname)

    # save results
    predicts = np.array(predicts)
    targets = np.array(targets)

    predicts = predicts.squeeze(1)
    targets = targets.squeeze(1)


if __name__ == "__main__":
    config_path = "src/config/transformer_config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    main(config)
