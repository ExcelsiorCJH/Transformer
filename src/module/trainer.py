import os
import itertools
import logging

import yaml
import dill
import omegaconf

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .utils import AverageMeter, EarlyStopping


class Trainer:
    def __init__(self, model, dm, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.grad_scaler = None
        if config.cuda.use_amp:
            self.grad_scaler = torch.cuda.amp.GradScaler()

        # model
        self.model = model.to(self.device)
        if config.cuda.use_multi_gpu:
            self.model = nn.DataParallel(self.model)

        # dm(datamodule)
        self.dm = dm
        self.train_loader = self.dm.train_dataloader()
        self.val_loader = self.dm.val_dataloader()
        self.pred_len = self.config.dm.pred_len
        self.label_len = self.config.dm.label_len
        self.task_type = self.config.dm.task_type

        # optimizer
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        # criterion
        self.criterion = nn.MSELoss()

        # early-stopping
        self.early_stopping = EarlyStopping(config.train.patience)

        # model-saving options
        self.version = 0
        self.ckpt_paths = []
        while True:
            ckpt_dir = self.config.train.ckpt_dir
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)

            self.save_path = os.path.join(
                ckpt_dir,
                f"version-{self.version}-{self.config.model.task_name}-{self.config.dm.dataset_name.lower()}",
            )
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                break
            else:
                self.version += 1
        self.summarywriter = SummaryWriter(self.save_path)

        if self.dm.scaler is not None:
            scaler_path = os.path.join(self.save_path, "data_scaler.pkl")
            with open(scaler_path, "wb") as f:
                dill.dump(self.dm.scaler, f)

        self.global_step = 0
        self.global_val_loss = 1e5
        self.eval_step = self.config.train.eval_step
        logging.basicConfig(
            filename=os.path.join(self.save_path, "experiment.log"),
            level=logging.INFO,
            format="%(asctime)s > %(message)s",
        )

        # experiment-logging options
        self.best_result = {"version": self.version}

    def configure_optimizers(self):
        # optimizer
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.train.learning_rate
        )

        # lr_scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.config.train.lr_patience,
            factor=self.config.train.factor,
        )
        return optimizer, lr_scheduler

    def save_checkpoint(
        self, epoch: int, val_loss: float, model: nn.Module, save_last: bool = False
    ) -> None:
        if save_last:
            logging.info(f"Save last trained model at {val_loss:.4f}. Saving model ...")

            ckpt_path = os.path.join(
                self.save_path, f"epoch_last_{epoch}_{val_loss:.4f}.pt"
            )
        else:
            logging.info(
                f"Val loss decreased ({self.global_val_loss:.4f} → {val_loss:.4f}). Saving model ..."
            )
            self.global_val_loss = val_loss

            ckpt_path = os.path.join(self.save_path, f"epoch_{epoch}_{val_loss:.4f}.pt")

            save_top_k = self.config.train.save_top_k
            self.ckpt_paths.append(ckpt_path)
            if save_top_k < len(self.ckpt_paths):
                for path in self.ckpt_paths[:-save_top_k]:
                    os.remove(path)

                self.ckpt_paths = self.ckpt_paths[-save_top_k:]

        if self.config.cuda.use_multi_gpu:
            torch.save(model.module.state_dict(), ckpt_path)
        else:
            torch.save(model.state_dict(), ckpt_path)

    def fit(self) -> dict:
        for epoch in tqdm(range(self.config.train.epochs), desc="epoch"):
            logging.info(f"* Learning Rate: {self.optimizer.param_groups[0]['lr']:.5f}")
            result = self._train_epoch(epoch)

            # update checkpoint
            if result["val_loss"] < self.global_val_loss:
                self.save_checkpoint(epoch, result["val_loss"], self.model)

            # early stop check
            self.early_stopping(result["val_loss"], logging)
            if self.early_stopping.early_stop:
                logging.info("Early Stopping")
                break

            self.lr_scheduler.step(result["val_loss"])

        if self.config.train.save_last:
            self.save_checkpoint(epoch, result["val_loss"], self.model, save_last=True)

        self.summarywriter.close()
        return self.version

    def _train_epoch(self, epoch: int) -> dict:
        train_loss = AverageMeter()

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader),
            desc="train_steps",
            total=len(self.train_loader),
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # decoder input
            dec_inp = torch.zeros_like(
                batch["future_values"][:, -self.pred_len :, :]
            ).float()
            dec_inp = torch.cat(
                [batch["future_values"][:, : self.label_len, :], dec_inp], dim=1
            ).float()

            self.optimizer.zero_grad()
            if self.config.cuda.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        past_values=batch["past_values"],
                        past_time_features=batch["past_time_features"],
                        future_values=dec_inp.to(self.device),
                        future_time_features=batch["future_time_features"],
                    )
                    outputs = outputs["last_hidden_states"]

                    f_dim = -1 if self.task_type == "MS" else 0
                    predicts = outputs[:, -self.pred_len :, f_dim:]
                    targets = batch["future_values"][:, -self.pred_len :, f_dim:]

                    loss = self.criterion(predicts, targets)

                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                outputs = self.model(
                    past_values=batch["past_values"],
                    past_time_features=batch["past_time_features"],
                    future_values=dec_inp.to(self.device),
                    future_time_features=batch["future_time_features"],
                )
                outputs = outputs["last_hidden_states"]

                f_dim = -1 if self.task_type == "MS" else 0
                predicts = outputs[:, -self.pred_len : f_dim :]
                targets = batch["future_values"][:, -self.pred_len :, f_dim:]

                loss = self.criterion(predicts, targets)
                loss.backward()
                self.optimizer.step()

            train_loss.update(loss.item())

            self.global_step += 1
            if self.global_step % self.eval_step == 0:
                logging.info(
                    f"[Version {self.version} Epoch {epoch}] global step: {self.global_step}, train loss: {loss.item():.3f}"
                )

        train_loss = train_loss.avg
        val_loss = self.validate(epoch)

        # tensorboard writing
        self.summarywriter.add_scalars(
            "lr", {"lr": self.optimizer.param_groups[0]["lr"]}, epoch
        )
        self.summarywriter.add_scalars(
            "loss/step", {"val": val_loss, "train": train_loss}, self.global_step
        )
        self.summarywriter.add_scalars(
            "loss/epoch", {"val": val_loss, "train": train_loss}, epoch
        )

        logging.info(f"** global step: {self.global_step}, val loss: {val_loss:.4f}")
        return {"val_loss": val_loss}

    def validate(self, epoch: int) -> dict:
        val_loss = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.val_loader),
                desc="valid_steps",
                total=len(self.val_loader),
            ):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # decoder input
                dec_inp = torch.zeros_like(
                    batch["future_values"][:, -self.pred_len :, :]
                ).float()
                dec_inp = torch.cat(
                    [batch["future_values"][:, : self.label_len, :], dec_inp], dim=1
                ).float()

                outputs = self.model(
                    past_values=batch["past_values"],
                    past_time_features=batch["past_time_features"],
                    future_values=dec_inp.to(self.device),
                    future_time_features=batch["future_time_features"],
                )
                outputs = outputs["last_hidden_states"]

                f_dim = -1 if self.task_type == "MS" else 0
                predicts = outputs[:, -self.pred_len :, f_dim:]
                targets = batch["future_values"][:, -self.pred_len :, f_dim:]

                loss = self.criterion(predicts, targets)
                val_loss.update(loss.item())

        return val_loss.avg
