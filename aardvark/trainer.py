import sys
import subprocess

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from misc_downscaling_functionality import *
from models import *
from loss_functions import *

sys.path.append("..")


class DDPTrainer:
    """
    Main class for training models using DDP
    """

    def __init__(
        self,
        model,
        rank,
        train_loader,
        val_loader,
        loss_function,
        save_path,
        learning_rate,
        sampler,
        weight_decay,
        test_loader=None,
        weights_path=None,
        tune_film=False,
    ):
        self.rank = rank
        self.model = model
        self.sampler = sampler
        self.tune_film = tune_film
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.loss_function = loss_function
        self.best_loss = 1000
        self.test_loader = test_loader

        self.model = self.model.to(rank)
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)

        if self.model.module.decoder == "vit":
            self.opt = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.95),
                weight_decay=1e-5,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, 891 * 80)

        else:
            self.opt = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        self.losses = []
        self.train_losses = []
        self.maes = []

    def _unravel_to_numpy(self, x):
        return x.view(-1).detach().cpu().numpy()

    def eval_epoch(self, fix_sigma, epoch):

        self.model.eval()

        mae = []
        lf = []
        lf_unnorm = []

        with torch.no_grad():
            for count, task in enumerate(self.val_loader):

                out = self.model(task, film_index=0)

                prev_step = None

                mae.append(
                    np.nanmean(
                        np.abs(
                            task["y_target"].detach().cpu()
                            - out[..., : task["y_target"].shape[-1]].detach().cpu()
                        )
                    )
                )
                l = (
                    self.loss_function(
                        task["y_target"], out, prev_step, fix_sigma=fix_sigma
                    )
                    .detach()
                    .item()
                )
                lf.append(l)

                try:
                    ic = self.train_loader.dataset.unnorm_base_context(
                        task["y_context"][:, :-11, ...]
                    ).permute(0, 3, 2, 1)
                    unnorm_pred = self.train_loader.dataset.unnorm_pred(out)
                    unnorm_target = self.train_loader.dataset.unnorm_pred(
                        task["y_target"]
                    )

                    unnorm_pred = unnorm_pred + ic
                    unnorm_target = unnorm_target + ic

                    lu = (
                        self.loss_function(
                            unnorm_target,
                            unnorm_pred,
                            prev_step,
                            fix_sigma=fix_sigma,
                            expand=True,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    lf_unnorm.append(lu)

                except:
                    pass

            if self.test_loader is not None:
                forecasts = []
                targets = []
                stations = []

                for count, task in enumerate(self.test_loader):
                    out = self.model(task, film_index=0)
                    forecasts.append(unnormalise_hadisd_tas(out.detach().cpu().numpy()))
                    targets.append(
                        unnormalise_hadisd_tas(task["y_target"].detach().cpu().numpy())
                    )
                    stations.append(
                        task["downscaling"]["x_target"].detach().cpu().numpy() * 360
                    )

                # Save the test outputs
                np.save(
                    self.save_path + f"forecast_{self.rank}_{self.epoch}.npy",
                    np.concatenate(forecasts, axis=0),
                )
                np.save(
                    self.save_path + f"targets_{self.rank}_{self.epoch}.npy",
                    np.concatenate(targets, axis=0),
                )
                np.save(
                    self.save_path + f"stations_{self.rank}_{self.epoch}.npy",
                    np.concatenate(stations, axis=0),
                )

        log_loss = np.nanmean(np.array(lf))

        if log_loss < self.best_loss:
            np.save(
                self.save_path + "unnorm_preds.npy",
                self.train_loader.dataset.unnorm_pred(out).detach().cpu().numpy(),
            )
            np.save(
                self.save_path + "unnorm_targets.npy",
                self.train_loader.dataset.unnorm_pred(task["y_target"])
                .detach()
                .cpu()
                .numpy(),
            )
        log_loss_unnorm = np.nanmean(np.stack(lf_unnorm), axis=0)

        if np.logical_and(self.rank == 0, self.epoch % 5 == 0):

            np.save(self.save_path + "preds_eval.npy", out.cpu().numpy())
            np.save(
                self.save_path + "y_target_eval.npy", task["y_target"].cpu().numpy()
            )

        return log_loss, log_loss_unnorm

    def train(self, n_epochs=100):

        torch.cuda.set_device(self.rank)
        subprocess.run(["cp", "reproduce_training.sh", f"{self.save_path}"])

        train_loss = []
        ll = []

        fix_sigma = False
        prev_step = None

        self.epoch = 0
        epoch_loss, log_loss_unnorm = self.eval_epoch(fix_sigma, 0)
        train_loss = np.mean(train_loss)

        for epoch in range(n_epochs):
            self.epoch = epoch

            self.sampler.set_epoch(epoch)

            self.model.train()
            train_loss = []
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for count, task in enumerate(tepoch):

                    out = self.model(task, film_index=0)

                    loss = self.loss_function(
                        task["y_target"], out, prev_step, fix_sigma=fix_sigma
                    )

                    loss.backward()
                    tepoch.set_postfix(loss=loss.item())
                    prev_step = out

                    self.opt.step()
                    self.opt.zero_grad()
                    train_loss.append(loss.item())
                    try:
                        if self.model.module.decoder == "vit":
                            if epoch > 20:
                                self.scheduler.step()
                    except:
                        pass

            epoch_loss, log_loss_unnorm = self.eval_epoch(fix_sigma, epoch)
            train_loss = np.mean(train_loss)
            ll.append(log_loss_unnorm)

            self.losses.append(epoch_loss)
            self.train_losses.append(train_loss)
            np.save(
                self.save_path + "losses_{}.npy".format(self.rank),
                np.array(self.losses),
            )
            np.save(
                self.save_path + "train_losses_{}.npy".format(self.rank),
                np.array(self.train_losses),
            )
            np.save(self.save_path + "rmse_{}.npy".format(self.rank), np.array(ll))

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

                if self.rank == 0:
                    if self.model.module.decoder == "vit":
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.opt.state_dict(),
                                "scheduler_state_dict": self.scheduler.state_dict(),
                                "loss": epoch_loss,
                            },
                            self.save_path + "epoch_{}".format(epoch),
                        )
                    else:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.opt.state_dict(),
                                "loss": epoch_loss,
                            },
                            self.save_path + "epoch_{}".format(epoch),
                        )

                try:
                    np.save(
                        self.save_path + "preds_train.npy".format(epoch),
                        out.detach().cpu().numpy(),
                    )
                    np.save(
                        self.save_path + "y_target_train.npy".format(epoch),
                        task["y_target"].detach().cpu().numpy(),
                    )
                except:
                    pass


class DDPTrainerE2E:
    """
    Training class for E2E
    """

    def __init__(
        self,
        model,
        rank,
        train_loader,
        val_loader,
        loss_function,
        save_path,
        learning_rate,
        sampler,
        weight_decay,
        hadisd_variable_name,
        test_loader=None,
        weights_path=None,
        tune_film=False,
    ):

        self.rank = rank
        self.model = model
        self.sampler = sampler
        self.tune_film = tune_film
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.loss_function = loss_function
        self.best_loss = 1000
        self.test_loader = test_loader
        self.hadisd_variable_name = hadisd_variable_name

        self.model = self.model.to(rank)
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)

        if self.model.module.decoder == "vit":
            self.opt = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.95),
                weight_decay=1e-5,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, 891 * 80)
        else:
            self.opt = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        self.losses = []
        self.train_losses = []

        self.maes = []

    def _unravel_to_numpy(self, x):
        return x.view(-1).detach().cpu().numpy()

    def eval_epoch(self, fix_sigma, epoch):

        self.model.eval()

        mae = []
        lf = []
        lf_unnorm = []

        with torch.no_grad():
            forecasts = []
            targets = []
            stations = []
            indices = []
            for count, task in tqdm(enumerate(self.val_loader)):

                out = self.model(task, film_index=0)
                forecasts.append(
                    unnormalise_hadisd_var(
                        out.detach().cpu().numpy(), self.hadisd_variable_name
                    )
                )
                targets.append(
                    unnormalise_hadisd_var(
                        task["y_target"].detach().cpu().numpy(),
                        self.hadisd_variable_name,
                    )
                )
                stations.append(
                    task["downscaling"]["x_target"].detach().cpu().numpy() * 360
                )
                indices.append(task["index"])

                prev_step = None

                mae.append(
                    np.nanmean(
                        np.abs(
                            task["y_target"].detach().cpu()
                            - out[..., : task["y_target"].shape[-1]].detach().cpu()
                        )
                    )
                )
                l = (
                    self.loss_function(
                        task["y_target"], out, prev_step, fix_sigma=fix_sigma
                    )
                    .detach()
                    .item()
                )
                lf.append(l)

                try:
                    ic = self.train_loader.dataset.unnorm_base_context(
                        task["y_context"][:, :-11, ...]
                    ).permute(0, 3, 2, 1)
                    unnorm_pred = self.train_loader.dataset.unnorm_pred(out)
                    unnorm_target = self.train_loader.dataset.unnorm_pred(
                        task["y_target"]
                    )

                    unnorm_pred = unnorm_pred + ic
                    unnorm_target = unnorm_target + ic

                    lu = (
                        self.loss_function(
                            unnorm_target,
                            unnorm_pred,
                            prev_step,
                            fix_sigma=fix_sigma,
                            expand=True,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    lf_unnorm.append(lu)

                except:
                    pass

            # Save the test outputs
            np.save(
                self.save_path + f"val_forecast_{self.rank}_{self.epoch}.npy",
                np.concatenate(forecasts, axis=0),
            )
            np.save(
                self.save_path + f"val_targets_{self.rank}_{self.epoch}.npy",
                np.concatenate(targets, axis=0),
            )
            np.save(
                self.save_path + f"val_stations_{self.rank}_{self.epoch}.npy",
                np.concatenate(stations, axis=0),
            )
            np.save(
                self.save_path + f"val_indices_{self.rank}_{self.epoch}.npy",
                np.concatenate(indices, axis=0),
            )

            if self.test_loader is not None:
                forecasts = []
                targets = []
                stations = []
                indices = []

                for count, task in enumerate(self.test_loader):
                    out = self.model(task, film_index=0)
                    forecasts.append(
                        unnormalise_hadisd_var(
                            out.detach().cpu().numpy(), self.hadisd_variable_name
                        )
                    )
                    targets.append(
                        unnormalise_hadisd_var(
                            task["y_target"].detach().cpu().numpy(),
                            self.hadisd_variable_name,
                        )
                    )
                    stations.append(
                        task["downscaling"]["x_target"].detach().cpu().numpy() * 360
                    )
                    indices.append(task["index"])

                np.save(
                    self.save_path + f"test_forecast_{self.rank}_{self.epoch}.npy",
                    np.concatenate(forecasts, axis=0),
                )
                np.save(
                    self.save_path + f"test_targets_{self.rank}_{self.epoch}.npy",
                    np.concatenate(targets, axis=0),
                )
                np.save(
                    self.save_path + f"test_stations_{self.rank}_{self.epoch}.npy",
                    np.concatenate(stations, axis=0),
                )
                np.save(
                    self.save_path + f"test_indices_{self.rank}_{self.epoch}.npy",
                    np.concatenate(indices, axis=0),
                )

        log_loss = np.nanmean(np.array(lf))

        if log_loss < self.best_loss:
            np.save(
                self.save_path + "unnorm_preds.npy",
                self.train_loader.dataset.unnorm_pred(out).detach().cpu().numpy(),
            )
            np.save(
                self.save_path + "unnorm_targets.npy",
                self.train_loader.dataset.unnorm_pred(task["y_target"])
                .detach()
                .cpu()
                .numpy(),
            )
        log_loss_unnorm = np.nanmean(np.stack(lf_unnorm), axis=0)

        if np.logical_and(self.rank == 0, self.epoch % 5 == 0):

            np.save(self.save_path + "preds_eval.npy", out.cpu().numpy())
            np.save(
                self.save_path + "y_target_eval.npy", task["y_target"].cpu().numpy()
            )

        return log_loss, log_loss_unnorm

    def train(self, n_epochs=100):

        torch.cuda.set_device(self.rank)
        subprocess.run(["cp", "reproduce_training.sh", f"{self.save_path}"])

        train_loss = []
        ll = []

        fix_sigma = False
        prev_step = None

        self.epoch = 0
        epoch_loss, log_loss_unnorm = self.eval_epoch(fix_sigma, 0)
        train_loss = np.mean(train_loss)

        for epoch in range(n_epochs):
            self.epoch = epoch
            self.sampler.set_epoch(epoch)

            epoch_loss, log_loss_unnorm = self.eval_epoch(fix_sigma, epoch)
            train_loss = np.mean(train_loss)
            ll.append(log_loss_unnorm)

            self.losses.append(epoch_loss)
            self.train_losses.append(train_loss)
            np.save(
                self.save_path + "losses_{}.npy".format(self.rank),
                np.array(self.losses),
            )
            np.save(
                self.save_path + "train_losses_{}.npy".format(self.rank),
                np.array(self.train_losses),
            )
            np.save(self.save_path + "rmse_{}.npy".format(self.rank), np.array(ll))

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

                if self.rank == 0:
                    if self.model.module.decoder == "vit":
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.opt.state_dict(),
                                "scheduler_state_dict": self.scheduler.state_dict(),
                                "loss": epoch_loss,
                            },
                            self.save_path + "epoch_{}".format(epoch),
                        )
                    else:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.opt.state_dict(),
                                "loss": epoch_loss,
                            },
                            self.save_path + "epoch_{}".format(epoch),
                        )

                try:
                    np.save(
                        self.save_path + "preds_train.npy".format(epoch),
                        out.detach().cpu().numpy(),
                    )
                    np.save(
                        self.save_path + "y_target_train.npy".format(epoch),
                        task["y_target"].detach().cpu().numpy(),
                    )
                except:
                    pass

            self.model.train()
            train_loss = []
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for count, task in tqdm(enumerate(tepoch)):
                    out = self.model(task, film_index=0)

                    loss = self.loss_function(
                        task["y_target"], out, prev_step, fix_sigma=fix_sigma
                    )

                    loss.backward()
                    tepoch.set_postfix(loss=loss.item())
                    prev_step = out

                    self.opt.step()
                    self.opt.zero_grad()
                    train_loss.append(loss.item())
                    if self.model.module.decoder == "vit":
                        if epoch > 20:
                            self.scheduler.step()
