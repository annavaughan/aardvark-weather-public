"""
NB: this script is for illustration purposes only and is not runnable as our
full dataset is not provided as part of the submission due to size constraints.
Many of the relevant paths to the data have been thus replaced by dummy paths.
"""

import os
import argparse
import pickle
import subprocess
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from loss_functions import WeightedRmseLoss, PressureWeightedRmseLoss, RmseLoss
from trainer import DDPTrainer
from loader import *
from models import *
from unet_wrap_padding import *

torch.set_float32_matmul_precision("medium")

weights = np.load(
    "path_to_lat_weights/weights_lat_1.npy",
).T[np.newaxis, ..., np.newaxis]


def weighted_rmse_loss(target, output):
    return np.sqrt(np.nanmean(((target - output) ** 2) * weights, axis=(0, 1, 2)))


def unnorm_era5(x, mean, std):
    x = x * std + mean
    return x


def norm_era5(x, mean, std):
    x = (x - mean) / std
    return x


def ddp_setup(rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank, world_size, output_dir, args):
    """
    Finetune the processor module for each leadtime
    """

    # Setup
    master_port = args.master_port
    lead_time = args.lead_time
    quicklook_id = args.output_dir
    era5_mode = args.era5_mode
    ddp_setup(rank, world_size, master_port)

    # Instantiate loss function
    if args.loss == "lw_rmse":
        lf = WeightedRmseLoss(args.res, weight_per_variable=False)
    elif args.loss == "lw_rmse_pressure_weighted":
        lf = PressureWeightedRmseLoss(args.res, args.era5_mode)
    elif args.loss == "rmse":
        lf = RmseLoss()

    # Load the pre-trained model
    with open(args.forecast_model_path + "/config.pkl", "rb") as handle:
        forecast_config = pickle.load(handle)

    model = ConvCNPWeather(
        in_channels=forecast_config["in_channels"],
        out_channels=forecast_config["out_channels"],
        int_channels=forecast_config["int_channels"],
        device="cuda",
        res=forecast_config["res"],
        gnp=bool(0),
        decoder=forecast_config["decoder"],
        mode=forecast_config["mode"],
        film=bool(forecast_config["film"]),
    )

    try:
        best_epoch = np.argmin(np.load(forecast_config["output_dir"] + "/losses_0.npy"))
        checkpoint = torch.load(
            forecast_config["output_dir"] + "/epoch_{}".format(best_epoch)
        )

    except:
        best_epoch = np.argmin(
            np.load("../results/" + forecast_config["output_dir"] + "/losses_0.npy")
        )
        checkpoint = torch.load(
            "../results/"
            + forecast_config["output_dir"]
            + "/epoch_{}".format(best_epoch)
        )

    new_state_dict = OrderedDict()
    for k, v in checkpoint["model_state_dict"].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    # Set the path to the prediction from the encoder
    path_to_context = args.assimilation_model_path

    # Finetune for each leadtime
    for lead_time in range(1, 11):
        print(f"Training lead time {lead_time}")

        # Setup the model
        model = ConvCNPWeather(
            in_channels=forecast_config["in_channels"],
            out_channels=forecast_config["out_channels"],
            int_channels=forecast_config["int_channels"],
            device="cuda",
            res=forecast_config["res"],
            gnp=bool(0),
            decoder=forecast_config["decoder"],
            mode=forecast_config["mode"],
            film=bool(forecast_config["film"]),
        )

        model.load_state_dict(new_state_dict)

        # Setup the loaders
        if lead_time == 1:
            path_to_context = args.assimilation_model_path
        else:
            path_to_context = f"{args.output_dir}/"

        print(f"Loading context data from {path_to_context}...")
        train_dataset = ForecastLoader(
            device="cuda",
            mode="train",
            lead_time=lead_time,
            era5_mode=args.era5_mode,
            res=args.res,
            frequency=args.frequency,
            diff=bool(args.diff),
            ic_path=path_to_context,
            finetune_step=lead_time,
            random_lt=False,
            finetune_eval_every=250,
            eval_steps=False,
        )

        val_dataset = ForecastLoader(
            device="cuda",
            mode="val",
            lead_time=lead_time,
            era5_mode=args.era5_mode,
            res=args.res,
            frequency=args.frequency,
            diff=bool(args.diff),
            ic_path=path_to_context,
            finetune_step=lead_time,
        )

        test_dataset = ForecastLoader(
            device="cuda",
            mode="test",
            lead_time=lead_time,
            era5_mode=args.era5_mode,
            res=args.res,
            frequency=args.frequency,
            diff=bool(args.diff),
            ic_path=path_to_context,
            finetune_step=lead_time,
        )

        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        test_sampler = DistributedSampler(test_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=test_sampler,
        )

        # Setup output dir
        try:
            output_dir = f"{args.output_dir}/forecast_{lead_time}/"
            os.mkdir(output_dir)
        except:
            pass

        # Run the finetuning
        trainer = DDPTrainer(
            model,
            rank,
            train_loader,
            val_loader,
            lf,
            output_dir,
            args.lr,
            train_sampler,
            weight_decay=args.weight_decay,
            tune_film=False,
        )

        # Finetune
        n_epochs = args.finetune_epochs
        trainer.train(n_epochs=n_epochs)

        torch.distributed.barrier()

        # Load the model just trained

        best_epoch = np.argmin(
            np.load(f"{args.output_dir}/forecast_{lead_time}/losses_0.npy")
        )
        checkpoint = torch.load(
            f"{args.output_dir}/forecast_{lead_time}/epoch_{best_epoch}"
        )

        new_state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.eval()

        # Now need to make predictions using this model to create data to train on for the next leadtime

        # Setup the output arrays
        if rank == 0:
            val_ic = np.memmap(
                f"{args.output_dir}/ic_val_{lead_time}.mmap",
                dtype="float32",
                mode="w+",
                shape=(len(val_dataset), 121, 240, 24),
            )

            test_ic = np.memmap(
                f"{args.output_dir}/ic_test_{lead_time}.mmap",
                dtype="float32",
                mode="w+",
                shape=(len(test_dataset), 121, 240, 24),
            )

            test_ic_target = np.memmap(
                f"{args.output_dir}/ic_test_target_{lead_time}.mmap",
                dtype="float32",
                mode="w+",
                shape=(len(test_dataset), 121, 240, 24),
            )

            train_ic = np.memmap(
                f"{args.output_dir}/ic_train_{lead_time}.mmap",
                dtype="float32",
                mode="w+",
                shape=(len(train_dataset), 121, 240, 24),
            )

        torch.distributed.barrier()
        val_ic = np.memmap(
            f"{args.output_dir}/ic_val_{lead_time}.mmap",
            dtype="float32",
            mode="r+",
            shape=(len(val_dataset), 121, 240, 24),
        )

        train_ic = np.memmap(
            f"{args.output_dir}/ic_train_{lead_time}.mmap",
            dtype="float32",
            mode="r+",
            shape=(len(train_dataset), 121, 240, 24),
        )

        test_ic = np.memmap(
            f"{args.output_dir}/ic_test_{lead_time}.mmap",
            dtype="float32",
            mode="r+",
            shape=(len(test_dataset), 121, 240, 24),
        )

        test_ic_target = np.memmap(
            f"{args.output_dir}/ic_test_target_{lead_time}.mmap",
            dtype="float32",
            mode="r+",
            shape=(len(test_dataset), 121, 240, 24),
        )

        print(f"{args.output_dir}/ic_train_{lead_time}.mmap")
        if lead_time > 1:
            if rank == 0:
                subprocess.run(
                    [
                        "rm",
                        f"{args.output_dir}/ic_train_{lead_time-1}.mmap",
                    ]
                )

        torch.distributed.barrier()

        # Generate predictions on the test set
        with tqdm(test_loader, unit="batch") as tepoch:
            for count, batch in enumerate(tepoch):
                inds = batch["target_index"].detach().cpu().numpy().astype(int)[:, 0]
                out = model(batch, film_index=batch["lt"])

                base_context = test_loader.dataset.unnorm_base_context(
                    batch["y_context"][:, :-11, ...]
                ).permute(0, 3, 2, 1)
                unnorm_pred = test_loader.dataset.unnorm_pred(out)
                unnorm_target = test_loader.dataset.unnorm_pred(batch["y_target"])

                unnorm_pred = unnorm_pred + base_context
                unnorm_target = unnorm_target + base_context

                test_ic[inds, ...] = unnorm_pred.detach().cpu()
                test_ic_target[inds, ...] = unnorm_target.detach().cpu()

        torch.distributed.barrier()

        # Generate predictions on the validation set
        with tqdm(val_loader, unit="batch") as tepoch:
            for count, batch in enumerate(tepoch):
                inds = batch["target_index"].detach().cpu().numpy().astype(int)[:, 0]
                out = model(batch, film_index=batch["lt"])

                base_context = val_loader.dataset.unnorm_base_context(
                    batch["y_context"][:, :-11, ...]
                ).permute(0, 3, 2, 1)
                unnorm_pred = val_loader.dataset.unnorm_pred(out)
                unnorm_pred = unnorm_pred + base_context

                val_ic[inds, ...] = unnorm_pred.detach().cpu()

        torch.distributed.barrier()

        # Generate predictions on the train set
        with tqdm(train_loader, unit="batch") as tepoch:
            for count, batch in enumerate(tepoch):
                inds = batch["target_index"].detach().cpu().numpy().astype(int)[:, 0]
                out = model(batch, film_index=batch["lt"])
                base_context = val_loader.dataset.unnorm_base_context(
                    batch["y_context"][:, :-11, ...]
                ).permute(0, 3, 2, 1)
                unnorm_pred = val_loader.dataset.unnorm_pred(out)
                unnorm_pred = unnorm_pred + base_context

                train_ic[inds, ...] = unnorm_pred.detach().cpu()

        torch.distributed.barrier()

        print("Setting up next lead time...")
        path_to_context = f"{args.output_dir}/"

        # Set the initial model weights for the next leadtime to be the current trained weights
        if lead_time > 1:
            best_epoch = np.argmin(
                np.load(f"{args.output_dir}/forecast_{lead_time}/losses_0.npy")
            )
            checkpoint = torch.load(
                f"{args.output_dir}/forecast_{lead_time}/epoch_{best_epoch}"
            )
            new_state_dict = OrderedDict()
            for k, v in checkpoint["model_state_dict"].items():
                name = k[7:]
                new_state_dict[name] = v

        torch.cuda.empty_cache()

    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir")
    parser.add_argument("--assimilation_model_path")
    parser.add_argument("--forecast_model_path")

    parser.add_argument("--loss", default="lw_rmse_pressure_weighted")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--master_port", default="12345")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lead_time", type=int, default=1)
    parser.add_argument("--era5_mode", default="4u")
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--res", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=6)

    parser.add_argument("--diff", type=int, default=1)
    parser.add_argument("--start_ind", type=int, default=0)
    parser.add_argument("--end_ind", type=int, default=24)
    parser.add_argument("--finetune_epochs", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda")

    # Setup results directory
    output_dir = args.output_dir
    try:
        os.mkdir(output_dir)
    except:
        pass

    # Save config
    config = vars(args)

    with open(output_dir + "/config.pkl", "wb") as f:
        pickle.dump(config, f)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=[world_size, output_dir, args], nprocs=world_size)
