"""
NB: this script is for illustration purposes only and is not runnable as our
full dataset is not provided as part of the submission, due to size constraints.
Many of the relevant paths to the data have been thus replaced by dummy paths.
"""

import os
import sys
import pickle
import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group


from trainer import DDPTrainer
from loss_functions import WeightedRmseLoss, PressureWeightedRmseLoss, RmseLoss
from misc_downscaling_functionality import ConvCNPWeatherOnToOff, DownscalingRmseLoss
from loader import *
from models import *
from unet_wrap_padding import *


sys.path.append("../npw/data")
torch.set_float32_matmul_precision("medium")


def ddp_setup(rank, world_size, master_port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def start_date(name):
    if name == "train":
        return "2007-01-02"
    elif name == "val":
        return "2019-01-01"
    elif name == "test":
        return "2018-01-01"
    else:
        raise Exception(f"Unrecognised split name {name}")


def end_date(name):
    if name == "train":
        return "2017-12-31"
    elif name == "val":
        return "2019-11-01"
    elif name == "test":
        return "2018-12-21"
    else:
        raise Exception(f"Unrecognised split name {name}")


def main(rank, world_size, output_dir, args):
    """
    Primary training script for the encoder, processor and decoder modules.
    """

    master_port = args.master_port
    lead_time = args.lead_time
    era5_mode = args.era5_mode
    weights_dir = args.weights_dir
    ddp_setup(rank, world_size, master_port)

    # Instantiate loss function
    if args.loss == "lw_rmse":
        lf = WeightedRmseLoss(
            args.res,
            start_ind=args.start_ind,
            end_ind=args.end_ind,
            weight_per_variable=bool(args.weight_per_variable),
        )
    elif args.loss == "lw_rmse_pressure_weighted":
        lf = PressureWeightedRmseLoss(args.res, era5_mode)
    elif args.loss == "rmse":
        lf = RmseLoss()
    elif args.loss == "downscaling_rmse":
        lf = DownscalingRmseLoss()

    # Setup datasets

    # Case 1: training encoder
    if args.mode == "assimilation":
        train_dataset = WeatherDatasetAssimilation(
            device="cuda",
            hadisd_mode="train",
            start_date="2007-01-02",
            end_date="2017-12-31",
            lead_time=0,
            era5_mode="4u",
            res=args.res,
            var_start=args.start_ind,
            var_end=args.end_ind,
            diff=bool(args.diff),
            two_frames=bool(args.two_frames),
        )
        val_dataset = WeatherDatasetAssimilation(
            device="cuda",
            hadisd_mode="train",
            start_date="2019-01-01",
            end_date="2019-12-31",
            lead_time=0,
            era5_mode="4u",
            res=args.res,
            var_start=args.start_ind,
            var_end=args.end_ind,
            diff=bool(args.diff),
            two_frames=bool(args.two_frames),
        )

    # Case 2: training processor
    elif args.mode == "forecast":
        if args.ic == "aardvark":
            train_dataset = FineTuneForecastLoaderNew(
                device="cuda",
                mode="train",
                lead_time=lead_time,
                era5_mode=era5_mode,
                res=args.res,
                frequency=args.frequency,
                diff=bool(args.diff),
                aardvark_ic_path=args.aardvark_ic_path,
                random_lt=True,
            )
            val_dataset = FineTuneForecastLoaderNew(
                device="cuda",
                mode="val",
                lead_time=lead_time,
                era5_mode=era5_mode,
                res=args.res,
                frequency=args.frequency,
                diff=bool(args.diff),
                aardvark_ic_path=args.aardvark_ic_path,
            )
        else:
            train_dataset = ForecastLoader(
                device="cuda",
                mode="train",
                lead_time=lead_time,
                era5_mode=era5_mode,
                res=args.res,
                frequency=args.frequency,
                diff=bool(args.diff),
                u_only=False,
                random_lt=False,
            )
            val_dataset = ForecastLoader(
                device="cuda",
                mode="val",
                lead_time=lead_time,
                era5_mode=era5_mode,
                res=args.res,
                frequency=args.frequency,
                diff=bool(args.diff),
                u_only=False,
                random_lt=False,
            )

    # Case 3: training decoder
    elif args.mode == "downscaling":

        train_dataset = ForecasterDatasetDownscaling(
            start_date="2007-01-02",
            end_date="2017-12-31",
            lead_time=args.lead_time,
            hadisd_var=args.var,
            mode="train",
            device="cuda",
            forecast_path=None,
        )

        val_dataset = ForecasterDatasetDownscaling(
            start_date="2019-01-01",
            end_date="2019-12-21",
            lead_time=args.lead_time,
            hadisd_var=args.var,
            mode="train",
            device="cuda",
            forecast_path=None,
        )

        try:
            os.mkdir(f"{output_dir}lt_{args.lead_time}")
        except FileExistsError:
            pass

        output_dir = f"{output_dir}lt_{args.lead_time}/"

    # Instantiate model

    if args.mode == "downscaling":
        model = ConvCNPWeatherOnToOff(
            in_channels=args.in_channels,
            out_channels=args.end_ind - args.start_ind,
            int_channels=args.int_channels,
            device="cuda",
            res=args.res,
            decoder=args.decoder,
            mode=args.mode,
            film=bool(args.film),
        )
    else:
        model = ConvCNPWeather(
            in_channels=args.in_channels,
            out_channels=args.end_ind - args.start_ind,
            int_channels=args.int_channels,
            device="cuda",
            res=args.res,
            gnp=bool(0),
            decoder=args.decoder,
            mode=args.mode,
            film=bool(args.film),
            two_frames=bool(args.two_frames),
        )

    # Instantiate loaders
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
    )

    # Instantiate trainer

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
        weights_path=weights_dir,
        tune_film=args.film,
    )

    # Train model

    trainer.train(n_epochs=args.epoch)
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    parser.add_argument("--mode")
    parser.add_argument("--weights_dir")
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--out_channels", type=int)
    parser.add_argument("--int_channels", type=int)
    parser.add_argument("--loss")
    parser.add_argument("--ic")
    parser.add_argument("--decoder")
    parser.add_argument("--film")
    parser.add_argument("--aardvark_ic_path")
    parser.add_argument("--two_frames", type=int, default=0)
    parser.add_argument("--weight_per_variable", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--master_port", default="12345")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lead_time", type=int)
    parser.add_argument("--era5_mode", default="4u")
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--res", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=6)
    parser.add_argument("--diff", type=int, default=1)
    parser.add_argument("--start_ind", type=int, default=0)
    parser.add_argument("--end_ind", type=int, default=24)
    parser.add_argument("--downscaling_train_start_date", default="1979-01-01")
    parser.add_argument("--downscaling_train_end_date", default="2017-12-31")
    parser.add_argument("--downscaling_context", default="era5")
    parser.add_argument("--downscaling_lead_time", type=int)
    parser.add_argument("--var", default=None)
    args = parser.parse_args()

    torch.device("cuda")

    # Create results directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Save config
    with open(output_dir + "/config.pkl", "wb") as f:
        pickle.dump(vars(args), f)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=[world_size, output_dir, args], nprocs=world_size)
