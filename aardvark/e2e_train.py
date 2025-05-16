import os
import sys
import pickle
import argparse

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from misc_downscaling_functionality import DownscalingRmseLoss
from e2e_model import *
from loader import *
from models import *
from unet_wrap_padding import *
from trainer import DDPTrainerE2E
from loss_functions import WeightedRmseLoss, PressureWeightedRmseLoss, RmseLoss

torch.set_float32_matmul_precision("medium")


def ddp_setup(rank, world_size, master_port):
    """
    Setup DDP
    """

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def start_date(name):
    """
    Set split start dates
    """

    if name == "train":
        return "2007-01-02"
    elif name == "val":
        return "2019-01-01"
    elif name == "test":
        return "2018-01-01"
    else:
        raise Exception(f"Unrecognised split name {name}")


def end_date(name):
    """
    Set split end dates
    """

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
    Run end to end training
    """

    master_port = args.master_port
    lead_time = args.lead_time
    era5_mode = args.era5_mode
    ddp_setup(rank, world_size, master_port)

    # Setup loss function
    if args.loss == "lw_rmse":
        lf = WeightedRmseLoss(
            args.res,
            start_ind=0,
            end_ind=24,
            weight_per_variable=False,
        )
    elif args.loss == "lw_rmse_pressure_weighted":
        lf = PressureWeightedRmseLoss(args.res, era5_mode)
    elif args.loss == "rmse":
        lf = RmseLoss()
    elif args.loss == "downscaling_rmse":
        lf = DownscalingRmseLoss()

    # Instantiate model
    model = ConvCNPWeatherE2E(
        device="cuda",
        lead_time=lead_time,
        se_model_path=args.se_model_path,
        forecast_model_path=args.forecast_model_path,
        sf_model_path=args.sf_model_path,
    )
    dist.barrier()

    # Instantiate loaders
    train_dataset = WeatherDatasetE2E(
        device="cuda",
        hadisd_mode="train",
        start_date="2007-01-02",
        end_date="2017-12-31",
        lead_time=lead_time,
        era5_mode="4u",
        mode="train",
        res=args.res,
        var_start=0,
        var_end=24,
        diff=bool(0),
        two_frames=bool(0),
        region=args.region,
        hadisd_var=args.var,
        max_steps_per_epoch=args.max_steps_per_epoch,
    )

    val_dataset = WeatherDatasetE2E(
        device="cuda",
        hadisd_mode="train",
        start_date="2019-01-01",
        end_date="2019-12-21",
        lead_time=lead_time,
        era5_mode="4u",
        mode="train",
        res=args.res,
        var_start=0,
        var_end=24,
        diff=bool(0),
        two_frames=bool(0),
        region=args.region,
        hadisd_var=args.var,
    )

    test_dataset = WeatherDatasetE2E(
        device="cuda",
        hadisd_mode="train",
        start_date="2018-01-01",
        end_date="2018-12-21",
        lead_time=lead_time,
        era5_mode="4u",
        mode="train",
        res=args.res,
        var_start=0,
        var_end=24,
        diff=bool(0),
        two_frames=bool(0),
        region=args.region,
        hadisd_var=args.var,
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
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
    )

    # Instantiate trainer
    trainer = DDPTrainerE2E(
        model,
        rank,
        train_loader,
        val_loader,
        lf,
        output_dir,
        args.lr,
        train_sampler,
        hadisd_variable_name=args.var,
        weight_decay=args.weight_decay,
        weights_path=None,
        tune_film=0,
        test_loader=test_loader,
    )

    # Train model
    trainer.train(n_epochs=args.epoch)
    destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    parser.add_argument("--loss")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--master_port", default="12345")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lead_time", type=int)
    parser.add_argument("--era5_mode", default="4u")
    parser.add_argument("--sf_model_path")
    parser.add_argument("--se_model_path")
    parser.add_argument("--forecast_model_path")

    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_steps_per_epoch", type=int)
    parser.add_argument("--res", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=6)
    parser.add_argument("--region", default="global")
    parser.add_argument("--var", choices=["tas", "u", "v", "psl", "ws"])
    args = parser.parse_args()

    torch.device("cuda")

    # Make results directory
    output_dir = args.output_dir
    try:
        os.mkdir(output_dir)
    except:
        pass

    # Save config
    with open(output_dir + "/config.pkl", "wb") as f:
        pickle.dump(vars(args), f)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=[world_size, output_dir, args], nprocs=world_size)
