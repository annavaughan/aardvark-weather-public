"""
NB: this script is for illustration purposes only and is not runnable as our
full dataset is not provided as part of the submission due to size constraints.
Many of the relevant paths to the data have been thus replaced by dummy paths.
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from loader import WeatherDatasetAssimilation
from models import *

torch.set_float32_matmul_precision("medium")


def unnorm(x, mean, std, diff=False, av_2019=None):

    x = x * std + mean
    if diff:
        return x + av_2019.transpose(0, 3, 2, 1)
    return x


if __name__ == "__main__":

    """
    Generate encoder predictions to be used as finetuning data for the processor module
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_model_path")
    args = parser.parse_args()

    # Load experiment config
    with open(args.encoder_model_path + "/config.pkl", "rb") as handle:
        forecast_config = pickle.load(handle)

    device = "cuda"

    # Setup normalisation factors
    era5_mean_spatial = None
    means = np.load(
        "aux_data_path/norm_factors/mean_{}_{}.npy".format(
            forecast_config["era5_mode"], forecast_config["res"]
        )
    )[np.newaxis, np.newaxis, np.newaxis, :]
    stds = np.load(
        "aux_data_path/norm_factors/std_{}_{}.npy".format(
            forecast_config["era5_mode"], forecast_config["res"]
        )
    )[np.newaxis, np.newaxis, np.newaxis, :]

    # Specify dates to generate predictions for
    labels = ["train", "test", "val"]
    dates = [
        ["2007-01-02", "2017-12-31"],
        ["2018-01-01", "2018-12-31"],
        ["2019-01-01", "2019-12-31"],
    ]

    # Iterate over loaders
    for label, date in zip(labels, dates):

        n_times = pd.date_range(date[0], date[1], freq="6H")

        # Setup a memmap to write out to
        ic = np.memmap(
            "{}/ic_{}.mmap".format(args.encoder_model_path, label),
            dtype="float32",
            mode="w+",
            shape=(len(n_times), 121, 240, 24),
        )

        var_group_preds = []
        var_group_targets = []

        # Setup loader
        dataset = WeatherDatasetAssimilation(
            device="cuda",
            hadisd_mode="train",
            start_date=date[0],
            end_date=date[1],
            lead_time=0,
            era5_mode="4u",
            res=1,
            var_start=0,
            var_end=24,
            diff=False,
        )

        loader = DataLoader(dataset, batch_size=64, shuffle=False)  # ,

        # Instantiate and load model
        model = ConvCNPWeather(
            in_channels=forecast_config["in_channels"],
            out_channels=forecast_config["out_channels"],
            int_channels=forecast_config["int_channels"],
            device="cuda",
            res=forecast_config["res"],
            gnp=bool(0),
            decoder=forecast_config["decoder"],
            mode=forecast_config["mode"],
            film=bool(0),
        )

        best_epoch = np.argmin(
            np.load("{}/losses_0.npy".format(args.encoder_model_path))
        )
        state_dict = torch.load(
            "{}/epoch_{}".format(args.encoder_model_path, best_epoch),
            map_location=device,
        )["model_state_dict"]
        state_dict = {k[7:]: v for k, v in zip(state_dict.keys(), state_dict.values())}
        model.load_state_dict(state_dict)
        model = nn.DataParallel(model)
        model = model.cuda()

        model.eval()

        # Generate predictions
        total = []
        target = []

        sum_count = 0
        with torch.no_grad():
            with tqdm(loader, unit="batch") as tepoch:
                for count, batch in enumerate(tepoch):

                    out = model(batch, film_index=batch["lt"]).detach().cpu().numpy()

                    out_unnorm = unnorm(
                        out,
                        means,
                        stds,
                        diff=False,
                        av_2019=era5_mean_spatial,
                    )

                    ic[sum_count : sum_count + out.shape[0], ...] = out_unnorm
                    sum_count += out.shape[0]
