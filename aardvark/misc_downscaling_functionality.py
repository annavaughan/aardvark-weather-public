import pickle

import torch
import numpy as np
import torch.nn as nn

from set_convs import convDeepSet
from unet_wrap_padding import Unet
from vit import *
from models import *

hadisd_publisher_shifts = {
    "tas": 273.15,
    "u": 0.0,
    "v": 0.0,
    "psl": 0.0,
    "ws": 0.0,
}

hadisd_publisher_scales = {
    "tas": 10,
    "u": 10,
    "v": 10,
    "psl": 100,
    "ws": 10.0,
}


def hadisd_normalisation_factors(var: str):
    path = "/home/azureuser/aux_data/norm_factors/"
    return {
        "mean": np.load(path + f"mean_hadisd_{var}.npy"),
        "std": np.load(path + f"std_hadisd_{var}.npy"),
    }


def unnormalise_hadisd_var(x, var):
    factors = hadisd_normalisation_factors(var)
    hadisd_shift = hadisd_publisher_shifts[var]
    hadisd_scale = hadisd_publisher_scales[var]

    return hadisd_shift + hadisd_scale * (factors["mean"] + factors["std"] * x)


class DownscalingRmseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, output, prev_step, fix_sigma=None, expand=True):

        target = torch.flatten(target.to(output.device))
        output = torch.flatten(output)

        tmp = torch.isnan(target)
        clean_target = target[~tmp]
        clean_output = output[~tmp]

        return torch.mean((clean_target - clean_output) ** 2)


class ConvCNPWeatherOnToOff(nn.Module):
    """
    ConvCNP for decoder
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        int_channels,
        device,
        res,
        data_path="../data/",
        mode="end_to_end",
        decoder=None,
        film=False,
    ):

        super().__init__()

        # Setup
        self.device = device

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.int_channels = int_channels
        self.decoder = decoder
        self.int_x = 256
        self.int_y = 128
        self.mode = mode
        self.film = film

        # Load lon-lat of internal discretisation
        self.era5_x = (
            torch.from_numpy(
                np.load(data_path + "grid_lon_lat/era5_x_{}.npy".format(res))
            ).float()
            / 360
        )
        self.era5_y = (
            torch.from_numpy(
                np.load(data_path + "grid_lon_lat/era5_y_{}.npy".format(res))
            ).float()
            / 360
        )

        # Setup setconv
        self.sc_out = convDeepSet(
            0.001, "OnToOff", density_channel=False, device=self.device
        )

        if self.mode not in ["downscaling", "end_to_end"]:
            unet_out_channels = out_channels
        else:
            unet_out_channels = int_channels

        # UNet backbone
        if self.decoder == "base":
            self.decoder_lr = Unet(
                in_channels=in_channels,
                out_channels=unet_out_channels,
                div_factor=1,
                film=film,
            )

        else:
            raise Exception(f"Expected to use base decoder, but got {self.decoder}")

        # Postprocessing MLP
        self.mlp = DownscalingMLP(
            in_channels=24 + 9,
            out_channels=1,
            h_channels=64,
            h_layers=2,
        )

    def forward(self, task, film_index):

        x = task["y_context"]
        batch_size = x.shape[0]

        # UNet backbone
        x = self.decoder_lr(x, film_index=task["lt"])

        # Transform to station predictions with setconv
        num_channels = x.shape[3]
        x = x.permute(0, 3, 1, 2)
        assert list(x.shape) == [batch_size, num_channels, 240, 121]
        x_target = task["x_target"]
        num_stations = x_target.shape[2]

        x = self.sc_out(
            x_in=task["x_context"],
            wt=x,
            x_out=[x_target[:, 0, :], x_target[:, 1, :]],
        )
        assert x.shape[0] == batch_size
        assert x.shape[2] == num_stations

        # Concatenate auxiliary data at stations
        alt_target = task["alt_target"]
        assert torch.isnan(alt_target).sum() == 0
        assert alt_target.shape[0] == batch_size
        assert alt_target.shape[2] == num_stations

        aux_time = task["aux_time"].squeeze(-1).repeat(1, 1, num_stations)
        assert aux_time.shape[0] == batch_size
        assert aux_time.shape[2] == num_stations

        x = torch.cat([x, alt_target, x_target, aux_time], dim=1).permute(0, 2, 1)
        assert x.shape[0] == batch_size
        assert x.shape[1] == num_stations

        tmp = self.mlp(x)
        assert list(tmp.shape) == [batch_size, num_stations, 1]
        y_hat = tmp.squeeze(-1)
        assert list(y_hat.shape) == [batch_size, num_stations]
        return y_hat


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(n_channels, n_channels), nn.ReLU())

    def forward(self, x):
        return self.block(x) + x


class DownscalingMLP(nn.Module):
    """
    MLP for handling auxiliary data at station locations
    """

    def __init__(self, in_channels, out_channels, h_channels, h_layers):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h_channels),
            *[ResidualBlock(h_channels) for _ in range(h_layers)],
            nn.Linear(h_channels, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


def find_epoch(results_dir: str) -> int:
    losses = np.load(results_dir + "losses_0.npy")
    epoch = np.argmin(losses)

    return epoch


def load_config(results_dir: str) -> dict:
    with open(results_dir + "config.pkl", "rb") as f:
        return pickle.load(f)


def load_model(results_dir, device, config_dir=None, epoch=None):
    """
    Load the ConvCNPWeatherOnToOff to generate forecasts
    """

    config_dir = config_dir or results_dir
    config = load_config(config_dir)
    epoch = epoch or find_epoch(results_dir)

    model = ConvCNPWeatherOnToOff(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        int_channels=config["int_channels"],
        device=device,
        res=config["res"],
        decoder=config["decoder"],
        mode=config["mode"],
        film=False,
    )

    full_state_dict = torch.load(results_dir + f"epoch_{epoch}", map_location=device)
    state_dict = full_state_dict["model_state_dict"]
    state_dict = {k[7:]: v for k, v in zip(state_dict.keys(), state_dict.values())}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def load_model_e2e(results_dir, lead_time, device):
    """
    Load the full E2E model to generate forecasts
    """

    config = load_config(results_dir)
    epoch = find_epoch(results_dir)

    model = ConvCNPWeatherE2E(
        device="cuda",
        lead_time=lead_time,
        se_model_path="../results/assimilation/all_v4_vit_ps3",
        forecast_model_path="../results/forecast/vit_fnl_randomlt_mlp",
        sf_model_path="../results/downscaling_NEW",
    )

    full_state_dict = torch.load(results_dir + f"epoch_{epoch}", map_location=device)
    state_dict = full_state_dict["model_state_dict"]
    state_dict = {k[7:]: v for k, v in zip(state_dict.keys(), state_dict.values())}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model
