import pickle

import torch
import torch.nn as nn
import numpy as np

from models import ConvCNPWeather
from misc_downscaling_functionality import ConvCNPWeatherOnToOff


class ConvCNPWeatherE2E(nn.Module):
    """
    Complete Aardvark weather model. This chains together the trained encoder,
    processor and decoder modules to create complete forecasts. It can be finetuned
    end to end to optimise predictions for a specific variable and location.
    """

    def __init__(
        self,
        device,
        lead_time,
        se_model_path,
        forecast_model_path,
        sf_model_path,
        return_gridded=False,
        aux_data_path=None,
    ):

        super().__init__()

        self.device = device
        self.lead_time = lead_time
        self.return_gridded = return_gridded

        # Load encoder
        self.se_model = self.load_se_model(se_model_path)

        # Load processor
        self.forecast_model = nn.ModuleList(
            [
                self.load_forecast_model(forecast_model_path, l + 1)
                for l in range(lead_time)
            ]
        )

        # Load decoder
        self.sf_model = self.load_sf_model(sf_model_path, lead_time)
        self.decoder = None

        # Setup normalisation factors
        self.forecast_input_means = (
            self.to_tensor(np.load(aux_data_path + "norm_factors/mean_4u_1.npy"))
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.forecast_input_stds = (
            self.to_tensor(np.load(aux_data_path + "norm_factors/std_4u_1.npy"))
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.forecast_pred_diff_means = (
            self.to_tensor(np.load(aux_data_path + "norm_factors/mean_diff_4u_1.npy"))
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.forecast_pred_diff_stds = (
            self.to_tensor(np.load(aux_data_path + "norm_factors/std_diff_4u_1.npy"))
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def to_tensor(self, arr):
        return torch.from_numpy(arr).float().to(self.device)

    def load_se_model(self, se_model_path):
        """
        Load the trained encoder module
        """

        with open(se_model_path + "/config.pkl", "rb") as handle:
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
            film=bool(0),
        )

        best_epoch = np.argmin(np.load("{}/losses_0.npy".format(se_model_path)))
        state_dict = torch.load(
            "{}/epoch_{}".format(se_model_path, best_epoch),
            map_location="cuda",
        )["model_state_dict"]
        state_dict = {k[7:]: v for k, v in zip(state_dict.keys(), state_dict.values())}
        model.load_state_dict(state_dict)
        model = model.to("cuda")
        return model

    def load_forecast_model(self, forecast_model_path, lead_time):
        """
        Load the trained processor module
        """

        with open(forecast_model_path + "/config.pkl", "rb") as handle:
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
            film=False,
        )
        state_dict = torch.load(
            f"{forecast_model_path}/forecast_{lead_time}/epoch_0",
            map_location="cuda",
        )["model_state_dict"]
        state_dict = {k[7:]: v for k, v in zip(state_dict.keys(), state_dict.values())}
        model.load_state_dict(state_dict)
        model = model.to("cuda")
        return model

    def load_sf_model(self, sf_model_path, lead_time):
        """
        Load the trained decoder module
        """

        with open(sf_model_path + "config.pkl", "rb") as handle:
            config = pickle.load(handle)

        model = ConvCNPWeatherOnToOff(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            int_channels=config["int_channels"],
            device="cuda",
            res=config["res"],
            decoder=config["decoder"],
            mode=config["mode"],
            film=False,
        )

        best_epoch = np.argmin(
            np.load("{}/lt_{}/losses_0.npy".format(sf_model_path, lead_time))
        )
        full_state_dict = torch.load(
            sf_model_path + f"/lt_{lead_time}/epoch_{best_epoch}", map_location="cuda"
        )
        state_dict = full_state_dict["model_state_dict"]
        state_dict = {k[7:]: v for k, v in zip(state_dict.keys(), state_dict.values())}
        model.load_state_dict(state_dict)
        model = model.to("cuda")
        model.eval()

        return model

    def process_se_output(self, task, x):
        """
        Reshape and normalise encoder output for input to processor
        """

        task["forecast"]["y_context"][:, :24, ...] = x.permute(0, 3, 2, 1)
        if self.return_gridded:
            return task, x.permute(0, 3, 2, 1)
        return task

    def process_forecast_output(self, task, x, last=False):
        """
        Reshape and normalise processor output for input to decoder
        """

        base_context = task["forecast"]["y_context"][:, :-11, ...].permute(0, 2, 3, 1)

        base_context = (
            base_context * self.forecast_input_stds + self.forecast_input_means
        ).permute(0, 3, 2, 1)

        x = self.forecast_pred_diff_means + x * self.forecast_pred_diff_stds

        unnorm_x = x + base_context.permute(0, 2, 3, 1)

        forecast = unnorm_x

        x = (unnorm_x - self.forecast_input_means) / self.forecast_input_stds

        task["downscaling"]["y_context"][:, :24, ...] = x.permute(0, 3, 2, 1)
        task["forecast"]["y_context"] = torch.cat(
            [
                x.permute(0, 3, 2, 1),
                task["forecast"]["y_context"][:, 24:, ...],
            ],
            axis=1,
        )
        if self.return_gridded:
            return task, forecast
        return task

    def forward(self, task, film_index=None):
        """
        Produce a forecast
        """

        # Generate initial state
        x = self.se_model(task["assimilation"], film_index=None)
        if self.return_gridded:
            task, initial_state = self.process_se_output(task, x)
        else:
            task = self.process_se_output(task, x)

        # Generate forecast
        for lt in range(self.lead_time):
            x = self.forecast_model[lt](task["forecast"], film_index=None)
            if self.return_gridded:
                task, forecast = self.process_forecast_output(task, x)
            else:
                task = self.process_forecast_output(task, x)

        # Generate station forecast
        x = self.sf_model(task["downscaling"], film_index=None)

        if self.return_gridded:
            initial_state = (
                initial_state.permute(0, 3, 2, 1) * self.forecast_input_stds
                + self.forecast_input_means
            )
            return x, forecast, initial_state
        return x
