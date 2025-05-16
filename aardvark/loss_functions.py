import numpy as np
import torch
import torch.nn as nn


class RmseLoss(nn.Module):
    """
    RMSE loss
    """

    def __init__(self, start_ind=0, end_ind=24):

        super().__init__()
        self.start_ind = start_ind
        self.end_ind = end_ind

    def forward(
        self,
        target,
        output,
        prev_step_output,
        fix_sigma=False,
        unwrap=False,
        expand=False,
    ):

        squared_diff = ((target.to(output.device) - output) ** 2)[
            ..., self.start_ind : self.end_ind
        ]
        return torch.mean(torch.sqrt(torch.nanmean(squared_diff, dim=(1, 2, 3))))


class PressureWeightedRmseLoss(nn.Module):
    """
    Latitude weighted pressure weighted RMSE loss used in training the processor
    """

    def __init__(
        self,
        res,
        era5_mode,
        data_dir,
        aux_data_dir,
        weight_per_variable=False,
    ):
        super().__init__()

        self.weights = torch.from_numpy(
            np.load(aux_data_dir + "lat_weights/weights_lat_{}.npy".format(res)).T[
                np.newaxis, ..., np.newaxis
            ]
        ).float()

        self.weight_per_variable = weight_per_variable
        self.variable_weights = torch.from_numpy(
            np.load(aux_data_dir + "loss_weights.npy")[
                np.newaxis, np.newaxis, np.newaxis, :
            ]
        ).float()

        self.pressure_levels = (
            torch.from_numpy(
                np.load(
                    data_dir + "era5/era5_pressure_levels_{}.npy".format(era5_mode)
                )[np.newaxis, np.newaxis, np.newaxis, :]
            ).float()
            / 1000
        )

    def forward(
        self,
        target,
        output,
        prev_step_output,
        fix_sigma=False,
        unwrap=False,
        expand=False,
    ):

        squared_diff = (target.to(output.device) - output) ** 2

        if not expand:
            weighted_sqared_diff = (
                squared_diff
                * self.weights.to(target.device)
                * self.pressure_levels.to(target.device)
            )
            return torch.mean(
                torch.sqrt(torch.nanmean(weighted_sqared_diff, dim=(1, 2, 3)))
            )

        weighted_sqared_diff = squared_diff * self.weights.to(target.device)

        if self.weight_per_variable:
            weighted_sqared_diff = weighted_sqared_diff * self.variable_weights.to(
                weighted_sqared_diff.device
            )

        return torch.mean(
            torch.sqrt(torch.nanmean(weighted_sqared_diff, dim=(1, 2))), dim=0
        )


class WeightedRmseLoss(nn.Module):
    """
    Latitude weighted RMSE loss
    """

    def __init__(
        self,
        res,
        data_dir,
        aux_data_dir,
        weight_per_variable=False,
        start_ind=0,
        end_ind=24,
    ):

        super().__init__()
        self.start_ind = start_ind
        self.end_ind = end_ind

        self.weights = torch.from_numpy(
            np.load(aux_data_dir + "lat_weights/weights_lat_{}.npy".format(res)).T[
                np.newaxis, ..., np.newaxis
            ]
        ).float()

        self.weight_per_variable = weight_per_variable
        self.variable_weights = torch.from_numpy(
            np.load(aux_data_dir + "loss_weights.npy")[
                np.newaxis, np.newaxis, np.newaxis, start_ind:end_ind
            ]
        ).float()

    def forward(
        self,
        target,
        output,
        prev_step_output,
        fix_sigma=False,
        unwrap=False,
        expand=False,
    ):
        squared_diff = (target.to(output.device) - output) ** 2

        if not expand:
            weighted_sqared_diff = squared_diff * self.weights.to(target.device)

            if self.weight_per_variable:
                weighted_sqared_diff = weighted_sqared_diff * self.variable_weights.to(
                    weighted_sqared_diff.device
                )

            x = torch.nanmean(
                torch.sqrt(torch.nanmean(weighted_sqared_diff, dim=(1, 2, 3)))
            )

            return x

        weighted_sqared_diff = squared_diff * self.weights.to(target.device)
        x = torch.mean(
            torch.sqrt(torch.nanmean(weighted_sqared_diff, dim=(1, 2))), dim=0
        )

        return x
