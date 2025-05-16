import sys

import numpy as np
import torch
import torch.nn as nn

from architectures import MLP
from set_convs import convDeepSet
from unet_wrap_padding import *
from vit import *

sys.path.append("../")


class ConvCNPWeather(nn.Module):
    """
    ConvCNP class used for the encoder and processor modules
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        int_channels,
        device,
        res,
        data_path="../data/",
        gnp=False,
        mode="assimilation",
        decoder=None,
        film=False,
        two_frames=False,
    ):

        super().__init__()

        self.device = device

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.int_channels = int_channels
        self.decoder = decoder
        self.int_x = 256
        self.int_y = 128
        self.data_path = data_path
        self.mode = mode
        self.film = film
        self.two_frames = two_frames

        N_SAT_VARS = 2
        N_ICOADS_VARS = 5
        N_HADISD_VARS = 5

        # Load internal grid longitude-latitude locations
        self.era5_x = (
            torch.from_numpy(
                np.load(self.data_path + "grid_lon_lat/era5_x_{}.npy".format(res))
            ).float()
            / 360
        )
        self.era5_y = (
            torch.from_numpy(
                np.load(self.data_path + "grid_lon_lat/era5_y_{}.npy".format(res))
            ).float()
            / 360
        )

        self.int_grid = [
            (torch.linspace(0, 360, 240) / 360).float().cuda(),
            (torch.linspace(-90, 90, 121) / 360).float().cuda(),
        ]

        self.int_grid = [self.int_grid[0].unsqueeze(0), self.int_grid[1].unsqueeze(0)]

        # Create input setconvs for each data modality
        self.ascat_setconvs = convDeepSet(
            0.001, "OnToOn", density_channel=True, device=self.device
        )
        self.amsua_setconvs = [
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(13)
        ]
        self.amsub_setconvs = [
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(12)
        ]
        self.hirs_setconvs = [
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(26)
        ]

        self.sat_setconvs = [
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(N_SAT_VARS)
        ]
        self.hadisd_setconvs = [
            convDeepSet(0.001, "OffToOn", density_channel=True, device=self.device)
            for _ in range(N_HADISD_VARS)
        ]
        self.icoads_setconvs = [
            convDeepSet(0.001, "OffToOn", density_channel=True, device=self.device)
            for _ in range(N_ICOADS_VARS)
        ]
        self.igra_setconvs = [
            convDeepSet(0.001, "OffToOn", density_channel=True, device=self.device)
            for _ in range(24)
        ]

        self.sc_out = convDeepSet(
            0.001, "OnToOff", density_channel=False, device=self.device
        )

        # Instantiate the decoder. Here decoder refers to decoder in a convCNP (i.e the ViT backbone)
        if self.decoder == "vit":
            self.decoder_lr = ViT(
                in_channels=in_channels,
                out_channels=out_channels,
                h_channels=512,
                depth=16,
                patch_size=5,
                per_var_embedding=True,
                img_size=[240, 121],
            )

        elif self.decoder == "vit_assimilation":
            self.decoder_lr = ViT(
                in_channels=256,
                out_channels=out_channels,
                h_channels=512,
                depth=8,
                patch_size=3,
                per_var_embedding=False,
                img_size=[256, 128],
            )

        self.mlp = MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            h_channels=128,
            h_layers=4,
        )
        self.break_next = False

    def encoder_hadisd(self, task, prefix):
        """
        Data preprocessing for HadISD
        """

        encodings = []
        for channel in range(4):
            encodings.append(
                self.hadisd_setconvs[channel](
                    x_in=[
                        task["x_context_hadisd_{}".format(prefix)][channel][:, 0, :],
                        task["x_context_hadisd_{}".format(prefix)][channel][:, 1, :],
                    ],
                    wt=task["y_context_hadisd_{}".format(prefix)][channel].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_sat(self, task, prefix):
        """
        Data preprocessing for Gridsat
        """

        encodings = []
        for channel in range(task["sat_{}".format(prefix)].shape[1]):
            encodings.append(
                self.sat_setconvs[channel](
                    x_in=task["sat_x_{}".format(prefix)],
                    wt=task["sat_{}".format(prefix)][:, channel : channel + 1, ...],
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_icoads(self, task, prefix):
        """
        Data preprocessing for ICOADS
        """

        encodings = []
        for channel in range(5):
            encodings.append(
                self.icoads_setconvs[channel](
                    x_in=task["icoads_x_{}".format(prefix)],
                    wt=task["icoads_{}".format(prefix)][:, channel, :].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)

        return encodings

    def encoder_amsua(self, task, prefix):
        """
        Data preprocessing for AMSU-A
        """

        encodings = []
        task["amsua_{}".format(prefix)][..., -1] = np.nan
        task["amsua_{}".format(prefix)][task["amsua_{}".format(prefix)] == 0] = np.nan
        for i in range(13):
            encodings.append(
                self.amsua_setconvs[i](
                    x_in=task["amsua_x_{}".format(prefix)],
                    wt=task["amsua_{}".format(prefix)].permute(0, 3, 2, 1)[
                        :, i : i + 1, ...
                    ],
                    x_out=self.int_grid,
                )
            )

        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_amsub(self, task, prefix):
        """
        Data preprocessing for AMSU-B
        """

        encodings = []
        task["amsub_{}".format(prefix)][task["amsub_{}".format(prefix)] == 0] = np.nan
        for i in range(12):
            encodings.append(
                self.amsua_setconvs[i](
                    x_in=task["amsub_x_{}".format(prefix)],
                    wt=task["amsub_{}".format(prefix)].permute(0, 3, 1, 2)[
                        :, i : i + 1, ...
                    ],
                    x_out=self.int_grid,
                )
            )

        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_hirs(self, task, prefix):
        """
        Data preprocessing for HIRS
        """

        encodings = []

        task["hirs_{}".format(prefix)][task["hirs_{}".format(prefix)] == 0] = np.nan
        for i in range(26):
            encodings.append(
                self.hirs_setconvs[i](
                    x_in=task["hirs_x_{}".format(prefix)],
                    wt=task["hirs_{}".format(prefix)].permute(0, 3, 1, 2)[
                        :, i : i + 1, ...
                    ],
                    x_out=self.int_grid,
                )
            )

        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_igra(self, task, prefix):
        """
        Data preprocessing for IGRA
        """

        encodings = []
        for channel in range(24):
            encodings.append(
                self.igra_setconvs[channel](
                    x_in=task["igra_x_{}".format(prefix)],
                    wt=task["igra_{}".format(prefix)][:, channel, :].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)

        return encodings

    def encoder_ascat(self, task, prefix):
        """
        Data preprocessing for ASCAT
        """

        task["ascat_{}".format(prefix)][
            torch.isnan(task["ascat_{}".format(prefix)])
        ] = 0
        e = nn.functional.interpolate(
            task["ascat_{}".format(prefix)].permute(0, 3, 1, 2), size=(240, 121)
        )
        e = torch.flip(e, dims=[-1])
        return e

    def encoder_iasi(self, task, prefix):
        """
        Data preprocessing for IASI
        """

        task["iasi_{}".format(prefix)][torch.isnan(task["iasi_{}".format(prefix)])] = 0
        e = nn.functional.interpolate(
            task["iasi_{}".format(prefix)].permute(0, 3, 1, 2), size=(240, 121)
        )
        e = torch.flip(e, dims=[-1])
        return e

    def forward(self, task, film_index):

        # Setup input
        if self.mode == "assimilation":

            self.int_grid = [i.to(task["y_target"].device) for i in self.int_grid]
            elev = nn.functional.interpolate(
                torch.flip(task["era5_elev_current"].permute(0, 1, 3, 2), dims=[2]),
                size=(self.int_grid[0].shape[1], self.int_grid[1].shape[1]),
            )
            elev = torch.flip(task["era5_elev_current"].permute(0, 1, 3, 2), dims=[2])

            if not self.two_frames:
                encodings = [
                    self.encoder_iasi(task, "current"),
                    self.encoder_ascat(task, "current"),
                    self.encoder_hadisd(task, "current"),
                    self.encoder_icoads(task, "current"),
                    self.encoder_sat(task, "current"),
                    self.encoder_amsua(task, "current"),
                    self.encoder_amsub(task, "current"),
                    self.encoder_igra(task, "current"),
                    self.encoder_hirs(task, "current"),
                    elev,
                    task["climatology_current"],
                    torch.ones_like(elev[:, :5, ...])
                    * task["aux_time_current"].unsqueeze(-1).unsqueeze(-1),
                ]
            else:
                # Option to pass two timesteps (t=-1 and t=0) as input
                encodings = [
                    self.encoder_iasi(task, "current"),
                    self.encoder_ascat(task, "current"),
                    self.encoder_hadisd(task, "current"),
                    self.encoder_icoads(task, "current"),
                    self.encoder_sat(task, "current"),
                    self.encoder_amsua(task, "current"),
                    self.encoder_amsub(task, "current"),
                    self.encoder_igra(task, "current"),
                    self.encoder_hirs(task, "current"),
                    self.encoder_iasi(task, "prev"),
                    self.encoder_ascat(task, "prev"),
                    self.encoder_hadisd(task, "prev"),
                    self.encoder_icoads(task, "prev"),
                    self.encoder_sat(task, "prev"),
                    self.encoder_amsua(task, "prev"),
                    self.encoder_amsub(task, "prev"),
                    self.encoder_igra(task, "prev"),
                    self.encoder_hirs(task, "prev"),
                    elev,
                    task["climatology_current"],
                    torch.ones_like(elev[:, :5, ...])
                    * task["aux_time_current"].unsqueeze(-1).unsqueeze(-1),
                ]
            x = torch.cat(encodings, dim=1)

        else:
            x = task["y_context"]

        if x.shape[-1] > x.shape[-2]:
            x = x.permute(0, 1, 3, 2)

        # Run ViT backbone
        if self.decoder == "vit":
            x = self.decoder_lr(x, lead_times=task["lt"])
            x = x.permute(0, 3, 1, 2)
        else:
            x = nn.functional.interpolate(x, size=(256, 128))
            x = self.decoder_lr(x, film_index=(task["lt"] * 0) + 1)

        # Process outputs

        if np.logical_and(
            self.mode == "assimilation", self.decoder == "vit_assimilation"
        ):
            x = nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(240, 121))
            return x.permute(0, 3, 2, 1)

        elif self.mode == "forecast":
            x = nn.functional.interpolate(x, size=(240, 121)).permute(0, 2, 3, 1)
            return x.permute(0, 2, 1, 3)

        return x
