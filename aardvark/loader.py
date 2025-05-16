import time as timelib
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from loader_utils_new import *
from data_shapes import *


class WeatherDataset(Dataset):
    """
    Base weather dataset class
    """

    def __init__(
        self,
        device,
        hadisd_mode,
        start_date,
        end_date,
        lead_time,
        era5_mode="train",
        res=1,
        filter_dates=None,
        diff=None,
    ):

        super().__init__()

        # Setup
        self.device = device
        self.mode = hadisd_mode
        self.data_path = "path_to_data/"
        self.aux_data_path = "path_to_auxiliary_data/"
        self.start_date = start_date
        self.end_date = end_date
        self.lead_time = lead_time
        self.era5_mode = era5_mode
        self.res = res
        self.filter_dates = filter_dates
        self.diff = diff

        # Date indexing
        self.dates = pd.date_range(start_date, end_date, freq="6H")
        if self.filter_dates == "start":
            self.index = np.array([i for i, d in enumerate(self.dates) if d.month < 7])
        elif self.filter_dates == "end":
            self.index = np.array([i for i, d in enumerate(self.dates) if d.month >= 7])
        else:
            self.index = np.array(range(len(self.dates)))

        # Load the input modalities
        print("Loading IGRA")
        self.load_igra()

        print("Loading AMSU-A")
        self.load_amsua()

        print("Loading AMSU-B")
        self.load_amsub()

        print("Loading ICOADS")
        self.load_icoads()

        print("Loading IASI")
        self.load_iasi()

        print("Loading GEO")
        self.load_sat_data()

        print("Loading HADISD")
        self.load_hadisd(self.mode)

        print("Loading ASCAT")
        self.load_ascat_data()
        self.load_hirs_data()

        # Load the ground truth data for training
        print("Loading ERA5")
        self.era5_sfc = [
            self.load_era5(year)
            for year in range(int(start_date[:4]), int(end_date[:4]) + 1)
        ]

        # Internal grid to longitude latitude correspondence
        self.era5_x = [
            self.to_tensor(
                np.load(self.data_path + "era5/era5_x_{}.npy".format(self.res))
            )
            / LATLON_SCALE_FACTOR,
            self.to_tensor(
                np.load(self.data_path + "era5/era5_y_{}.npy".format(self.res))
            )
            / LATLON_SCALE_FACTOR,
        ]

        # Orography
        self.era5_elev = self.to_tensor(
            np.load(self.data_path + "era5/elev_vars_{}.npy".format(self.res))
        )
        self.era5_elev = torch.flip(self.era5_elev.permute(0, 2, 1), [-1])
        xx, yy = torch.meshgrid(self.era5_x[0], self.era5_x[1])
        self.era5_lonlat = torch.stack([xx, yy])

        # Climatology
        self.climatology = np.memmap(
            self.data_path + "climatology_data.mmap",
            dtype="float32",
            mode="r",
            shape=CLIMATOLOGY_SHAPE,
        )

        # Setup normalisation factors
        if self.diff:
            self.era5_mean_spatial = np.load(
                self.aux_data_path + "era5_spatial_means.npy"
            )[0, ...]
            self.means = np.load(self.aux_data_path + "era5_avdiff_means.npy")[
                :, np.newaxis, np.newaxis, ...
            ]
            self.stds = np.load(self.aux_data_path + "era5_avdiff_stds.npy")[
                :, np.newaxis, np.newaxis, ...
            ]
        else:
            self.means = np.load(
                self.aux_data_path
                + "norm_factors/mean_{}_{}.npy".format(self.era5_mode, self.res)
            )[:, np.newaxis, np.newaxis, ...]
            self.stds = np.load(
                self.aux_data_path
                + "norm_factors/std_{}_{}.npy".format(self.era5_mode, self.res)
            )[:, np.newaxis, np.newaxis, ...]

    def load_icoads(self):
        """
        Load the ICOADS data
        """

        self.icoads_y = np.memmap(
            self.data_path + "icoads/1999_2021_icoads_y.mmap",
            dtype="float32",
            mode="r",
            shape=ICOADS_Y_SHAPE,
        )

        self.icoads_x = (
            np.memmap(
                self.data_path + "icoads/1999_2021_icoads_x.mmap",
                dtype="float32",
                mode="r",
                shape=ICOADS_X_SHAPE,
            )
            / LATLON_SCALE_FACTOR
        )
        self.icoads_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_icoads.npy")
        )
        self.icoads_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_icoads.npy")
        )
        self.icoads_means = self.to_tensor(
            np.nanmean(self.icoads_y[-365 * 4 :, ...], axis=(0, 2))[:, np.newaxis]
        )
        self.icoads_stds = self.to_tensor(
            np.nanstd(self.icoads_y[-365 * 4 :, ...], axis=(0, 2))[:, np.newaxis]
        )
        self.icoads_index_offset = ICOADS_OFFSETS[self.start_date]
        return

    def load_igra(self):
        """
        Load the IGRA data
        """

        self.igra_y = np.memmap(
            self.data_path + "igra/1999_2021_igra_y.mmap",
            dtype="float32",
            mode="r",
            shape=IGRA_Y_SHAPE,
        )

        self.igra_x = np.copy(
            np.memmap(
                self.data_path + "igra/1999_2021_igra_x.mmap",
                dtype="float32",
                mode="r",
                shape=IGRA_X_SHAPE,
            )
        )
        self.igra_x = self.igra_x / LATLON_SCALE_FACTOR

        self.igra_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_igra.npy")
        )
        self.igra_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_igra.npy")
        )

        self.igra_index_offset = IGRA_OFFSETS[self.start_date]

        return

    def load_amsua(self):
        """
        Load the AMSU-A data
        """

        self.amsua_y = np.memmap(
            self.data_path + "amsua/2007_2021_amsua.mmap",
            dtype="float32",
            mode="r",
            shape=AMSUA_Y_SHAPE,
        )
        self.amsua_index_offset = AMSUA_OFFSETS[self.start_date]

        xx = np.linspace(-180, 179, 360, dtype=np.float32)
        xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
        yy = np.linspace(90, -90, 180, dtype=np.float32) / LATLON_SCALE_FACTOR
        self.amsua_x = [xx, yy]

        self.amsua_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_amsua.npy")
        )
        self.amsua_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_amsua.npy")
        )

        return

    def load_amsub(self):
        """
        Load the AMSU-B data
        """

        self.amsub_y = np.memmap(
            self.data_path + "amsub_mhs/2007_2021_amsub.mmap",
            dtype="float32",
            mode="r",
            shape=AMSUB_Y_SHAPE,
        )
        self.amsub_index_offset = AMSUB_OFFSETS[self.start_date]

        xx = np.linspace(0, 359, 360, dtype=np.float32)
        xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
        yy = np.linspace(90, -90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
        self.amsub_x = [xx, yy]

        self.amsub_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_amsub.npy")
        )
        self.amsub_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_amsub.npy")
        )

        return

    def load_ascat_data(self):
        """
        Load the ASCAT data
        """

        self.ascat_y = np.memmap(
            self.data_path + "ascat/2007_2021_ascat.mmap",
            dtype="float32",
            mode="r",
            shape=ASCAT_Y_SHAPE,
        )
        self.ascat_index_offset = ASCAT_OFFSETS[self.start_date]

        xx = np.linspace(0, 359, 360, dtype=np.float32)
        xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
        yy = np.linspace(-90, 90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
        self.ascat_x = [xx, np.copy(yy[::-1])]

        self.ascat_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_ascat.npy")
        )
        self.ascat_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_ascat.npy")
        )

        return

    def load_hirs_data(self):
        """
        Load the HIRS data
        """

        self.hirs_y = np.memmap(
            self.data_path + "hirs/2007_2021_hirs.mmap",
            dtype="float32",
            mode="r",
            shape=HIRS_Y_SHAPE,
        )
        self.hirs_index_offset = ASCAT_OFFSETS[self.start_date]

        xx = np.linspace(0, 359, 360, dtype=np.float32)
        xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
        yy = np.linspace(-90, 90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
        self.hirs_x = [xx, np.copy(yy[::-1])]

        self.hirs_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/hirs_means.npy")
        )
        self.hirs_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/hirs_stds.npy")
        )

        return

    def load_sat_data(self):
        """
        Load the GRIDSAT data
        """

        self.sat_y = np.memmap(
            self.data_path + "gridsat/gridsat_data.mmap",
            dtype="float32",
            mode="r",
            shape=GRIDSAT_Y_SHAPE,
        )

        xx = np.load(self.data_path + "gridsat/sat_x.npy") / LATLON_SCALE_FACTOR
        yy = np.load(self.data_path + "gridsat/sat_y.npy") / LATLON_SCALE_FACTOR
        self.sat_x = [xx, yy]
        self.sat_index_offset = SAT_OFFSETS[self.start_date]

        self.sat_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_sat.npy")
        )
        self.sat_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_sat.npy")
        )

        return

    def load_iasi(self):
        """
        Load the IASI data
        """

        self.iasi = np.memmap(
            self.data_path + "2007_2021_iasi_subset.mmap",
            dtype="float32",
            mode="r",
            shape=IASI_Y_SHAPE,
        )
        self.iasi_index_offset = ASCAT_OFFSETS[self.start_date]

        xx = np.linspace(0, 359, 360, dtype=np.float32)
        xx = ((xx + 360) % 360) / LATLON_SCALE_FACTOR
        yy = np.linspace(-90, 90, 181, dtype=np.float32) / LATLON_SCALE_FACTOR
        self.iasi_x = [xx, np.copy(yy[::-1])]

        self.iasi_means = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/mean_iasi.npy")
        )
        self.iasi_stds = self.to_tensor(
            np.load(self.aux_data_path + "norm_factors/std_iasi.npy")
        )

        return

    def load_hadisd(self, mode):
        """
        Load the HADISD data
        """

        self.hadisd_x = []
        self.hadisd_alt = []
        self.hadisd_y = []
        hadisd_vars = ["tas", "tds", "psl", "u", "v"]
        for var in hadisd_vars:
            lon = lon_to_0_360(
                np.load(
                    self.data_path + "hadisd_processed/{}_lon_{}.npy".format(var, mode)
                )
            )
            lat = np.load(
                self.data_path + "hadisd_processed/{}_lat_{}.npy".format(var, mode)
            )
            alt = np.load(
                self.data_path + "hadisd_processed/{}_alt_{}.npy".format(var, mode)
            )

            vals = np.memmap(
                self.data_path
                + "hadisd_processed/{}_vals_{}.memmap".format(var, self.mode),
                dtype="float32",
                mode="r",
                shape=get_hadisd_shape(mode),
            )

            self.hadisd_x.append(np.stack([lon, lat], axis=-1) / LATLON_SCALE_FACTOR)
            self.hadisd_alt.append(alt)
            self.hadisd_y.append(vals)

        self.hadisd_index_offset = HADISD_OFFSETS[self.start_date]

        self.hadisd_means = [
            self.to_tensor(
                np.load(
                    self.aux_data_path + "norm_factors/mean_hadisd_{}.npy".format(var)
                )
            )
            for var in hadisd_vars
        ]
        self.hadisd_stds = [
            self.to_tensor(
                np.load(
                    self.aux_data_path + "norm_factors/std_hadisd_{}.npy".format(var)
                )
            )
            for var in hadisd_vars
        ]

        return

    def load_era5(self, year):
        """
        Load the ERA5 training data
        """

        if year % 4 == 0:
            d = 366 * 4
        else:
            d = 365 * 4

        if self.era5_mode == "sfc":
            levels = 4
        elif self.era5_mode == "13u":
            levels = 69
        else:
            levels = 24

        if self.res == 1:
            x = 240
            y = 121
        elif self.res == 5:
            x = 64
            y = 32
        mmap = np.memmap(
            self.data_path
            + "/era5/era5_{}_{}_6_{}.memmap".format(self.era5_mode, self.res, year),
            dtype="float32",
            mode="r",
            shape=(d, levels, x, y),
        )
        return mmap

    def norm_era5(self, x):

        x = (x - self.means) / self.stds
        return x

    def unnorm_era5(self, x):

        x = x * self.stds + self.means
        return x

    def norm_data(self, x, means, stds):
        return (x - means) / stds

    def norm_hadisd(self, x):
        for i in range(5):
            x[i] = (x[i] - self.hadisd_means[i]) / self.hadisd_stds[i]
        return x

    def __len__(self):
        return self.index.shape[0] - 1 - 1

    def to_tensor(self, arr):
        return torch.from_numpy(arr).float().to(self.device)

    def get_time_aux(self, current_date):
        """
        Return the auxiliary temporal channels given a date
        """

        doy = current_date.dayofyear
        year = (current_date.year - 2007) / 15
        time_of_day = current_date.hour
        return np.array(
            [
                np.cos(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.sin(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.cos(np.pi * 2 * time_of_day / 24),
                np.sin(np.pi * 2 * time_of_day / 24),
                year,
            ]
        )


class WeatherDatasetAssimilation(WeatherDataset):
    """
    Encoder training loader
    """

    def __init__(
        self,
        device,
        hadisd_mode,
        start_date,
        end_date,
        lead_time,
        era5_mode="sfc",
        res=1,
        filter_dates=None,
        var_start=0,
        var_end=24,
        diff=False,
        two_frames=False,
    ):

        super().__init__(
            device,
            hadisd_mode,
            start_date,
            end_date,
            lead_time,
            era5_mode,
            res=res,
            filter_dates=filter_dates,
            diff=diff,
        )

        # Setup

        self.var_start = var_start
        self.var_end = var_end
        self.diff = diff
        self.two_frames = two_frames

    def load_era5_time(self, index):
        """
        ERA5 ground truth data loading
        """

        date = self.dates[index]
        year = date.year
        hour = date.hour
        doy = (date.dayofyear - 1) * 4 + (hour // 6)

        era5 = self.era5_sfc[year - int(self.start_date[:4])][doy, ...]
        era5 = np.copy(era5)
        if self.diff:
            era5 = era5 - self.era5_mean_spatial
        era5 = self.norm_era5(era5[np.newaxis, ...])[0, ...]
        return era5

    def load_year_end(self, year, doy):
        data_1 = self.era5_sfc[year - int(self.start_date[:4])][doy : doy + 1, ...]
        missing = self.lead_time - data_1.shape[0] + 1
        data_2 = self.era5_sfc[year - int(self.start_date[:4]) + 1][:missing, ...]
        data = np.concatenate([data_1, data_2])
        return data

    def load_era5_slice(self, index):
        """
        ERA5 ground truth data loading
        """

        date = self.dates[index]
        year = date.year
        doy = (date.dayofyear - 1) * 4

        next_date = self.dates[index + 1]
        next_year = next_date.year

        if next_year != year:
            era5 = self.load_year_end(year, doy)
        else:
            era5 = self.era5_sfc[year - int(self.start_date[:4])][doy : doy + 1, ...]

        era5 = self.norm_era5(np.copy(era5))
        return era5

    def __getitem__(self, index):

        if self.two_frames:
            # Case 1: loading t=0 and t=-1
            index = index + 1
            current = self.get_index(index, "current")
            prev = self.get_index(index - 1, "prev")
            current["y_target"] = current["y_target_current"]

            return {**current, **prev}
        else:
            # Case 2: loading t=0
            current = self.get_index(index, "current")
            current["y_target"] = current["y_target_current"]

            return {**current}

    def unnorm_pred(self, x):
        dev = x.device
        x = x.detach().cpu().numpy()

        x = (
            x
            * self.stds[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
            + self.means[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
        )
        if bool(self.diff):
            x = (
                x
                + self.era5_mean_spatial[np.newaxis, ...].transpose(0, 3, 2, 1)[
                    ..., self.var_start : self.var_end
                ]
            )
        return torch.from_numpy(x).float().to(dev)

    def get_index(self, index, prefix):
        """
        Load data for the relevant index respecting different offsets depending on the modality
        """

        index = self.index[index]
        date = self.dates[index]

        # ICOADS
        icoads_x = self.icoads_x[index + self.icoads_index_offset, ...]
        icoads_y = self.icoads_y[index + self.icoads_index_offset, ...]
        icoads_x = [icoads_x[0, :], icoads_x[1, :]]
        icoads_x = [self.to_tensor(i) for i in icoads_x]
        icoads_y = self.to_tensor(icoads_y)
        icoads_y = self.norm_data(icoads_y, self.icoads_means, self.icoads_stds)

        # GRIDSAT
        sat_y = self.sat_y[index + self.sat_index_offset, ...]
        sat_x = [self.to_tensor(i) for i in self.sat_x]
        sat_y = self.to_tensor(sat_y)
        sat_y = self.norm_data(sat_y, self.sat_means, self.sat_stds)

        # AMSU-A
        amsua_y = self.to_tensor(self.amsua_y[index + self.amsua_index_offset, ...])
        amsua_y[amsua_y < -998] = torch.nan
        amsua_x = [self.to_tensor(i) for i in self.amsua_x]
        amsua_y[amsua_y < -998] = np.nan
        amsua_y = self.norm_data(amsua_y, self.amsua_means, self.amsua_stds)

        # AMSU-B
        amsub_y = self.to_tensor(self.amsub_y[index + self.amsub_index_offset, ...])
        amsub_y[amsub_y < -998] = torch.nan
        amsub_x = [self.to_tensor(i) for i in self.amsub_x]
        amsub_y[amsub_y < -998] = np.nan
        amsub_y = self.norm_data(amsub_y, self.amsub_means, self.amsub_stds)

        # IASI
        iasi_y = self.to_tensor(self.iasi[index + self.iasi_index_offset, ...])
        iasi_x = [self.to_tensor(i) for i in self.iasi_x]
        iasi_y = self.norm_data(iasi_y, self.iasi_means, self.iasi_stds)

        # IGRA
        igra_y = self.to_tensor(self.igra_y[index + self.igra_index_offset, ...])
        igra_x = [self.igra_x[:, 0], self.igra_x[:, 1]]
        igra_x = [self.to_tensor(i) for i in igra_x]
        igra_y = self.norm_data(igra_y, self.igra_means, self.igra_stds)

        # ASCAT
        ascat_y = self.to_tensor(self.ascat_y[index + self.ascat_index_offset, ...])
        ascat_x = [self.to_tensor(i) for i in self.ascat_x]
        ascat_y[..., 4][ascat_y[..., 4] < -990] = np.nan
        ascat_y = self.norm_data(ascat_y, self.ascat_means, self.ascat_stds)

        # HIRS
        hirs_y = self.to_tensor(self.hirs_y[index + self.hirs_index_offset, ...])
        hirs_y[hirs_y < -998] = np.nan
        hirs_x = [self.to_tensor(i) for i in self.hirs_x]
        hirs_y = self.norm_data(hirs_y, self.hirs_means, self.hirs_stds)

        # HadISD
        x_context_hadisd = self.hadisd_x
        y_context_hadisd = [
            i[index + self.hadisd_index_offset, :] for i in self.hadisd_y
        ]
        x_context_hadisd = [self.to_tensor(i).permute(1, 0) for i in x_context_hadisd]
        y_context_hadisd = [self.to_tensor(i) for i in y_context_hadisd]
        y_context_hadisd = self.norm_hadisd(y_context_hadisd)

        # ERA5
        era5 = self.to_tensor(self.load_era5_time(index))
        era5_target = era5.permute(2, 1, 0)
        era5_x = self.era5_x

        # AUxiliary variables
        aux_time = self.to_tensor(self.get_time_aux(date))
        climatology = self.climatology[date.hour // 6, date.dayofyear - 1, ...]

        task = {
            "x_context_hadisd_{}".format(prefix): x_context_hadisd,
            "y_context_hadisd_{}".format(prefix): y_context_hadisd,
            "climatology_{}".format(prefix): self.to_tensor(climatology),
            "sat_x_{}".format(prefix): sat_x,
            "sat_{}".format(prefix): sat_y,
            "icoads_x_{}".format(prefix): icoads_x,
            "icoads_{}".format(prefix): icoads_y,
            "igra_x_{}".format(prefix): igra_x,
            "igra_{}".format(prefix): igra_y,
            "amsua_{}".format(prefix): amsua_y,
            "amsua_x_{}".format(prefix): amsua_x,
            "amsub_{}".format(prefix): amsub_y,
            "amsub_x_{}".format(prefix): amsub_x,
            "iasi_{}".format(prefix): iasi_y,
            "iasi_x_{}".format(prefix): iasi_x,
            "ascat_{}".format(prefix): ascat_y,
            "ascat_x_{}".format(prefix): ascat_x,
            "hirs_{}".format(prefix): hirs_y,
            "hirs_x_{}".format(prefix): hirs_x,
            "y_target_{}".format(prefix): era5_target[
                ..., self.var_start : self.var_end
            ],
            "era5_x_{}".format(prefix): era5_x,
            "era5_elev_{}".format(prefix): self.era5_elev,
            "era5_lonlat_{}".format(prefix): self.era5_lonlat,
            "aux_time_{}".format(prefix): aux_time,
            "lt": torch.Tensor([self.var_start]),
        }

        return task


class HadISDDataset(Dataset):
    """
    HadISD dataset for decoder training
    """

    def __init__(self, var, mode, device, start_date, end_date):
        super().__init__()

        # Setup
        if not mode in ["train", "val", "test"]:
            raise Exception(f"mode is {mode}. Must be train, val, or test.")

        self.var = var
        self.mode = mode
        self.start_date = start_date
        self.device = device
        dates = pd.date_range(start_date, end_date, freq="6H")
        self.index = np.array(range(len(dates)))

        # Load the hadISD data
        self.load_hadisd()

    def load_hadisd(self):
        """
        Load the raw HadISD data
        """

        data_path = "path_to_data/"
        aux_data_path = "path_to_auxiliary_data/"
        var = self.var
        mode = self.mode

        vals = np.memmap(
            data_path + f"hadisd_processed/{var}_vals_{mode}.memmap",
            dtype="float32",
            mode="r",
            shape=get_hadisd_shape(mode),
        )

        lon = lon_to_0_360(
            np.load(data_path + f"hadisd_processed/{var}_lon_{mode}.npy")
        )
        lat = np.load(data_path + f"hadisd_processed/{var}_lat_{mode}.npy")
        self.hadisd_x = np.stack([lon, lat], axis=-1) / LATLON_SCALE_FACTOR
        self.hadisd_alt = np.load(
            data_path + f"hadisd_processed/{var}_alt_{mode}_final.npy"
        )
        self.hadisd_y = vals

        self.hadisd_index_offset = HADISD_OFFSETS[self.start_date]
        self.hadisd_means = self.to_tensor(
            np.load(aux_data_path + f"norm_factors/mean_hadisd_{var}.npy")
        )
        self.hadisd_stds = self.to_tensor(
            np.load(aux_data_path + f"norm_factors/std_hadisd_{var}.npy")
        )
        return

    def norm_hadisd(self, x):
        return (x - self.hadisd_means) / self.hadisd_stds

    def unnorm_pred(self, x):
        return self.hadisd_means + self.hadisd_stds * x

    def __len__(self):
        return self.index.shape[0] - 2

    def to_tensor(self, arr):
        return torch.from_numpy(np.array(arr)).float().to(self.device)

    def __getitem__(self, index):
        index = self.index[index]

        # Get longitude-latitude locations
        x_target = self.to_tensor(self.hadisd_x).permute(1, 0)

        # Get altitude and normalise
        m_alt = np.expand_dims(np.load("path_to_mean_alt.npy"), 1)
        s_alt = np.expand_dims(np.load("path_to_std_alt.npy"), 1)
        alt_target = self.to_tensor((self.hadisd_alt - m_alt) / s_alt)[:, :]

        # Get observations
        y_target = self.norm_hadisd(
            self.to_tensor(self.hadisd_y[index + self.hadisd_index_offset, :])
        )

        assert x_target.shape[0] == 2
        n_stations = x_target.shape[1]
        assert alt_target.shape[1] == n_stations
        assert y_target.shape[0] == n_stations

        return {"x": x_target, "altitude": alt_target, "y": y_target}


class AardvarkICDataset(Dataset):
    """
    Helper dataset to handle initial condition loading for decoder training
    """

    def __init__(self, device, start_date, end_date, lead_time=0):
        super().__init__()

        # Setup

        if lead_time == 0:
            # If leadtime is 0 load the output of the encoder
            if start_date == "2007-01-02" and end_date == "2017-12-31":
                ic_fname = "ic_train.mmap"
            elif start_date == "2019-01-01" and end_date == "2019-12-01":
                ic_fname = "ic_val.mmap"
            elif start_date == "2018-01-01" and end_date == "2018-12-31":
                ic_fname = "ic_test.mmap"
            else:
                print((start_date, end_date))
                raise Exception("Invalid start and end date")

            dates = pd.date_range(start_date, end_date, freq="6H")

            self.data = np.memmap(
                "path_to_encoder_predictions/" + ic_fname,
                dtype="float32",
                mode="r",
                shape=(len(dates), 121, 240, 24),  # shape of the output
            )
        else:
            # if leadtime >0 load the forecast generated from the encoder prediction
            if start_date == "2007-01-02" and end_date == "2017-12-31":
                ic_fname = f"ic_train_{lead_time}.mmap"

            elif start_date == "2019-01-01" and end_date == "2019-12-01":
                ic_fname = f"ic_val_{lead_time}.mmap"
            elif start_date == "2018-01-01" and end_date == "2018-12-31":
                ic_fname = f"ic_test_{lead_time}.mmap"
            else:
                print((start_date, end_date))
                raise Exception("Invalid start and end date.")

            dates = pd.date_range(start_date, end_date, freq="6H")[(lead_time) * 4 :]
            ic_shape = (len(dates), 121, 240, 24)

            self.data = np.memmap(
                self.data_path + "forecast_finetune/" + ic_fname,
                dtype="float32",
                mode="r",
                shape=ic_shape,
            )

        self.device = device

        # Normalisation
        aux_data_path = "path_to_auxiliary_data/"
        mean_factors_path = aux_data_path + f"norm_factors/mean_4u_1.npy"
        std_factors_path = aux_data_path + f"norm_factors/std_4u_1.npy"
        self.means = np.load(mean_factors_path)[:, np.newaxis, np.newaxis, ...]
        self.stds = np.load(std_factors_path)[:, np.newaxis, np.newaxis, ...]

    def __getitem__(self, index):
        # Load Aardvark prediction and normalise
        data_raw = np.transpose(np.copy(self.data[index, :, :, :]), (2, 1, 0))
        data = (data_raw - self.means) / self.stds
        return torch.from_numpy(data).to(self.device)


class WeatherDatasetDownscaling(Dataset):
    """
    Main decoder training dataset. Uses AardvarkICDataset and HadISDDataset to
    handle processor output and station data
    """

    def __init__(
        self,
        device,
        hadisd_mode,
        start_date,
        end_date,
        context_mode,
        era5_mode="sfc",
        res=1,
        hadisd_var="tas",
        lead_time=1,
    ):
        # The context mode determines whether we make use of ERA5 or our own ICs.
        if not context_mode in ["era5", "aardvark"]:
            raise Exception(
                f"context_mode must be era5 or aardvark, got {context_mode}"
            )

        super().__init__()

        # Setup
        self.lead_time = lead_time

        self.device = device
        self.data_path = "path_to_data/"
        self.aux_data_path = "path_to_auxiliary_data/"
        self.start_date = start_date
        self.end_date = end_date
        self.era5_mode = era5_mode
        self.res = res
        self.context_mode = context_mode

        self.dates = pd.date_range(start_date, end_date, freq="6H")
        self.index = np.array(range(len(self.dates)))

        # Load ERA5 data for pre-training
        self.era5_sfc = [
            self.load_era5(year)
            for year in range(int(start_date[:4]), int(end_date[:4]) + 1)
        ]

        raw_era5_lon = np.load(self.data_path + f"era5/era5_x_{res}.npy")
        raw_era5_lat = np.load(self.data_path + f"era5/era5_y_{res}.npy")
        self.era5_x = [
            self.to_tensor(raw_era5_lon) / LATLON_SCALE_FACTOR,
            self.to_tensor(raw_era5_lat) / LATLON_SCALE_FACTOR,
        ]

        # Load orography
        elev_path = self.data_path + f"era5/elev_vars_{res}.npy"
        self.era5_elev = self.to_tensor(np.load(elev_path)).permute(0, 2, 1)

        # Normalisation
        mean_factors_path = (
            self.aux_data_path + f"norm_factors/mean_{era5_mode}_{res}.npy"
        )
        std_factors_path = (
            self.aux_data_path + f"norm_factors/std_{era5_mode}_{res}.npy"
        )
        self.means = np.load(mean_factors_path)[:, np.newaxis, np.newaxis, ...]
        self.stds = np.load(std_factors_path)[:, np.newaxis, np.newaxis, ...]

        # HadISD data
        self.hadisd_data = HadISDDataset(
            var=hadisd_var,
            mode=hadisd_mode,
            device=device,
            start_date=start_date,
            end_date=end_date,
        )

        if context_mode == "aardvark":
            # Load the Aardvark encoder predictions
            self.aardvark_data = AardvarkICDataset(
                device, start_date, end_date, lead_time
            )

    def load_era5(self, year):
        """
        Load the raw ERA5 data
        """

        if year % 4 == 0:
            d = 366 * 4
        else:
            d = 365 * 4

        if self.era5_mode == "sfc":
            levels = 4
        elif self.era5_mode == "13u":
            levels = 69
        else:
            levels = 24

        if self.res == 1:
            x = 240
            y = 121
        elif self.res == 5:
            x = 64
            y = 32
        mmap = np.memmap(
            self.data_path
            + "era5/era5_{}_{}_6_{}.memmap".format(self.era5_mode, self.res, year),
            dtype="float32",
            mode="r",
            shape=(d, levels, x, y),
        )
        return mmap

    def norm_era5(self, x):
        x = (x - self.means) / self.stds
        return x

    def unnorm_era5(self, x):
        x = x * self.stds + self.means
        return x

    def unnorm_pred(self, x):
        return self.hadisd_data.unnorm_pred(x)

    def norm_data(self, x, means, stds):
        return (x - means) / stds

    def __len__(self):
        return self.index.shape[0] - (self.lead_time) * 4

    def to_tensor(self, arr):
        return torch.from_numpy(np.array(arr)).float().to(self.device)

    def get_time_aux(self, current_date):
        """
        Get auxiliary time variables for a given date
        """

        doy = current_date.dayofyear
        year = (current_date.year - 2007) / 15
        time_of_day = current_date.hour
        return np.array(
            [
                np.cos(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.sin(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.cos(np.pi * 2 * time_of_day / 24),
                np.sin(np.pi * 2 * time_of_day / 24),
                year,
            ]
        )

    def load_era5_time(self, index):
        """
        Load ERA5 training data
        """

        date = self.dates[index]
        year = date.year
        hour = date.hour
        doy = (date.dayofyear - 1) * 4 + (hour // 6)

        era5 = self.era5_sfc[year - int(self.start_date[:4])][doy, ...]
        era5 = np.copy(era5)
        era5 = self.norm_era5(era5[np.newaxis, ...])[0, ...]
        return era5

    def load_year_end(self, year, doy):
        data_1 = self.era5_sfc[year - int(self.start_date[:4])][doy : doy + 1, ...]
        missing = self.lead_time - data_1.shape[0] + 1
        data_2 = self.era5_sfc[year - int(self.start_date[:4]) + 1][:missing, ...]
        data = np.concatenate([data_1, data_2])
        return data

    def __getitem__(self, index):

        index = self.index[index]
        date = self.dates[index + 4 * self.lead_time]

        # Get HadISD data
        hadisd_slice = self.hadisd_data[index + 4 * self.lead_time]

        # Get lon-lat
        x_context = self.era5_x
        n_lon = x_context[0].shape[0]
        n_lat = x_context[1].shape[0]

        # Get auxiliary time
        aux_time = torch.reshape(self.to_tensor(self.get_time_aux(date)), (-1, 1, 1))

        # Load the context (either aardvark or ERA5 for use in pre-training)
        if self.context_mode == "era5":
            y_context_obs = self.to_tensor(
                self.load_era5_time(index + 4 * self.lead_time)
            )

        elif self.context_mode == "aardvark":
            y_context_obs = self.aardvark_data[index]

        else:
            raise Exception

        y_context = torch.cat(
            [
                y_context_obs,
                self.era5_elev.permute(0, 2, 1),
                aux_time.repeat(1, n_lon, n_lat),
            ]
        )

        assert y_context.shape[1] == n_lon
        assert y_context.shape[2] == n_lat

        x = hadisd_slice["x"]
        alt = hadisd_slice["altitude"]
        y = hadisd_slice["y"]

        return {
            "x_target": x,
            "alt_target": alt,
            "y_target": y,
            "y_context": y_context,
            "x_context": x_context,
            "aux_time": aux_time,
            "lt": torch.Tensor([0]),
        }


class ForecasterDatasetDownscaling(Dataset):
    """
    Dataset to generate decoder predictions from pre-saved Aardvark forecasts
    """

    def __init__(
        self,
        start_date,
        end_date,
        lead_time,
        hadisd_var,
        mode,
        device,
        forecast_path,
        region="global",
    ):
        super().__init__()

        # Setup

        if not mode in ["train", "val", "test"]:
            raise Exception(f"Mode is {mode}. Must be either train, val, or test")

        self.device = device
        self.start_date = start_date
        self.end_date = end_date
        self.lead_time = lead_time
        self.mode = mode
        self.offset = np.timedelta64(lead_time, "D").astype("timedelta64[ns]")

        self.dates = pd.date_range(start_date, end_date, freq="6H")[:-30]

        # Normalisation
        aux_data_path = "auxiliary_data_path/"
        self.means = np.load(aux_data_path + "norm_factors/mean_4u_1.npy")
        self.stds = np.load(aux_data_path + "norm_factors/std_4u_1.npy")

        # Load auxiliary data
        self.load_npy_file()
        data_path = "data_path/"
        res = "1"
        raw_era5_lon = np.load(data_path + f"era5/era5_x_{res}.npy")
        raw_era5_lat = np.load(data_path + f"era5/era5_y_{res}.npy")
        self.era5_x = [
            self.to_tensor(raw_era5_lon) / LATLON_SCALE_FACTOR,
            self.to_tensor(raw_era5_lat) / LATLON_SCALE_FACTOR,
        ]
        elev_path = data_path + f"era5/elev_vars_{res}.npy"
        self.era5_elev = self.to_tensor(np.load(elev_path)).permute(0, 2, 1)

        # Load hadISD
        self.hadisd_data = HadISDDataset(
            var=hadisd_var,
            mode="train",
            device=device,
            start_date=start_date,
            end_date=end_date,
        )

        # Subset to region
        self.region = region
        if self.region != "global":
            self.mask = np.load(
                self.data_path + f"hadisd_processed/tas_mask_train_{region}.npy"
            )

    def date_range(self):
        return np.arange(
            start=np.datetime64(self.start_date).astype("datetime64[ns]"),
            stop=np.datetime64(self.end_date).astype("datetime64[ns]"),
            step=np.timedelta64(1, "D").astype("timedelta64[ns]"),
        )

    def load_npy_file(self):
        """
        Load the pre-saved Aardvark forecasts
        """

        dates = pd.date_range(self.start_date, self.end_date, freq="6H")

        if self.mode == "train":
            dates = dates[:-40]  # Need 10 day offset at end of year

        self.Y_context = np.memmap(
            "path_to_forecasts/forecast_{}.mmap".format(self.mode),
            dtype="float32",
            mode="r",
            shape=(len(dates), 121, 240, 24, 11),
        )

        return

    def norm_era5(self, x):
        return (x - self.means) / self.stds

    def norm_hadisd(self, x):
        return self.hadisd_data.norm_hadisd(x)

    def unnorm_pred(self, x):
        return self.hadisd_data.unnorm_pred(x)

    def __len__(self):
        return len(self.dates) - 40  # Need 10 day offset at end of year

    def to_tensor(self, arr):
        return torch.from_numpy(np.array(arr)).float().to(self.device)

    def get_time_aux(self, index):
        """
        Get the auxiliary time variables
        """

        current_date = (self.dates + self.offset)[index]
        doy = current_date.dayofyear
        year = (current_date.year - 2007) / 15
        time_of_day = current_date.hour
        return np.array(
            [
                np.cos(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.sin(np.pi * 2 * doy / DAYS_IN_YEAR),
                np.cos(np.pi * 2 * time_of_day / 24),
                np.sin(np.pi * 2 * time_of_day / 24),
                year,
            ]
        )

    def __getitem__(self, index):

        # Load target data
        hadisd_slice = self.hadisd_data[index + 4 * self.lead_time]

        x_context = self.era5_x
        n_lon = x_context[0].shape[0]
        n_lat = x_context[1].shape[0]

        # Load auxiliary time
        aux_time = torch.reshape(self.to_tensor(self.get_time_aux(index)), (-1, 1, 1))

        # Load input
        y_context = self.norm_era5(self.Y_context[index, ..., self.lead_time])
        y_context = torch.cat(
            [
                self.to_tensor(y_context).permute(2, 1, 0),
                self.era5_elev.permute(0, 2, 1),
                aux_time.repeat(1, n_lon, n_lat),
            ]
        )

        assert y_context.shape[1] == n_lon
        assert y_context.shape[2] == n_lat

        # Handle region masking
        if self.region != "global":
            hadisd_slice["y"][self.mask] = np.nan

        return {
            "x_target": hadisd_slice["x"],
            "alt_target": hadisd_slice["altitude"],
            "y_target": hadisd_slice["y"],
            "y_context": y_context,
            "x_context": x_context,
            "aux_time": aux_time,
            "lt": torch.Tensor([0]),
        }


class ForecastLoader(Dataset):
    """
    Loader for finetuning the processor module
    """

    def __init__(
        self,
        device,
        mode,
        lead_time,
        era5_mode="sfc",
        res=5,
        frequency=24,
        norm=True,
        diff=False,
        rollout=False,
        random_lt=False,
        u_only=False,
        ic_path=None,
        finetune_step=None,
        finetune_eval_every=100,
        eval_steps=False,
    ):

        super().__init__()

        # Setup
        self.device = device
        self.mode = mode
        self.data_path = "data_path/"

        self.lead_time = lead_time
        self.era5_mode = era5_mode
        self.res = res
        self.frequency = frequency
        self.norm = norm
        self.diff = diff
        self.rollout = rollout
        self.random_lt = random_lt
        self.u_only = u_only
        self.ic_path = ic_path

        self.finetune_step = finetune_step
        self.finetune_eval_every = finetune_eval_every
        self.eval_steps = eval_steps

        if self.frequency == 6:
            self.lead_time = self.lead_time * 4
            freq = "6H"

        else:
            freq = "1D"

        if self.mode == "train":
            self.dates = pd.date_range("1979-01-01", "2017-12-31", freq=freq)
        elif self.mode == "tune":
            self.dates = pd.date_range("2018-01-01", "2018-12-31", freq=freq)
        elif self.mode == "test":
            self.dates = pd.date_range("2018-01-01", "2018-12-31", freq=freq)
        elif self.mode == "val":
            self.dates = pd.date_range("2019-01-01", "2019-12-31", freq=freq)

        # Load the predictions from the previous leadtime to be the new context set
        if self.finetune_step is not None:

            if self.mode == "train":
                self.dates = pd.date_range("2007-01-02", "2017-12-31", freq=freq)
                ic_shape = (
                    len(self.dates) - max(0, (self.finetune_step - 1) * 4),
                    121,
                    240,
                    24,
                )
            elif self.mode == "val":
                self.dates = pd.date_range("2019-01-01", "2019-12-31", freq=freq)
                ic_shape = (
                    len(self.dates) - max(0, (self.finetune_step - 1) * 4),
                    121,
                    240,
                    24,
                )
            elif self.mode == "test":
                self.dates = pd.date_range("2018-01-01", "2018-12-31", freq=freq)
                ic_shape = (
                    len(self.dates) - max(0, (self.finetune_step - 1) * 4),
                    121,
                    240,
                    24,
                )

            if self.finetune_step > 1:
                print(ic_shape)
                self.ic = np.memmap(
                    self.ic_path
                    + "ic_{}_{}.mmap".format(self.mode, self.finetune_step - 1),
                    dtype="float32",
                    mode="r",
                    shape=ic_shape,
                )
            elif self.ic_path is not None:

                self.ic = np.memmap(
                    self.ic_path + "ic_{}.mmap".format(self.mode),
                    dtype="float32",
                    mode="r",
                    shape=ic_shape,
                )

        elif self.ic_path is not None:
            if self.mode == "train":
                self.dates = pd.date_range("2007-01-02", "2017-12-31", freq=freq)
            ic_shape = (len(self.dates), 121, 240, 24)

            self.ic = np.memmap(
                self.ic_path + "/ic_{}.mmap".format(self.mode),
                dtype="float32",
                mode="r",
                shape=ic_shape,
            )

        # Orography
        self.era5_elev = np.float32(
            np.load(self.data_path + "era5/elev_vars_{}.npy".format(res))
        )
        elev_mean = self.era5_elev.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
        elev_std = self.era5_elev.std(axis=(1, 2))[:, np.newaxis, np.newaxis]
        self.era5_elev = (self.era5_elev - elev_mean) / elev_std

        # ERA5 ground truth data for training
        self.era5_sfc = [
            self.load_era5(year)
            for year in range(int(self.dates[0].year), int(self.dates[-1].year) + 1)
        ]

        # Noramalisation factors
        self.means = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/mean_{}_{}.npy".format(self.era5_mode, self.res)
                )
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        self.stds = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/std_{}_{}.npy".format(self.era5_mode, self.res)
                )
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        self.diff_means = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/mean_diff_{}_{}.npy".format(
                        self.era5_mode, self.res
                    )
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.diff_stds = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/std_diff_{}_{}.npy".format(self.era5_mode, self.res)
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.diff_means_1 = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/mean_diff_{}_{}_6h.npy".format(
                        self.era5_mode, self.res
                    )
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.diff_stds_1 = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/std_diff_{}_{}_6h.npy".format(
                        self.era5_mode, self.res
                    )
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.diff_means_2 = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/mean_diff_{}_{}_12h.npy".format(
                        self.era5_mode, self.res
                    )
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.diff_stds_2 = (
            self.to_tensor(
                np.load(
                    self.data_path
                    + "norm_factors/std_diff_{}_{}_12h.npy".format(
                        self.era5_mode, self.res
                    )
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.means_dict = {
            0: self.diff_means,
            2: self.diff_means_2,
            3: self.diff_means_1,
        }

        self.stds_dict = {0: self.diff_stds, 2: self.diff_stds_2, 3: self.diff_stds_1}

    def __len__(self):
        if np.logical_and(self.eval_steps, self.mode == "train"):
            return self.finetune_eval_every * 12 * 4

        return self.dates.shape[0] - self.lead_time

    def to_tensor(self, arr):

        return torch.from_numpy(arr).float().to(self.device)

    def norm_era5(self, x):
        x = (x - self.means) / self.stds
        return x

    def norm_era5_tendency(self, x, lt_offset):

        x = (x - self.means_dict[lt_offset]) / self.stds_dict[lt_offset]
        return x

    def unnorm_pred(self, x):
        x = x * self.diff_stds.unsqueeze(0) + self.diff_means.unsqueeze(0)
        return x

    def unnorm_base_context(self, x):
        x = x * self.stds.unsqueeze(0) + self.means.unsqueeze(0)
        return x

    def load_era5(self, year):
        """
        Load ERA5 data for training
        """

        if year % 4 == 0:
            d = 366
        else:
            d = 365

        if self.frequency == 6:
            d = d * 4

        if self.era5_mode == "sfc":
            levels = 4
        elif self.era5_mode == "13u":
            levels = 69
        else:
            levels = 24

        if self.res == 1:
            x = 240
            y = 121
        elif self.res == 5:
            x = 64
            y = 32

        mmap = np.memmap(
            self.data_path
            + "era5/era5_{}_{}_{}_{}.memmap".format(
                self.era5_mode, self.res, self.frequency, year
            ),
            dtype="float32",
            mode="r",
            shape=(d, levels, x, y),
        )
        return mmap

    def load_era5_time(self, index):
        """
        Load ERA5 data for training
        """
        date = self.dates[index]
        year = date.year
        doy = date.dayofyear - 1
        hour = date.hour
        if self.frequency == 6:
            era5 = self.era5_sfc[year - int(self.dates[0].year)][
                doy * 4 + hour // 6, ...
            ]
        else:
            era5 = self.era5_sfc[year - int(self.dates[0].year)][doy, ...]

        return np.copy(era5)

    def make_time_channels(self, index, x, y):
        """
        Make auxiliary time channels
        """

        date = self.dates[index]
        hour = date.hour
        doy = date.dayofyear - 1
        if date.year % 4 == 0:
            n_days = 366
        else:
            n_days = 365
        hour_sin = np.sin(hour * np.pi / 12) * np.float32(np.ones((1, x, y)))
        hour_cos = np.cos(hour * np.pi / 12) * np.float32(np.ones((1, x, y)))
        doy_sin = np.sin(doy * 2 * np.pi / n_days) * np.float32(np.ones((1, x, y)))
        doy_cos = np.cos(doy * 2 * np.pi / n_days) * np.float32(np.ones((1, x, y)))

        return np.concatenate([hour_sin, hour_cos, doy_sin, doy_cos])

    def __getitem__(self, index):

        # Option to offset to random leadtime
        lt_offset = 0
        if self.random_lt:
            lt_offset = np.random.choice([0, 2, 3])

        # Load ground truth data
        y_target = self.to_tensor(
            self.load_era5_time(index + self.lead_time - lt_offset)
        )

        # Load either initial condition or ERA5 depending on task
        if self.ic_path is not None:
            era5_ts0 = self.ic[index].copy().transpose(2, 1, 0)

        else:
            era5_ts0 = self.load_era5_time(index)

        # Auxiliary time
        time = self.make_time_channels(index, era5_ts0.shape[1], era5_ts0.shape[2])
        era5_ts0 = self.to_tensor(
            np.concatenate([era5_ts0, self.era5_elev, time], axis=0)
        )
        y_context = era5_ts0.permute(0, 2, 1)[:, ...]

        # Normalisation
        if self.diff:
            y_target = (y_target - era5_ts0[:24, ...]).permute(2, 1, 0)
            y_target = self.norm_era5_tendency(y_target, lt_offset)
            y_context[:24, ...] = self.norm_era5(y_context[:24, ...])

        else:
            if self.norm:
                y_context[:24, ...] = self.norm_era5(y_context[:24, ...], lt_offset)
                y_target = self.norm_era5(y_target, lt_offset)
            y_target = y_target.permute(2, 1, 0)

        if self.rollout:
            # Option to return entire timeseries of target data
            targets = []
            for t in range(self.lead_time + 1):
                t = self.to_tensor(self.load_era5_time(index + t))
                targets.append(t.permute(2, 1, 0))
            targets = torch.stack(targets, dim=-1)[..., ::4]

            return {
                "y_context": y_context.permute(0, 2, 1),
                "y_target": y_target,
                "targets": targets,
                "lt": self.to_tensor(np.array([lt_offset])),
            }

        else:
            return {
                "y_context": y_context.permute(0, 2, 1),
                "y_target": y_target[..., :],
                "lt": self.to_tensor(np.array([lt_offset])),
                "target_index": self.to_tensor(np.array([index])),
            }


class WeatherDatasetE2E(WeatherDataset):
    """
    Dataset for running Aardvark end-to-end
    """

    def __init__(
        self,
        device,
        hadisd_mode,
        start_date,
        end_date,
        lead_time,
        mode,
        hadisd_var,
        max_steps_per_epoch=None,
        era5_mode="sfc",
        res=1,
        filter_dates=None,
        var_start=0,
        var_end=24,
        diff=False,
        two_frames=False,
        region="global",
    ):

        super().__init__(
            device,
            hadisd_mode,
            start_date,
            end_date,
            lead_time,
            era5_mode,
            res=res,
            filter_dates=filter_dates,
            diff=diff,
        )

        # Setup
        self.var_start = var_start
        self.var_end = var_end
        self.diff = diff
        self.two_frames = two_frames
        self.region = region
        self.lead_time = lead_time
        self.mode = mode
        self.max_steps_per_epoch = max_steps_per_epoch

        # Initialise encoder dataset
        self.assimilation_dataset = WeatherDatasetAssimilation(
            device="cuda",
            hadisd_mode="train",
            start_date=start_date,
            end_date=end_date,
            lead_time=0,
            era5_mode="4u",
            res=1,
            var_start=0,
            var_end=24,
            diff=False,
            two_frames=False,
        )

        # Initialise forecast dataset
        self.forecast_dataset = ForecastLoader(
            device="cuda",
            mode=mode,
            lead_time=lead_time,
            era5_mode=era5_mode,
            res=1,
            frequency=6,
            diff=True,
            u_only=False,
            random_lt=False,
        )

        # Initialise downscaling dataset
        self.downscaling_dataset = ForecasterDatasetDownscaling(
            start_date=start_date,
            end_date=end_date,
            lead_time=lead_time,
            hadisd_var=hadisd_var,
            mode=mode,
            device=device,
            forecast_path=None,
            region=region,
        )

    def __len__(self):
        if self.max_steps_per_epoch:
            return self.max_steps_per_epoch
        return len(self.downscaling_dataset) - 40  # Need 10 day offset at end of year

    def __getitem__(self, index):

        if self.max_steps_per_epoch:
            index = np.random.choice(
                np.arange(len(self.downscaling_dataset) - 40)
            )  # Need 10 day offset at end of year

        # Get data for each of the three datasets
        assimilation = self.assimilation_dataset.__getitem__(index)
        forecast = self.forecast_dataset.__getitem__(index)
        downscaling = self.downscaling_dataset.__getitem__(index)

        # Create task
        task = {
            "assimilation": assimilation,
            "forecast": forecast,
            "downscaling": downscaling,
            "index": torch.tensor(index),
        }

        # Add y target to allow for end to end finetuning if needed
        task["y_target"] = task["downscaling"]["y_target"]

        return task

    def unnorm_pred(self, x):

        dev = x.device
        x = x.detach().cpu().numpy()

        x = (
            x
            * self.stds[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
            + self.means[np.newaxis, ...].transpose(0, 2, 3, 1)[
                ..., self.var_start : self.var_end
            ]
        )
        if bool(self.diff):
            x = (
                x
                + self.era5_mean_spatial[np.newaxis, ...].transpose(0, 3, 2, 1)[
                    ..., self.var_start : self.var_end
                ]
            )
        return torch.from_numpy(x).float().to(dev)
