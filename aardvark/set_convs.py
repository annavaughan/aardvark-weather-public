import torch
import torch.nn as nn


class convDeepSet(nn.Module):
    """
    ConvDeepSet used to translate ungridded observations to a gridded representation and
    vice versa.
    """

    def __init__(
        self,
        init_ls,
        mode,
        device,
        density_channel=True,
        step=0.25,
        grid=False,
    ):
        super().__init__()
        self.init_ls = torch.nn.Parameter(torch.tensor([init_ls]))
        self.grid = grid
        self.step = step
        self.density_channel = density_channel
        self.mode = mode

        self.init_ls.requires_grad = True
        self.device = device

    def compute_weights(self, x1, x2):

        dists2 = self.pw_dists2(x1.unsqueeze(-1), x2.unsqueeze(-1))

        d = torch.exp((-0.5 * dists2) / (self.init_ls.to(x1.device)) ** 2)
        return d

    def pw_dists2(self, a, b):

        norms_a = torch.sum(a**2, axis=-1)[..., :, None]
        norms_b = torch.sum(b**2, axis=-1)[..., None, :]

        return norms_a + norms_b - 2 * torch.matmul(a, b.permute(0, 2, 1))

    def forward(self, x_in, wt, x_out):

        # Add a density channel
        density_channel = torch.ones_like(wt[:, 0:1, ...])
        density_channel[torch.isnan(wt[:, 0:1, ...])] = 0

        wt = torch.cat([density_channel, wt], dim=1)
        wt[torch.isnan(wt)] = 0

        if self.mode == "OffToOn":
            # Case 1: converting off-the-grid data to a gridded representation

            in_lon_mask = ~torch.isnan(x_in[0])
            in_lat_mask = ~torch.isnan(x_in[1])

            x_in[0][~in_lon_mask] = 0
            x_in[1][~in_lat_mask] = 0

            ws = [self.compute_weights(xzi, xi) for xzi, xi in zip(x_in, x_out)]

            ws[0] = ws[0] * in_lon_mask.unsqueeze(-1).int()
            ws[1] = ws[1] * in_lat_mask.unsqueeze(-1).int()

            ee = torch.einsum("...cw,...wx,...wy->...cxy", wt, ws[0], ws[1])

        elif self.mode == "OnToOn":
            # Case 2: converting between gridded representations

            ws = [self.compute_weights(xzi, xi) for xzi, xi in zip(x_in, x_out)]
            ee = torch.einsum("...cwh,...wx,...hy->...cxy", wt, ws[0], ws[1])

        elif self.mode == "OnToOff":

            # Case 3: converting a gridded representation to off-the-grid predictions
            out_lon_mask = ~torch.isnan(x_out[0])
            out_lat_mask = ~torch.isnan(x_out[1])
            x_out[0][~out_lon_mask] = 0
            x_out[1][~out_lat_mask] = 0

            ws = [self.compute_weights(xzi, xi) for xzi, xi in zip(x_in, x_out)]

            ws[0] = ws[0] * out_lon_mask.unsqueeze(-2).int()
            ws[1] = ws[1] * out_lat_mask.unsqueeze(-2).int()

            ee = torch.einsum("...cwh,...wx,...hx->...cx", wt, ws[0], ws[1])
        if self.density_channel:
            ee = torch.cat(
                [
                    ee[:, 0:1, ...],
                    ee[:, 1:, ...] / torch.clamp(ee[:, 0:1, ...], min=1e-6, max=1e5),
                ],
                dim=1,
            )

            return ee
        else:
            ee = ee[:, 1:, ...] / torch.clamp(ee[:, 0:1, ...], min=1e-6, max=1e5)
            return ee
