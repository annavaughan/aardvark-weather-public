import math

import torch
import torch.nn as nn


def cylindrical_conv_pad(x, w_pad):
    return torch.cat([x[..., -w_pad:], x, x[..., :w_pad]], axis=-1)


class CylindricalConv2D(nn.Conv2d):
    """
    UNet with cylinderical boundary conditions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1

        self.h_pad = self.kernel_size[0] // 2
        self.w_pad = self.kernel_size[1] // 2

    def forward(self, x: torch.Tensor):
        x = nn.functional.pad(x, (0, 0, self.h_pad, self.h_pad))
        return super().forward(cylindrical_conv_pad(x, self.w_pad))


class CylindricalConvTranspose2D(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1

        self.sh, self.sw = self.stride
        self.kh, self.kw = self.kernel_size

        self.h_pad = math.ceil(((self.sh - 1) + 2 * (self.kh // 2)) / self.sh)
        self.w_pad = math.ceil(((self.sw - 1) + 2 * (self.kw // 2)) / self.sw)

        self.h0 = self.sh * self.h_pad - (self.sh - 1) + (self.kh // 2)
        self.w0 = self.sw * self.w_pad - (self.sw - 1) + (self.kw // 2)

        self._bias = nn.Parameter(10**-3 * torch.randn(out_channels))

    def forward(self, x: torch.Tensor):

        Nh = x.shape[2] * self.sh
        Nw = x.shape[3] * self.sw

        x = cylindrical_conv_pad(x, self.w_pad)
        x = nn.functional.pad(x, (0, 0, self.h_pad, self.h_pad))

        x = super().forward(x)

        return x[:, :, self.h0 : self.h0 + Nh, self.w0 : self.w0 + Nw]


class Down(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        p=0,
        film=False,
        down=True,
        attn=False,
    ):

        super().__init__()

        self.film = film
        self.attn = attn

        self.conv_1 = CylindricalConv2D(
            in_channels, out_channels, kernel_size=3, stride=1
        )
        if down:
            self.conv_2 = CylindricalConv2D(
                out_channels, out_channels, kernel_size=3, stride=2
            )
        else:
            self.conv_2 = CylindricalConv2D(
                out_channels, out_channels, kernel_size=3, stride=1
            )

        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.GELU()

        if film:

            self.gamma_1 = torch.nn.Parameter(
                torch.ones(10, out_channels, 1, 1),
            )
            self.gamma_2 = torch.nn.Parameter(
                torch.ones(10, out_channels, 1, 1),
            )
            self.beta_1 = torch.nn.Parameter(
                torch.zeros(10, out_channels, 1, 1),
            )
            self.beta_2 = torch.nn.Parameter(
                torch.zeros(10, out_channels, 1, 1),
            )

        if self.attn:
            self.mha = AttentionBlock(n_channels=out_channels, n_heads=8)

    def forward(self, xi, film_index=None):

        film_index = film_index[:, 0].int()

        x = self.conv_1(xi)
        x = self.bn_1(x)
        if self.film:
            g1 = torch.index_select(self.gamma_1, 0, film_index)
            b1 = torch.index_select(self.beta_1, 0, film_index)

            x = g1 * x + b1

        x = self.activation(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        if self.film:
            g2 = torch.index_select(self.gamma_2, 0, film_index)
            b2 = torch.index_select(self.beta_2, 0, film_index)
            x = g2 * x + b2
        x = self.activation(x)

        if self.attn:
            x = self.mha(x)

        return x


class Up(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        p,
        bilinear=False,
        film=False,
        stride=2,
        attn=False,
    ):
        super().__init__()

        self.film = film

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
        else:
            self.up = CylindricalConvTranspose2D(
                in_channels, out_channels, kernel_size=3, stride=stride
            )

        self.conv = Down(
            out_channels,
            out_channels,
            p=0,
            film=film,
            down=False,
            attn=attn,
        )

    def forward(self, x1, x2, film_index=None):
        x1 = self.up(x1)
        x1 = self.conv(x1, film_index=film_index)

        if x1.shape[-1] != x2.shape[-1]:
            x1 = x1[..., :, :-1]
        if x1.shape[-2] != x2.shape[-2]:
            x1 = x1[..., :-1, :]

        return torch.cat([x2, x1], dim=1)


class Unet(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        div_factor=1,
        p=0.0,
        context=True,
        film=False,
        film_base=True,
    ):
        super(Unet, self).__init__()

        self.n_channels = in_channels
        self.bilinear = True
        self.fp = nn.Softplus()
        self.variances = nn.Parameter(torch.zeros([out_channels]))
        self.context = context
        self.film = film

        m = 1

        self.down1 = Down(
            self.n_channels,
            m * 128 // div_factor,
            p=0,
            film=film,
            attn=False,
        )
        self.down2 = Down(
            m * 128 // div_factor,
            m * 256 // div_factor,
            p=0,
            film=film,
            attn=False,
        )
        self.down3 = Down(
            m * 256 // div_factor,
            m * 512 // div_factor,
            p=0,
            film=film,
            attn=False,
        )
        self.down4 = Down(
            m * 512 // div_factor,
            m * 512 // div_factor,
            p=0,
            film=film,
            attn=False,
        )
        self.up1 = Up(
            m * 512 // div_factor,
            m * 512 // div_factor,
            p=0,
            film=film,
            attn=False,
        )
        self.up2 = Up(
            m * 1024 // div_factor,
            m * 256 // div_factor,
            p=0,
            film=film,
            attn=False,
        )
        self.up3 = Up(
            m * 512 // div_factor,
            m * 128 // div_factor,
            p=0,
            film=film,
            attn=False,
        )
        self.up4 = Up(
            m * 256 // div_factor,
            m * 64 // div_factor,
            p=0,
            film=film,
            attn=False,
        )

        self.out = nn.Conv2d(
            m * 64 // div_factor + in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x, film_index=None):

        x1 = x.contiguous()
        x2 = self.down1(x1, film_index=film_index)
        x3 = self.down2(x2, film_index=film_index)
        x4 = self.down3(x3, film_index=film_index)
        x5 = self.down4(x4, film_index=film_index)
        x = self.up1(x5, x4, film_index=film_index)
        x = self.up2(x, x3, film_index=film_index)
        x = self.up3(x, x2, film_index=film_index)
        x = self.up4(x, x1, film_index=film_index)
        x = self.out(x)

        return x.permute(0, 2, 3, 1)
