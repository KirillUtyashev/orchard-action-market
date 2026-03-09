import torch
import torch.nn as nn

from debug.code.nn.encoders import EncoderOutput
from debug.code.core.enums import DEVICE


class MainNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            mlp_dims: tuple[int, ...] = (128, 128),
            negative_slope: float = 0.01,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for hd in mlp_dims:
            layers += [nn.Linear(d, hd), nn.LeakyReLU(negative_slope=negative_slope)]
            d = hd
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

        gain = nn.init.calculate_gain("leaky_relu", param=negative_slope)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, enc: EncoderOutput) -> torch.Tensor:
        x = enc.scalar.to(DEVICE).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)      # (1, D)
        return self.net(x)


class CNNMainNet(nn.Module):
    def __init__(
            self,
            grid_shape: tuple[int, int, int],
            scalar_dim: int,
            output_dim: int,
            conv_channels: list[int] = None,
            kernel_size: int = 3,
            mlp_dims: tuple[int, ...] = (128, 128),
            use_mlp: bool = True,
            negative_slope: float = 0.01,
    ):
        super().__init__()
        C, H, W = grid_shape
        conv_channels = conv_channels or [32, 64]
        self.use_mlp = bool(use_mlp)

        conv_layers: list[nn.Module] = []
        in_ch = C
        for out_ch in conv_channels:
            conv_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*conv_layers)

        head_input_dim = in_ch * H * W + scalar_dim
        if self.use_mlp:
            mlp_layers: list[nn.Module] = []
            d = head_input_dim
            for hd in mlp_dims:
                mlp_layers += [nn.Linear(d, hd), nn.LeakyReLU(negative_slope=negative_slope)]
                d = hd
            mlp_layers.append(nn.Linear(d, output_dim))
            self.head = nn.Sequential(*mlp_layers)
        else:
            self.head = nn.Linear(head_input_dim, output_dim)

        gain = nn.init.calculate_gain("leaky_relu", param=negative_slope)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, enc: EncoderOutput) -> torch.Tensor:
        grid = enc.grid.to(DEVICE).float()
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)            # (1, C, H, W)

        flat = self.cnn(grid).flatten(start_dim=1)  # (B, in_ch*H*W)

        if enc.scalar is not None:
            scalar = enc.scalar.to(DEVICE).float()
            if scalar.dim() == 1:
                scalar = scalar.unsqueeze(0)    # (1, D)
            flat = torch.cat([flat, scalar], dim=1)

        return self.head(flat)
