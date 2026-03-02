import torch
import torch.nn as nn


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

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))


class CNNMainNet(nn.Module):
    def __init__(
            self,
            grid_shape: tuple[int, int, int],
            scalar_dim: int,
            output_dim: int,
            conv_channels: list[int] = None,
            kernel_size: int = 3,
            mlp_dims: tuple[int, ...] = (128, 128),
            negative_slope: float = 0.01,
    ):
        super().__init__()
        C, H, W = grid_shape
        conv_channels = conv_channels or [32, 64]

        conv_layers: list[nn.Module] = []
        in_channels = C
        for out_channels in conv_channels:
            conv_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_channels = out_channels
        self.cnn = nn.Sequential(*conv_layers)

        mlp_input_dim = in_channels * H * W + scalar_dim

        mlp_layers: list[nn.Module] = []
        d = mlp_input_dim
        for hd in mlp_dims:
            mlp_layers += [nn.Linear(d, hd), nn.LeakyReLU(negative_slope=negative_slope)]
            d = hd
        mlp_layers.append(nn.Linear(d, output_dim))
        self.mlp = nn.Sequential(*mlp_layers)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = x[:, :4, :, :]
        scalars = x[:, 4, 0, :2]
        flat = self.cnn(grid).flatten(start_dim=1)
        return self.mlp(torch.cat([flat, scalars], dim=1))
