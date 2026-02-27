import torch
import torch.nn as nn


class MainNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 4,
            negative_slope: float = 0.01,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers += [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(negative_slope=negative_slope)]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=negative_slope)]
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Xavier init (with gain for leaky_relu)
        gain = nn.init.calculate_gain("leaky_relu", param=negative_slope)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.net(x)


class CNNMainNet(nn.Module):
    def __init__(
            self,
            grid_shape: tuple[int, int, int],  # (C, H, W) e.g. (4, 6, 6)
            scalar_dim: int,                    # number of scalar features concatenated after CNN
            output_dim: int,
            conv_channels: list[int] = [32, 64],
            kernel_size: int = 3,
            hidden_dim: int = 128,
            num_layers: int = 2,
            negative_slope: float = 0.01,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        C, H, W = grid_shape

        # --- CNN encoder ---
        conv_layers: list[nn.Module] = []
        in_channels = C
        for out_channels in conv_channels:
            conv_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_channels = out_channels
        self.cnn = nn.Sequential(*conv_layers)

        # Compute flattened CNN output size
        cnn_flat_dim = in_channels * H * W
        mlp_input_dim = cnn_flat_dim + scalar_dim

        # --- MLP head ---
        mlp_layers: list[nn.Module] = []
        if num_layers == 1:
            mlp_layers.append(nn.Linear(mlp_input_dim, output_dim))
        else:
            mlp_layers += [nn.Linear(mlp_input_dim, hidden_dim), nn.LeakyReLU(negative_slope=negative_slope)]
            for _ in range(num_layers - 2):
                mlp_layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=negative_slope)]
            mlp_layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*mlp_layers)

        # --- Xavier init ---
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
        # x: (batch, 5, H, W)
        grid = x[:, :4, :, :]          # (batch, 4, H, W) -> into CNN
        scalars = x[:, 4, 0, :2]       # (batch, 2) -> scalars stored at row 0, cols 0-1

        cnn_out = self.cnn(grid)
        flat = cnn_out.flatten(start_dim=1)
        combined = torch.cat([flat, scalars], dim=1)
        return self.mlp(combined)

