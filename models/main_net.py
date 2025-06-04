import torch.nn as nn
import torch.nn.functional as F


class MainNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # --- 1D conv: from 1 channel → 6 channels, kernel=3, padding=1 to keep length ==
        self.layer1 = nn.Conv1d(1, 6, kernel_size=3, stride=1)
        if input_dim == 10:
            self.layer2 = nn.Linear(6 * ((input_dim * 2) - 2), 128)
            self.layer3 = nn.Linear(128, 128)
            self.layer4 = nn.Linear(128, output_dim)
        elif input_dim == 5:
            self.layer2 = nn.Linear(6 * ((input_dim * 2) - 2), 64)
            self.layer3 = nn.Linear(64, 64)
            self.layer4 = nn.Linear(64, output_dim)
        elif input_dim == 20:
            self.layer2 = nn.Linear(6 * ((input_dim * 2) - 2), 256)
            self.layer3 = nn.Linear(256, 256)
            self.layer4 = nn.Linear(256, output_dim)
        else:
            self.layer2 = nn.Linear(6 * ((input_dim * 2) - 2), 512)
            self.layer3 = nn.Linear(512, 256)
            self.layer4 = nn.Linear(256, output_dim)

        # Xavier initialization
        for m in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # x: [B, state_dim] → reshape to [B,1,state_dim] for Conv1d
        x = x.unsqueeze(1)
        x = F.leaky_relu(self.layer1(x))        # [B, 6, L]
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.layer2(x))          # [B, 128]
        x = F.leaky_relu(self.layer3(x))          # [B, 128]
        return self.layer4(x)
