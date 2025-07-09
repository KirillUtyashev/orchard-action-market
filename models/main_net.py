import torch
import torch.nn as nn
import torch.nn.functional as F

# #
# class MainNet(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=4):
#         super().__init__()
#         # --- 1D conv: from 2 channels (agents, apples) → 6 channels, kernel=3, padding=1 to keep length == input_dim
#         self.layer1 = nn.Conv1d(1, 6, kernel_size=3, stride=1)
#         self.layer2 = nn.Linear(6 * ((input_dim * 2) - 2), hidden_dim)
#         self.layer3 = nn.Linear(hidden_dim, hidden_dim)
#         self.layer4 = nn.Linear(hidden_dim, output_dim)
#
#         # 1D conv: from 1 channel → 6 channels, kernel=3, no padding -> length_out = input_dim - 2
#
#         # if input_dim == 10:
#         #     self.layer2 = nn.Linear(6 * ((input_dim * 2) - 2), 128)
#         #     self.layer3 = nn.Linear(128, 128)
#         #     self.layer4 = nn.Linear(128, output_dim)
#         # elif input_dim == 5:
#         #     self.layer2 = nn.Linear(6 * ((input_dim * 2) - 2), 64)
#         #     self.layer3 = nn.Linear(64, 64)
#         #     self.layer4 = nn.Linear(64, output_dim)
#         # elif input_dim == 20:
#         #     self.layer2 = nn.Linear(6 * ((input_dim * 2) - 2), 256)
#         #     self.layer3 = nn.Linear(256, 256)
#         #     self.layer4 = nn.Linear(256, output_dim)
#         # else:
#         #     self.layer2 = nn.Linear(6 * ((input_dim * 2) - 2), 2 ** (
#         #                 6 * ((input_dim * 2) - 2) - 1).bit_length())
#         #     self.layer3 = nn.Linear(2 ** (6 * ((input_dim * 2) - 2) - 1).bit_length(), 256)
#         #     self.layer4 = nn.Linear(256, output_dim)
#
#         # After conv, feature length remains input_dim, so flattened size is 6 * input_dim
#         # conv_output_dim = 6 * input_dim
#         # We append 2 scalars (row, col)
#         # fc_input_dim = conv_output_dim + 2
#
#         # Fully connected layers
#         # self.layer2 = nn.Linear(6 * ((input_dim * 2) - 2), 64)
#         # self.layer3 = nn.Linear(64, 64)
#         # self.layer4 = nn.Linear(64, output_dim)
#
#         # Xavier initialization
#         for m in [self.layer1, self.layer2, self.layer3, self.layer4]:
#             if hasattr(m, 'weight'):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # x: [B, state_dim] → reshape to [B,1,state_dim] for Conv1d
#         x = x.unsqueeze(1)
#         x = F.leaky_relu(self.layer1(x))        # [B, 6, L]
#         x = x.view(x.size(0), -1)
#         x = F.leaky_relu(self.layer2(x))          # [B, 128]
#         x = F.leaky_relu(self.layer3(x))          # [B, 128]
#         return self.layer4(x)
#
# #


class MainNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=4):
        super().__init__()
        self.layers_list = nn.ModuleList()

        if num_layers == 1:
            self.layers_list.append(nn.Linear(input_dim * 2, output_dim))
        else:
            self.layers_list.append(nn.Linear(input_dim * 2, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers_list.append(nn.Linear(hidden_dim, output_dim))

        # Xavier initialization
        for m in self.layers_list:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(len(self.layers_list)):
            x = F.leaky_relu(self.layers_list[i](x))
        return x
