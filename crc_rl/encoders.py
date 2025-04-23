# Encoder module for CRC-RL
# QueryEncoder and KeyEncoder (Siamese, with EMA update for key)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, obs_shape, latent_dim):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_dim = self._get_conv_out_dim(obs_shape)
        self.fc = nn.Linear(conv_out_dim, latent_dim)
    def _get_conv_out_dim(self, obs_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            out = self.conv(dummy)
        return out.view(1, -1).size(1)
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return F.normalize(x, dim=-1)

def update_ema(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * s_param.data + (1 - tau) * t_param.data)
