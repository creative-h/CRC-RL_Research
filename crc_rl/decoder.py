# Decoder for CRC-RL autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_shape):
        super().__init__()
        c, h, w = obs_shape
        self.fc = nn.Linear(latent_dim, 128 * (h // 8) * (w // 8))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, 3, stride=2, output_padding=1),
            nn.Sigmoid()
        )
        self.h, self.w = h, w

    def forward(self, z):
        x = self.fc(z)
        h8, w8 = self.h // 8, self.w // 8
        x = x.view(-1, 128, h8, w8)
        print("After fc/view:", x.shape)
        for layer in self.deconv:
            x = layer(x)
            print("After", layer, ":", x.shape)
        x = F.interpolate(x, size=(self.h, self.w), mode='bilinear', align_corners=False)
        print("After interpolate:", x.shape)
        return x
