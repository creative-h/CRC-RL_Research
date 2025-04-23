# Feature predictor for CRC-RL
import torch
import torch.nn as nn

class FeaturePredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, x):
        return self.fc(x)
