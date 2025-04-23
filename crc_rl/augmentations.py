# Data augmentation for CRC-RL
import torch
import torchvision.transforms as T
from PIL import Image
import random

def get_augmentation(obs_shape):
    c, h, w = obs_shape
    return T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop((h, w), scale=(0.8, 1.0)),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.Resize((h, w)),  # Ensure output size matches input
        T.ToTensor()
    ])

def augment(obs):
    # obs: [B, C, H, W] tensor
    aug = get_augmentation(obs.shape[1:])
    return torch.stack([aug(img) for img in obs])
