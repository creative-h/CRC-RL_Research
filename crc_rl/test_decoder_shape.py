import torch
from crc_rl.decoder import Decoder

obs_shape = (3, 84, 84)
latent_dim = 50
batch_size = 8

decoder = Decoder(latent_dim, obs_shape)
z = torch.randn(batch_size, latent_dim)
recon = decoder(z)
print('Final recon shape:', recon.shape)
