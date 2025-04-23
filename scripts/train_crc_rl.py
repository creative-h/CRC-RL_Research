import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from crc_rl.encoders import Encoder, update_ema
from crc_rl.decoder import Decoder
from crc_rl.feature_predictor import FeaturePredictor
from crc_rl.losses import crc_loss
from crc_rl.augmentations import augment
from crc_rl.sac_agent import SACAgent
from crc_rl.replay_buffer import ReplayBuffer

# DeepMind Control Suite integration
def make_env():
    try:
        from dm_control import suite
    except ImportError:
        raise ImportError("Please install dm_control: !pip install dm_control")
    env = suite.load(
        domain_name="cartpole",
        task_name="balance",
        task_kwargs={"random": 0},
        visualize_reward=False,
        environment_kwargs={"flat_observation": False}
    )
    return env

def get_obs(env, ts):
    # Render pixels directly from the physics engine
    pixels = env.physics.render(height=84, width=84, camera_id=0)
    obs = torch.tensor(pixels, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return obs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_shape = (3, 84, 84)
    latent_dim = 50
    action_dim = 1  # Cartpole is 1D action
    encoder_q = Encoder(obs_shape, latent_dim).to(device)
    encoder_k = Encoder(obs_shape, latent_dim).to(device)
    decoder = Decoder(latent_dim, obs_shape).to(device)
    feature_pred = FeaturePredictor(latent_dim).to(device)
    agent = SACAgent(latent_dim, action_dim, device)
    buffer = ReplayBuffer(10000, obs_shape, action_dim, device)
    env = make_env()
    ts = env.reset()
    obs = get_obs(env, ts)
    import torch.optim as optim
    optim_q = optim.Adam(encoder_q.parameters(), lr=1e-3)
    optim_dec = optim.Adam(decoder.parameters(), lr=1e-3)
    optim_pred = optim.Adam(feature_pred.parameters(), lr=1e-3)
    for step in range(10):  # Increase for real training
        obs_aug = augment(obs.unsqueeze(0)).to(device)
        obs_key = augment(obs.unsqueeze(0)).to(device)
        z_q = encoder_q(obs_aug)
        z_k = encoder_k(obs_key)
        recon = decoder(z_q)
        pred = feature_pred(z_q)
        # Dummy action for Cartpole: random [-1, 1]
        action = torch.tensor([[2.0 * torch.rand(1).item() - 1.0]], device=device)
        ts_next = env.step(action.cpu().numpy())
        next_obs = get_obs(env, ts_next)
        reward = torch.tensor([[ts_next.reward]], dtype=torch.float32)
        done = torch.tensor([[float(ts_next.last())]], dtype=torch.float32)
        buffer.add(obs.cpu().numpy(), action.cpu().numpy(), reward.numpy(), next_obs.cpu().numpy(), done.numpy())
        # Sample from buffer if enough samples
        if buffer.size > 1:
            batch = buffer.sample(1)
            # CRC loss example
            loss = crc_loss(z_q, z_k, recon, obs_aug, z_q, z_k)
            optim_q.zero_grad(); optim_dec.zero_grad(); optim_pred.zero_grad()
            loss.backward()
            optim_q.step(); optim_dec.step(); optim_pred.step()
            print(f"Step {step} CRC loss: {loss.item()}")
        obs = next_obs

if __name__ == "__main__":
    main()
