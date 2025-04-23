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
import csv
import matplotlib.pyplot as plt

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
    pixels = env.physics.render(height=84, width=84, camera_id=0).copy()  # Ensure positive strides
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
    csv_file = "crc_loss_log.csv"
    reward_file = "reward_log.csv"
    with open(csv_file, "w", newline="") as f, open(reward_file, "w", newline="") as rf:
        writer = csv.writer(f)
        reward_writer = csv.writer(rf)
        writer.writerow(["step", "crc_loss"])
        reward_writer.writerow(["step", "reward"])
        total_reward = 0.0
        episode = 0
        for step in range(5000):  # Run for 5000 steps (longer experiment)
            obs_aug = augment(obs.unsqueeze(0)).to(device)
            obs_key = augment(obs.unsqueeze(0)).to(device)
            z_q = encoder_q(obs_aug)
            z_k = encoder_k(obs_key)
            recon = decoder(z_q)
            pred = feature_pred(z_q)
            action = torch.tensor([[2.0 * torch.rand(1).item() - 1.0]], device=device)
            ts_next = env.step(action.cpu().numpy())
            next_obs = get_obs(env, ts_next)
            reward = float(ts_next.reward)
            total_reward += reward
            done = float(ts_next.last())
            buffer.add(obs.cpu().numpy(), action.cpu().numpy(), [[reward]], next_obs.cpu().numpy(), [[done]])
            if buffer.size > 1:
                batch = buffer.sample(1)
                loss = crc_loss(z_q, z_k, recon, obs_aug, z_q, z_k)
                optim_q.zero_grad(); optim_dec.zero_grad(); optim_pred.zero_grad()
                loss.backward()
                optim_q.step(); optim_dec.step(); optim_pred.step()
                writer.writerow([step, loss.item()])
                if step % 100 == 0:
                    print(f"Step {step} CRC loss: {loss.item()}")
            if done:
                reward_writer.writerow([step, total_reward])
                print(f"Episode {episode} total reward: {total_reward}")
                total_reward = 0.0
                episode += 1
                ts_next = env.reset()
                next_obs = get_obs(env, ts_next)
            obs = next_obs
    print(f"CRC loss log saved to {csv_file}")
    print(f"Reward log saved to {reward_file}")

    # Plot CRC loss curve
    import pandas as pd
    crc_log = pd.read_csv(csv_file)
    plt.figure(figsize=(10,4))
    plt.plot(crc_log['step'], crc_log['crc_loss'], label='CRC Loss')
    plt.xlabel('Step')
    plt.ylabel('CRC Loss')
    plt.title('CRC Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('crc_loss_curve.png')
    plt.show()

    # Plot reward curve
    reward_log = pd.read_csv(reward_file)
    plt.figure(figsize=(10,4))
    plt.plot(reward_log['step'], reward_log['reward'], label='Episode Reward')
    plt.xlabel('Step')
    plt.ylabel('Total Reward')
    plt.title('Episode Reward Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reward_curve.png')
    plt.show()

if __name__ == "__main__":
    main()
