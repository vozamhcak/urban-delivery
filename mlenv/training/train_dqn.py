# mlenv/training/train_dqn.py
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from mlenv.env import DeliveryEnv
from mlenv.models import DQNAgent
from mlenv.training.replay_buffer import ReplayBuffer
from mlenv.utils import TrainingConfig, save_model


def epsilon_by_episode(cfg: TrainingConfig, episode: int) -> float:
    """
    Linear decrease of eps from epsilon_start to epsilon_end
    over epsilon_decay_episodes. After that, fix epsilon_end.
    """
    if episode >= cfg.epsilon_decay_episodes:
        return cfg.epsilon_end
    frac = episode / cfg.epsilon_decay_episodes
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def plot_rewards(episode_rewards, window, out_path):
    """
    Draws graph:
    - raw rewards by episodes
    - moving average over window
    """
    if len(episode_rewards) == 0:
        return

    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards, dtype=np.float32)

    # moving average
    if len(rewards) >= window:
        kernel = np.ones(window, dtype=np.float32) / window
        moving_avg = np.convolve(rewards, kernel, mode="valid")
        ma_episodes = episodes[window - 1 :]
    else:
        moving_avg = None
        ma_episodes = None

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.4, label="Episode reward")
    if moving_avg is not None:
        plt.plot(ma_episodes, moving_avg, linewidth=2.5, label=f"Moving avg (window={window})")

    plt.title("Reward over episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    cfg = TrainingConfig()

    # override number of episodes for presentation task
    cfg.total_episodes = 10_000

    # for backend import to work correctly, run from project root:
    # python -m mlenv.training.train_dqn
    print("Using device:", cfg.device)

    # directories for logs, checkpoints and plots
    os.makedirs("mlenv_logs", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
    plots_dir = "mlenv_plots"
    os.makedirs(plots_dir, exist_ok=True)

    env = DeliveryEnv(
        num_couriers=cfg.num_couriers,
        orders_per_minute=cfg.orders_per_minute,
        max_episode_orders=cfg.max_episode_orders,
        max_episode_time=cfg.max_episode_time,
        courier_speed=1.5,
        invalid_action_penalty=cfg.invalid_action_penalty,
        gamma=cfg.gamma,
    )

    agent = DQNAgent(
        obs_dim=env.observation_dim,
        num_actions=env.num_actions,
        lr=cfg.lr,
        gamma=cfg.gamma,
        device=cfg.device,
    )

    buffer = ReplayBuffer(obs_dim=env.observation_dim, capacity=cfg.replay_capacity)

    total_steps = 0
    episode_rewards = []  # store rewards for plotting graph

    start_time = time.time()

    for episode in range(1, cfg.total_episodes + 1):
        obs = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        episode_steps = 0

        eps = epsilon_by_episode(cfg, episode)
        done = False

        while not done and episode_steps < cfg.max_steps_per_episode:
            action = agent.select_action(obs, eps)
            next_obs, reward, done, info = env.step(action)

            buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # training after accumulating enough transitions
            if buffer.size >= cfg.start_learning_after and total_steps % cfg.train_every_step == 0:
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = buffer.sample(
                    cfg.batch_size
                )
                loss = agent.update(
                    batch_obs,
                    batch_actions,
                    batch_rewards,
                    batch_next_obs,
                    batch_dones,
                )
                episode_loss += loss

            # update target network
            if total_steps % cfg.target_update_every == 0:
                agent.update_target()

        # log episode results
        episode_rewards.append(episode_reward)
        avg_loss = episode_loss / max(1, episode_steps)
        elapsed = time.time() - start_time

        print(
            f"[Episode {episode:05d}/{cfg.total_episodes}] "
            f"steps={episode_steps:03d} "
            f"reward={episode_reward:7.2f} "
            f"avg_loss={avg_loss:.4f} "
            f"eps={eps:.3f} "
            f"buffer={buffer.size:06d} "
            f"time={elapsed/60:.1f} min"
        )

        # every 500 episodes — save model and graph
        if episode % 500 == 0:
            # model checkpoint
            save_model(agent.q_net, cfg.checkpoint_path)
            print(f"Checkpoint saved to {cfg.checkpoint_path}")

            # reward graph from start to current episode
            plot_path = os.path.join(plots_dir, f"rewards_{episode:05d}.png")
            plot_rewards(episode_rewards, window=100, out_path=plot_path)
            print(f"Reward plot saved to {plot_path}")

    # final saved model
    save_model(agent.q_net, cfg.checkpoint_path)
    final_plot_path = os.path.join(plots_dir, f"rewards_final_{cfg.total_episodes:05d}.png")
    plot_rewards(episode_rewards, window=100, out_path=final_plot_path)

    print(f"Training finished. Final model saved to {cfg.checkpoint_path}")
    print(f"Final reward plot saved to {final_plot_path}")


if __name__ == "__main__":
    main()
