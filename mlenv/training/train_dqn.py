# mlenv/training/train_dqn.py
import os
import time

import numpy as np
import torch

from mlenv.env import DeliveryEnv
from mlenv.models import DQNAgent
from mlenv.training.replay_buffer import ReplayBuffer
from mlenv.utils import TrainingConfig, save_model


def epsilon_by_episode(cfg: TrainingConfig, episode: int) -> float:
    if episode >= cfg.epsilon_decay_episodes:
        return cfg.epsilon_end
    frac = episode / cfg.epsilon_decay_episodes
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def main():
    cfg = TrainingConfig()

    # чтобы импорт backend корректно работал, запускай из корня проекта:
    # python -m mlenv.training.train_dqn
    print("Using device:", cfg.device)

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
    os.makedirs("mlenv_logs", exist_ok=True)

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

            # обучение после накопления достаточного количества переходов
            if buffer.size >= cfg.start_learning_after and total_steps % cfg.train_every_step == 0:
                batch = buffer.sample(cfg.batch_size)
                loss = agent.update(*batch)
                episode_loss += loss

            # обновляем target-сеть
            if total_steps % cfg.target_update_every == 0:
                agent.update_target()

        avg_loss = episode_loss / max(1, episode_steps)
        print(
            f"[Episode {episode:04d}] steps={episode_steps:03d} "
            f"reward={episode_reward:.1f} loss={avg_loss:.4f} eps={eps:.3f}"
        )

        # Простое autosave раз в 50 эпизодов
        if episode % 50 == 0:
            save_model(agent.q_net, cfg.checkpoint_path)
            print(f"Checkpoint saved to {cfg.checkpoint_path}")

    # финальная сохранённая модель
    save_model(agent.q_net, cfg.checkpoint_path)
    print(f"Training finished. Final model saved to {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
