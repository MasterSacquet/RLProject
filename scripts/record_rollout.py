"""enregistre des rollout videos pour custom DQN, SB3 agents, DQN safety aware avec highway-v0."""

import argparse
from html import parser
import os
import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN as DQN_SB3

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from rlproject.highway_env_wrapper import HighwayV0Env
from rlproject.dqn_agent import DQNAgent


def record_custom(model_path, output_dir, episodes, seed):
    wrapper = HighwayV0Env(render_mode="rgb_array")
    env = wrapper.env
    env = RecordVideo(env, video_folder=output_dir, name_prefix=f"custom_seed{seed}", episode_trigger=lambda e: True)

    obs, _ = env.reset(seed=seed)
    state_dim = np.array(obs).flatten().shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    checkpoint = torch.load(model_path, map_location=agent.device, weights_only=False)
    agent.q_net.load_state_dict(checkpoint["model_state_dict"])
    agent.q_net.eval()
    agent.epsilon = 0.0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs = np.array(obs)
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.select_action(obs, env.action_space)
            obs, _, done, truncated, _ = env.step(action)
            obs = np.array(obs)

    env.close()


def record_sb3(model_path, output_dir, episodes, seed):
    wrapper = HighwayV0Env(render_mode="rgb_array")
    env = wrapper.env
    env = RecordVideo(env, video_folder=output_dir, name_prefix=f"sb3_seed{seed}", episode_trigger=lambda e: True)

    model = DQN_SB3.load(model_path, env=env)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs = np.array(obs)
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            obs = np.array(obs)

    env.close()

def record_safety_aware(model_path, output_dir, episodes, seed, variant):
    wrapper = HighwayV0Env(render_mode="rgb_array")
    env = wrapper.env
    env = RecordVideo(env, video_folder=output_dir, name_prefix=f"{variant}_seed{seed}", episode_trigger=lambda e: True)

    obs, _ = env.reset(seed=seed)
    state_dim = np.array(obs).flatten().shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    checkpoint = torch.load(model_path, map_location=agent.device, weights_only=False)
    agent.q_net.load_state_dict(checkpoint["model_state_dict"])
    agent.q_net.eval()
    agent.epsilon = 0.0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs = np.array(obs)
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.select_action(obs, env.action_space)
            obs, _, done, truncated, _ = env.step(action)
            obs = np.array(obs)

    env.close()

def main():
    parser = argparse.ArgumentParser(description="Record rollout videos for custom DQN or SB3")
    parser.add_argument("--agent", type=str, choices=["custom", "sb3", "conservative", "moderate"], required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="rollouts")
    parser.add_argument("--custom-model", type=str, default="checkpoints_custom/last_model.pth")
    parser.add_argument("--sb3-model", type=str, default="checkpoints_sb3/dqn_highway.zip")
    parser.add_argument("--conservative-model", type=str, default="checkpoints_safety_aware_conservative/best_model.pth")
    parser.add_argument("--moderate-model",     type=str, default="checkpoints_safety_aware_moderate/best_model.pth")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.agent == "custom":
        record_custom(args.custom_model, args.output_dir, args.episodes, args.seed)
    if args.agent == "custom":
        record_custom(args.custom_model, args.output_dir, args.episodes, args.seed)
    elif args.agent == "sb3":
        record_sb3(args.sb3_model, args.output_dir, args.episodes, args.seed)
    elif args.agent == "conservative":
        record_safety_aware(args.conservative_model, args.output_dir, args.episodes, args.seed, "conservative")
    elif args.agent == "moderate":
        record_safety_aware(args.moderate_model, args.output_dir, args.episodes, args.seed, "moderate")

    print("Rollout recording complete")


if __name__ == "__main__":
    main()
