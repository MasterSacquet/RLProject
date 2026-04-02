""" Evalue custom DQN et SB3 models avec différentes seeds et sauvegarde les stats."""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from highway_env_wrapper import HighwayV0Env
from dqn_agent import DQNAgent
from stable_baselines3 import DQN as DQN_SB3


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_stats(rewards):
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards)),
    }


def evaluate_custom(model_path, episode_seeds):
    wrapper = HighwayV0Env()
    env = wrapper.env

    obs, _ = env.reset(seed=episode_seeds[0])
    state_dim = np.array(obs).flatten().shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    checkpoint = torch.load(model_path, map_location=agent.device, weights_only=False)
    agent.q_net.load_state_dict(checkpoint["model_state_dict"])
    agent.q_net.eval()
    agent.epsilon = 0.0

    rewards = []
    lengths = []

    for seed in episode_seeds:
        obs, _ = env.reset(seed=seed)
        obs = np.array(obs)
        episode_reward = 0.0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(obs, env.action_space)
            obs, reward, done, truncated, _ = env.step(action)
            obs = np.array(obs)
            episode_reward += float(reward)
            steps += 1

        rewards.append(episode_reward)
        lengths.append(steps)

    env.close()
    return rewards, lengths


def evaluate_sb3(model_path, episode_seeds):
    wrapper = HighwayV0Env()
    env = wrapper.env

    model = DQN_SB3.load(model_path, env=env)

    rewards = []
    lengths = []

    for seed in episode_seeds:
        obs, _ = env.reset(seed=seed)
        obs = np.array(obs)
        episode_reward = 0.0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            obs = np.array(obs)
            episode_reward += float(reward)
            steps += 1

        rewards.append(episode_reward)
        lengths.append(steps)

    env.close()
    return rewards, lengths


def build_episode_seeds(base_seed, num_episodes):
    return [base_seed * 1000 + i for i in range(num_episodes)]


def parse_seeds(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation for Custom DQN and SB3 DQN")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated list of seeds")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per seed")
    parser.add_argument("--custom-model", type=str, default="checkpoints_custom/last_model.pth")
    parser.add_argument("--sb3-model", type=str, default="checkpoints_sb3/dqn_highway.zip")
    parser.add_argument("--output-dir", type=str, default="comparison_results")

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "num_episodes": args.episodes,
        "seeds": seeds,
        "custom_dqn": {"per_seed": {}, "all_rewards": []},
        "stable_baselines": {"per_seed": {}, "all_rewards": []},
        "table_markdown": "",
    }

    for seed in seeds:
        set_global_seeds(seed)
        episode_seeds = build_episode_seeds(seed, args.episodes)

        custom_rewards, custom_lengths = evaluate_custom(args.custom_model, episode_seeds)
        sb3_rewards, sb3_lengths = evaluate_sb3(args.sb3_model, episode_seeds)

        results["custom_dqn"]["per_seed"][str(seed)] = {
            "episode_seeds": episode_seeds,
            "rewards": custom_rewards,
            "lengths": custom_lengths,
            "stats": compute_stats(custom_rewards),
        }
        results["stable_baselines"]["per_seed"][str(seed)] = {
            "episode_seeds": episode_seeds,
            "rewards": sb3_rewards,
            "lengths": sb3_lengths,
            "stats": compute_stats(sb3_rewards),
        }

        results["custom_dqn"]["all_rewards"].extend(custom_rewards)
        results["stable_baselines"]["all_rewards"].extend(sb3_rewards)

    results["custom_dqn"]["overall_stats"] = compute_stats(results["custom_dqn"]["all_rewards"])
    results["stable_baselines"]["overall_stats"] = compute_stats(results["stable_baselines"]["all_rewards"])

    table_lines = [
        "| seed | custom_mean | custom_std | sb3_mean | sb3_std |",
        "|---|---|---|---|---|",
    ]

    for seed in seeds:
        c_stats = results["custom_dqn"]["per_seed"][str(seed)]["stats"]
        s_stats = results["stable_baselines"]["per_seed"][str(seed)]["stats"]
        table_lines.append(
            f"| {seed} | {c_stats['mean']:.2f} | {c_stats['std']:.2f} | {s_stats['mean']:.2f} | {s_stats['std']:.2f} |"
        )

    c_overall = results["custom_dqn"]["overall_stats"]
    s_overall = results["stable_baselines"]["overall_stats"]
    table_lines.append(
        f"| overall | {c_overall['mean']:.2f} | {c_overall['std']:.2f} | {s_overall['mean']:.2f} | {s_overall['std']:.2f} |"
    )

    table_markdown = "\n".join(table_lines)
    results["table_markdown"] = table_markdown

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"multiseed_eval_{stamp}.json")
    table_path = os.path.join(args.output_dir, f"multiseed_eval_{stamp}.md")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(table_path, "w") as f:
        f.write("# Multi-seed Evaluation Table\n\n")
        f.write(f"Episodes per seed: {args.episodes}\n\n")
        f.write(table_markdown)
        f.write("\n")

    print("Evaluation complete")
    print(f"JSON saved to: {json_path}")
    print(f"Table saved to: {table_path}")


if __name__ == "__main__":
    main()
