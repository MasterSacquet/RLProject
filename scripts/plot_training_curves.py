"""Plot les courbes d'entrainement depuis metrics.json et sauvegarde une image de comparaison."""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def safe_list(value):
    return value if isinstance(value, list) else []


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from metrics.json files")
    parser.add_argument("--custom-metrics", type=str, default="checkpoints_custom/metrics.json")
    parser.add_argument("--sb3-metrics", type=str, default="checkpoints_sb3/metrics.json")
    parser.add_argument("--output-dir", type=str, default="comparison_results")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    custom = load_json(args.custom_metrics)
    sb3 = load_json(args.sb3_metrics)

    custom_rewards = safe_list(custom.get("rewards"))
    custom_losses = safe_list(custom.get("losses"))
    custom_avg = safe_list(custom.get("avg_rewards_100"))

    sb3_rewards = safe_list(sb3.get("episode_rewards"))
    sb3_losses = safe_list(sb3.get("losses"))
    sb3_timesteps = safe_list(sb3.get("timesteps"))
    sb3_loss_timesteps = safe_list(sb3.get("loss_timesteps"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Curves: Custom DQN vs SB3", fontsize=14, fontweight="bold")

    # Custom rewards
    ax = axes[0, 0]
    if custom_rewards:
        ax.plot(custom_rewards, label="Custom reward", alpha=0.7)
    if custom_avg:
        start = max(0, len(custom_rewards) - len(custom_avg))
        ax.plot(range(start, start + len(custom_avg)), custom_avg, label="Custom avg100", linewidth=2)
    ax.set_title("Custom DQN Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend()

    # Custom loss
    ax = axes[0, 1]
    if custom_losses:
        ax.plot(custom_losses, label="Custom loss", color="tab:red", alpha=0.7)
    ax.set_title("Custom DQN Loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    # SB3 rewards
    ax = axes[1, 0]
    if sb3_rewards:
        x = sb3_timesteps if sb3_timesteps else list(range(len(sb3_rewards)))
        ax.plot(x, sb3_rewards, label="SB3 reward", alpha=0.7)
    ax.set_title("SB3 Rewards")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend()

    # SB3 loss
    ax = axes[1, 1]
    if sb3_losses:
        x = sb3_loss_timesteps if len(sb3_loss_timesteps) == len(sb3_losses) else list(range(len(sb3_losses)))
        ax.plot(x, sb3_losses, label="SB3 loss", color="tab:red", alpha=0.7)
    ax.set_title("SB3 Loss")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"training_curves_{stamp}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
