"""
Genere les plots comparatifs Custom vs SB3.
Structure et style alignes avec plot_safety_aware.py.

Sorties :
    comparison_results/multiseed_training_curves.png
    comparison_results/multiseed_loss_curves.png
    comparison_results/multiseed_reward_vs_std.png
    comparison_results/multiseed_reward_bar.png

Usage :
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --eval-json comparison_results/multiseed_eval_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ================ CONFIG ================

AGENT_STYLES = {
    "custom": {"color": "#2196F3", "label": "Custom DQN", "linestyle": "-"},
    "sb3": {"color": "#F44336", "label": "SB3 DQN", "linestyle": "--"},
}

METRICS_PATHS = {
    "custom": "checkpoints_custom/metrics.json",
    "sb3": "checkpoints_sb3/metrics.json",
}

OUTPUT_DIR = "comparison_results"


# ================ HELPERS ================

def moving_average(values: list, window: int = 50) -> np.ndarray:
    """Moyenne mobile pour lisser les courbes d'entrainement."""
    result = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        result[i] = np.mean(values[i - window + 1 : i + 1])
    return result


def load_latest_eval_json(output_dir: str) -> dict | None:
    """Charge le fichier JSON d'evaluation multiseed le plus recent."""
    pattern = os.path.join(output_dir, "multiseed_eval_*.json")
    files = sorted(glob(pattern))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


def _extract_training_rewards(agent_name: str, metrics: dict) -> list[float]:
    if agent_name == "custom":
        return metrics.get("rewards", [])
    return metrics.get("episode_rewards", [])


def _extract_training_losses(agent_name: str, metrics: dict) -> np.ndarray:
    losses = np.asarray(metrics.get("losses", []), dtype=float)
    if losses.size == 0:
        return losses

    # On borne les valeurs extremes pour garder un plot lisible.
    finite_mask = np.isfinite(losses)
    losses = losses[finite_mask]
    if losses.size == 0:
        return losses
    clip_max = np.percentile(losses, 99)
    return np.clip(losses, 0.0, clip_max)


def _resample_to_length(values: np.ndarray, target_len: int) -> np.ndarray:
    """Re-echantillonne une serie sur une longueur cible (axe episodes)."""
    if values.size == 0:
        return values
    if target_len <= 1:
        return values[:1]
    if values.size == target_len:
        return values

    src_x = np.linspace(0.0, 1.0, num=values.size)
    dst_x = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(dst_x, src_x, values)


# ================ PLOT 1 : TRAINING CURVES ================

def plot_training_curves(output_dir: str, max_episodes: int = 500, window: int = 50) -> None:
    """Superpose les courbes d'entrainement (reward lisse) de Custom et SB3."""
    fig, ax = plt.subplots(figsize=(10, 5))
    found_any = False

    for agent_name, metrics_path in METRICS_PATHS.items():
        if not Path(metrics_path).exists():
            print(f"  metrics.json introuvable pour {agent_name} ({metrics_path}) - ignore.")
            continue

        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        rewards = _extract_training_rewards(agent_name, metrics)
        rewards = rewards[:max_episodes]
        smoothed = moving_average(rewards, window=window)
        style = AGENT_STYLES[agent_name]

        ax.plot(
            smoothed,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
            linewidth=1.8,
        )
        found_any = True

    if not found_any:
        print("  Aucun metrics.json trouve - training curves ignorees.")
        plt.close()
        return

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Recompense (moyenne mobile {window} episodes)")
    ax.set_title("Courbes d'entrainement - Custom vs SB3")
    ax.set_xlim(0, max_episodes - 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "multiseed_training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Training curves -> {out_path}")


# ================ PLOT 2 : LOSS CURVES ================

def plot_loss_curves(output_dir: str, max_episodes: int = 500, window: int = 50) -> None:
    """Superpose les courbes de loss (lissees) de Custom et SB3 sur 500 episodes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    found_any = False

    for agent_name, metrics_path in METRICS_PATHS.items():
        if not Path(metrics_path).exists():
            continue

        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        losses = _extract_training_losses(agent_name, metrics)
        if losses.size == 0:
            continue

        losses_resampled = _resample_to_length(losses, max_episodes)
        smoothed = moving_average(losses_resampled.tolist(), window=window)
        style = AGENT_STYLES[agent_name]

        ax.plot(
            smoothed,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
            linewidth=1.8,
        )
        found_any = True

    if not found_any:
        print("  Aucune loss trouvee - loss curves ignorees.")
        plt.close()
        return

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Loss (moyenne mobile {window} episodes)")
    ax.set_title("Courbes de loss - Custom vs SB3")
    ax.set_xlim(0, max_episodes - 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "multiseed_loss_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Loss curves -> {out_path}")


# ================ PLOT 3 : REWARD MEAN vs STD ================

def plot_reward_vs_std(eval_data: dict, output_dir: str) -> None:
    """Scatter plot : reward moyen vs ecart-type pour chaque agent (overall)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    model_map = {
        "custom": "custom_dqn",
        "sb3": "stable_baselines",
    }

    for short_name, full_name in model_map.items():
        style = AGENT_STYLES[short_name]
        overall = eval_data[full_name]["overall_stats"]

        x = overall["std"]
        y = overall["mean"]

        ax.scatter(x, y, color=style["color"], label=style["label"], s=80)
        ax.annotate(style["label"], (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Ecart-type reward (overall)")
    ax.set_ylabel("Reward moyenne (overall)")
    ax.set_title("Tradeoff performance / stabilite")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "multiseed_reward_vs_std.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Reward vs std -> {out_path}")


# ================ PLOT 4 : BAR CHART REWARD PER SEED ================

def plot_reward_bar(eval_data: dict, output_dir: str) -> None:
    """Bar chart comparant reward mean (+/- std) par seed et overall."""
    seeds = eval_data["seeds"]
    labels = [f"Seed {s}" for s in seeds] + ["Overall"]

    custom_seed_stats = eval_data["custom_dqn"]["per_seed"]
    sb3_seed_stats = eval_data["stable_baselines"]["per_seed"]

    custom_means = [custom_seed_stats[str(s)]["stats"]["mean"] for s in seeds]
    custom_stds = [custom_seed_stats[str(s)]["stats"]["std"] for s in seeds]
    sb3_means = [sb3_seed_stats[str(s)]["stats"]["mean"] for s in seeds]
    sb3_stds = [sb3_seed_stats[str(s)]["stats"]["std"] for s in seeds]

    custom_means.append(eval_data["custom_dqn"]["overall_stats"]["mean"])
    custom_stds.append(eval_data["custom_dqn"]["overall_stats"]["std"])
    sb3_means.append(eval_data["stable_baselines"]["overall_stats"]["mean"])
    sb3_stds.append(eval_data["stable_baselines"]["overall_stats"]["std"])

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        custom_means,
        width,
        yerr=custom_stds,
        capsize=4,
        label=AGENT_STYLES["custom"]["label"],
        color=AGENT_STYLES["custom"]["color"],
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        sb3_means,
        width,
        yerr=sb3_stds,
        capsize=4,
        label=AGENT_STYLES["sb3"]["label"],
        color=AGENT_STYLES["sb3"]["color"],
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reward moyenne")
    ax.set_title("Reward par seed - Custom vs SB3")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "multiseed_reward_bar.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Reward bar chart -> {out_path}")


# ================ MAIN ================

def main():
    parser = argparse.ArgumentParser(description="Plots comparatifs multiseed")
    parser.add_argument("--eval-json", type=str, default=None,
                        help="Chemin vers le JSON d'evaluation (defaut: le plus recent)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max-episodes", type=int, default=500,
                        help="Nombre d'episodes en abscisse pour les training curves")
    parser.add_argument("--window", type=int, default=50,
                        help="Fenetre de moyenne mobile")
    parser.add_argument("--loss-window", type=int, default=50,
                        help="Fenetre de moyenne mobile pour le plot loss")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Training curves (ne necessite pas le JSON d'eval) ----
    print("\n  Generation des training curves...")
    plot_training_curves(args.output_dir, max_episodes=args.max_episodes, window=args.window)

    print("\n  Generation des loss curves...")
    plot_loss_curves(args.output_dir, max_episodes=args.max_episodes, window=args.loss_window)

    # ---- Reward plots (necessitent le JSON d'eval) ----
    eval_json_path = args.eval_json
    if eval_json_path is None:
        eval_data = load_latest_eval_json(args.output_dir)
        if eval_data is None:
            print("\n  Aucun fichier multiseed_eval_*.json trouve dans "
                  f"{args.output_dir}.")
            print("  Lance d'abord evaluate_multiseed.py puis relance ce script.")
            return
    else:
        with open(eval_json_path, encoding="utf-8") as f:
            eval_data = json.load(f)

    print("\n  Generation du scatter reward vs std...")
    plot_reward_vs_std(eval_data, args.output_dir)

    print("\n  Generation du bar chart reward...")
    plot_reward_bar(eval_data, args.output_dir)

    print(f"\n  Tous les plots sont dans {args.output_dir}/")


if __name__ == "__main__":
    main()
