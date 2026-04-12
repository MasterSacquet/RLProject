"""
Génère les plots comparatifs pour les 3 agents safety-aware.
Nécessite d'avoir lancé evaluate_safety_aware.py au préalable
(pour le scatter et le bar chart), et les metrics.json de chaque
entraînement (pour les training curves).

Sorties :
    comparison_results/safety_aware_training_curves.png
    comparison_results/safety_aware_reward_vs_collision.png
    comparison_results/safety_aware_collision_rate_bar.png

Usage :
    python scripts/plot_safety_aware.py
    python scripts/plot_safety_aware.py --eval-json comparison_results/safety_aware_eval_20240101_120000.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ================ CONFIG ================

AGENT_STYLES = {
    "baseline":     {"color": "#2196F3", "label": "Baseline DQN",     "linestyle": "-"},
    "conservative": {"color": "#F44336", "label": "DQN Conservative", "linestyle": "--"},
    "moderate":     {"color": "#4CAF50", "label": "DQN Moderate",     "linestyle": "-."},
}

METRICS_PATHS = {
    "baseline":     "checkpoints_custom/metrics.json",
    "conservative": "checkpoints_safety_aware_conservative/metrics.json",
    "moderate":     "checkpoints_safety_aware_moderate/metrics.json",
}

OUTPUT_DIR = "comparison_results"


# ================ HELPERS ================

def moving_average(values: list, window: int = 50) -> np.ndarray:
    """Moyenne mobile pour lisser les courbes d'entraînement."""
    result = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        result[i] = np.mean(values[i - window + 1 : i + 1])
    return result


def load_latest_eval_json(output_dir: str) -> dict | None:
    """Charge le fichier JSON d'évaluation le plus récent."""
    pattern = os.path.join(output_dir, "safety_aware_eval_*.json")
    files   = sorted(glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


# ================ PLOT 1 : TRAINING CURVES ================

def plot_training_curves(output_dir: str) -> None:
    """Superpose les courbes d'entraînement (reward lissé) des 3 agents."""
    fig, ax = plt.subplots(figsize=(10, 5))
    found_any = False

    for agent_name, metrics_path in METRICS_PATHS.items():
        if not Path(metrics_path).exists():
            print(f"  ⚠️  metrics.json introuvable pour {agent_name} ({metrics_path}) — ignoré.")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        rewards  = metrics.get("rewards", [])
        smoothed = moving_average(rewards, window=50)
        style    = AGENT_STYLES[agent_name]

        ax.plot(smoothed, color=style["color"], linestyle=style["linestyle"],
                label=style["label"], linewidth=1.8)
        found_any = True

    if not found_any:
        print("  ⚠️  Aucun metrics.json trouvé — training curves ignorées.")
        plt.close()
        return

    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense (moyenne mobile 50 épisodes)")
    ax.set_title("Courbes d'entraînement — Baseline vs Safety-Aware")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "safety_aware_training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✅ Training curves → {out_path}")


# ================ PLOT 2 : SCATTER REWARD vs COLLISION ================

def plot_reward_vs_collision(eval_data: dict, output_dir: str) -> None:
    """Scatter plot : reward moyen vs taux de collision pour chaque agent."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for agent_name, agent_results in eval_data["results"].items():
        style   = AGENT_STYLES.get(agent_name, {"color": "gray", "label": agent_name})
        overall = agent_results["overall"]

        x = overall["collision_rate"] * 100      # en %
        y = overall["reward_mean"]
        e = overall["reward_std"]

        ax.errorbar(x, y, yerr=e, fmt="o", color=style["color"],
                    label=style["label"], markersize=10, capsize=5, linewidth=2)
        ax.annotate(style["label"], (x, y),
                    textcoords="offset points", xytext=(8, 4), fontsize=9,
                    color=style["color"])

    ax.set_xlabel("Taux de collision (%)")
    ax.set_ylabel("Récompense moyenne")
    ax.set_title("Tradeoff Sécurité / Performance")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    plt.tight_layout()

    out_path = os.path.join(output_dir, "safety_aware_reward_vs_collision.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✅ Scatter tradeoff → {out_path}")


# ================ PLOT 3 : BAR CHART COLLISION RATE ================

def plot_collision_rate_bar(eval_data: dict, output_dir: str) -> None:
    """Bar chart comparant le taux de collision des 3 agents par seed."""
    seeds       = eval_data["seeds"]
    agent_names = list(eval_data["results"].keys())
    n_agents    = len(agent_names)
    n_seeds     = len(seeds)

    x      = np.arange(n_seeds + 1)   # seeds + overall
    width  = 0.8 / n_agents
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, agent_name in enumerate(agent_names):
        style        = AGENT_STYLES.get(agent_name, {"color": "gray", "label": agent_name})
        agent_results = eval_data["results"][agent_name]

        rates = []
        for seed in seeds:
            summary = agent_results["per_seed"][str(seed)]["summary"]
            rates.append(summary["collision_rate"] * 100)
        rates.append(agent_results["overall"]["collision_rate"] * 100)

        offset = (i - n_agents / 2 + 0.5) * width
        bars   = ax.bar(x + offset, rates, width, label=style["label"],
                        color=style["color"], alpha=0.85)

        # Afficher la valeur au-dessus de chaque barre
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{rate:.1f}%", ha="center", va="bottom", fontsize=8)

    labels = [f"Seed {s}" for s in seeds] + ["Overall"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Taux de collision (%)")
    ax.set_title("Taux de collision par seed — Baseline vs Safety-Aware")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "safety_aware_collision_rate_bar.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✅ Bar chart collision → {out_path}")


# ================ MAIN ================

def main():
    parser = argparse.ArgumentParser(description="Plots comparatifs safety-aware")
    parser.add_argument("--eval-json",  type=str, default=None,
                        help="Chemin vers le JSON d'évaluation (défaut: le plus récent)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Training curves (ne nécessite pas le JSON d'éval) ----
    print("\n  Génération des training curves...")
    plot_training_curves(args.output_dir)

    # ---- Scatter + bar chart (nécessitent le JSON d'éval) ----
    eval_json_path = args.eval_json
    if eval_json_path is None:
        eval_data = load_latest_eval_json(args.output_dir)
        if eval_data is None:
            print("\n  ⚠️  Aucun fichier safety_aware_eval_*.json trouvé dans "
                  f"{args.output_dir}.")
            print("  Lance d'abord evaluate_safety_aware.py puis relance ce script.")
            return
    else:
        with open(eval_json_path) as f:
            eval_data = json.load(f)

    print("\n  Génération du scatter tradeoff...")
    plot_reward_vs_collision(eval_data, args.output_dir)

    print("\n  Génération du bar chart collision...")
    plot_collision_rate_bar(eval_data, args.output_dir)

    print(f"\n  Tous les plots sont dans {args.output_dir}/")


if __name__ == "__main__":
    main()