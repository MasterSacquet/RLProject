"""
Évalue les trois agents (baseline, conservative, moderate) sur 3 seeds x 50 épisodes.
Utilise safety_metrics.py d'Albane pour calculer collision_rate, safety_margin, etc.

Usage :
    python scripts/evaluate_safety_aware.py
    python scripts/evaluate_safety_aware.py --seeds 0,1,3 --episodes 50
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from rlproject.highway_env_wrapper import HighwayV0Env
from rlproject.dqn_agent import DQNAgent
from rlproject.safety_metrics import run_episode, compute_safety_summary


# ================ AGENTS À ÉVALUER ================
AGENTS = {
    "baseline":     "checkpoints_custom/best_model.pth",
    "conservative": "checkpoints_safety_aware_conservative/best_model.pth",
    "moderate":     "checkpoints_safety_aware_moderate/best_model.pth",
}


# ================ HELPERS ================

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_agent(model_path: str, env) -> DQNAgent:
    """Charge un DQNAgent depuis un checkpoint .pth — même logique que evaluate_multiseed.py."""
    obs, _ = env.reset()
    state_dim  = np.array(obs).flatten().shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    checkpoint = torch.load(model_path, map_location=agent.device, weights_only=False)
    agent.q_net.load_state_dict(checkpoint["model_state_dict"])
    agent.q_net.eval()
    agent.epsilon = 0.0  # pas d'exploration en évaluation
    return agent


def make_select_action(agent: DQNAgent, action_space):
    """Retourne une fonction select_action compatible avec run_episode() d'Albane."""
    def select_action(obs):
        obs_flat = np.array(obs).flatten()
        return agent.select_action(obs_flat, action_space)
    return select_action


def build_episode_seeds(base_seed: int, num_episodes: int) -> list[int]:
    """Même convention que evaluate_multiseed.py."""
    return [base_seed * 1000 + i for i in range(num_episodes)]


def evaluate_agent(model_path: str, seeds: list[int], num_episodes: int) -> dict:
    """
    Évalue un agent sur toutes les seeds.
    Retourne un dict structuré par seed + overall.
    """
    wrapper = HighwayV0Env()
    env     = wrapper.env
    agent   = load_agent(model_path, env)
    select_action = make_select_action(agent, env.action_space)

    per_seed    = {}
    all_episodes = []

    for seed in seeds:
        set_global_seeds(seed)
        episode_seeds = build_episode_seeds(seed, num_episodes)
        seed_episodes = []

        for ep_seed in episode_seeds:
            record = run_episode(env, select_action, seed=ep_seed)
            seed_episodes.append(record)

        per_seed[str(seed)] = {
            "summary": compute_safety_summary(seed_episodes),
            "episodes": seed_episodes,
        }
        all_episodes.extend(seed_episodes)

    env.close()

    return {
        "per_seed": per_seed,
        "overall":  compute_safety_summary(all_episodes),
    }


# ================ TABLEAU MARKDOWN ================

def build_markdown_table(results: dict, seeds: list[int]) -> str:
    """Génère le tableau de comparaison au format Markdown."""
    lines = [
        "# Safety-Aware Evaluation Results\n",
        "## Par seed\n",
        "| Agent | Seed | Reward Mean | Reward Std | Collision Rate | Mean Crashes | Safety Margin |",
        "|---|---|---|---|---|---|---|",
    ]

    for agent_name, agent_results in results.items():
        for seed in seeds:
            s = agent_results["per_seed"][str(seed)]["summary"]
            margin = f"{s['safety_margin']:.3f}" if s["safety_margin"] is not None else "N/A"
            lines.append(
                f"| {agent_name} | {seed} "
                f"| {s['reward_mean']:.2f} | {s['reward_std']:.2f} "
                f"| {s['collision_rate']:.2%} | {s['mean_crashes']:.3f} "
                f"| {margin} |"
            )

    lines += [
        "\n## Overall (toutes seeds confondues)\n",
        "| Agent | Reward Mean | Reward Std | Collision Rate | Mean Crashes | Safety Margin |",
        "|---|---|---|---|---|---|",
    ]

    for agent_name, agent_results in results.items():
        s = agent_results["overall"]
        margin = f"{s['safety_margin']:.3f}" if s["safety_margin"] is not None else "N/A"
        lines.append(
            f"| {agent_name} "
            f"| {s['reward_mean']:.2f} | {s['reward_std']:.2f} "
            f"| {s['collision_rate']:.2%} | {s['mean_crashes']:.3f} "
            f"| {margin} |"
        )

    return "\n".join(lines)


# ================ MAIN ================

def parse_seeds(text: str) -> list[int]:
    return [int(p.strip()) for p in text.split(",") if p.strip()]


def main():
    parser = argparse.ArgumentParser(description="Évaluation safety-aware — 3 agents")
    parser.add_argument("--seeds",      type=str, default="0,1,3")
    parser.add_argument("--episodes",   type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="comparison_results")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for agent_name, model_path in AGENTS.items():
        print(f"\n  Évaluation de {agent_name} ({model_path})...")
        if not Path(model_path).exists():
            print(f"  ⚠️  Checkpoint introuvable : {model_path} — agent ignoré.")
            continue
        all_results[agent_name] = evaluate_agent(model_path, seeds, args.episodes)
        overall = all_results[agent_name]["overall"]
        print(f"  ✅ {agent_name} | reward: {overall['reward_mean']:.2f} ± {overall['reward_std']:.2f} "
              f"| collision rate: {overall['collision_rate']:.2%}")

    # ---- Sauvegarde JSON ----
    stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"safety_aware_eval_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({"seeds": seeds, "episodes": args.episodes, "results": all_results}, f, indent=2)

    # ---- Sauvegarde tableau Markdown ----
    table_md   = build_markdown_table(all_results, seeds)
    table_path = os.path.join(args.output_dir, f"safety_aware_eval_{stamp}.md")
    with open(table_path, "w") as f:
        f.write(table_md)

    print(f"\n  JSON    → {json_path}")
    print(f"  Tableau → {table_path}")
    print("\n  Évaluation terminée.")


if __name__ == "__main__":
    main()