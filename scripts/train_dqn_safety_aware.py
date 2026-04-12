"""
Script d'entraînement pour l'agent DQN avec reward shaping safety-aware.
Entraîne deux variantes :
    - conservative : lambda=10, mu=50  (forte pénalité de sécurité)
    - moderate     : lambda=5,  mu=30  (pénalité équilibrée)

Usage :
    python train_dqn_safety_aware.py --variant conservative
    python train_dqn_safety_aware.py --variant moderate
    python train_dqn_safety_aware.py --variant all   # entraîne les deux
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from rlproject.highway_env_wrapper import HighwayV0Env
from rlproject.dqn_agent import DQNAgent
from rlproject.replay_buffer import ReplayBuffer
from rlproject.reward_shaper import apply_safety_shaping  # <-- la seule nouveauté


# ================ CONFIG ================
NUM_EPISODES = 500
MAX_STEPS = 300
SAVE_INTERVAL = 50

VARIANT_TO_SAVE_PATH = {
    "conservative": "checkpoints_safety_aware_conservative",
    "moderate":     "checkpoints_safety_aware_moderate",
}


# ================ FONCTION D'ENTRAÎNEMENT ================

def train(variant: str) -> None:
    """Lance un entraînement complet pour la variante donnée.

    Args:
        variant: "conservative" ou "moderate"
    """
    save_path = VARIANT_TO_SAVE_PATH[variant]
    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT — variante : {variant.upper()}")
    print(f"  Checkpoints → {save_path}/")
    print(f"{'='*60}\n")

    # ---- Environnement ----
    print("  Préparation de l'environnement...")
    env = HighwayV0Env()
    obs, _ = env.reset()

    state_dim  = np.array(obs).flatten().shape[0]
    action_dim = env.env.action_space.n
    print(f"  État dim: {state_dim}, Action dim: {action_dim}")

    # ---- Agent & buffer ----
    agent  = DQNAgent(state_dim, action_dim)
    buffer = ReplayBuffer(capacity=500_000)

    best_reward  = -float("inf")
    best_episode = 0

    # ---- Métriques ----
    # On ajoute crash_per_episode et shaped_rewards par rapport au script de base
    metrics = {
        "variant":              variant,
        "episodes":             [],
        "rewards":              [],          # récompense originale (highway-v0)
        "shaped_rewards":       [],          # récompense après shaping
        "losses":               [],
        "epsilons":             [],
        "avg_rewards_100":      [],
        "crash_per_episode":    [],          # 1 si au moins un crash, sinon 0
    }

    # ================ BOUCLE D'ENTRAÎNEMENT ================
    print(f"  Entraînement sur {NUM_EPISODES} épisodes...\n")

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = np.array(state).flatten()

        total_reward         = 0.0   # cumul récompense originale
        total_shaped_reward  = 0.0   # cumul récompense shapée (vue par l'agent)
        episode_losses       = []
        episode_crashed      = False

        for step in range(MAX_STEPS):
            action = agent.select_action(state, env.env.action_space)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.array(next_state).flatten()

            # --------------------------------------------------
            # REWARD SHAPING  ← seule différence avec train_dqn_custom.py
            # On transforme la récompense brute avant de la donner à l'agent.
            # apply_safety_shaping cherche "crashed" dans info automatiquement.
            # --------------------------------------------------
            shaped_reward = apply_safety_shaping(
                baseline_reward=reward,
                info=info,
                variant=variant,
            )

            # Détecter si un crash s'est produit ce step
            if info.get("crashed", False):
                episode_crashed = True

            # L'agent apprend avec la récompense shapée
            buffer.push(state, action, shaped_reward, next_state, done)
            loss = agent.train(buffer)
            if loss is not None:
                episode_losses.append(loss)

            state               = next_state
            total_reward        += reward          # on garde la trace de la vraie récompense
            total_shaped_reward += shaped_reward

            if done or truncated:
                break

        # ---- Logging de fin d'épisode ----
        metrics["episodes"].append(episode)
        metrics["rewards"].append(float(total_reward))
        metrics["shaped_rewards"].append(float(total_shaped_reward))
        metrics["crash_per_episode"].append(int(episode_crashed))

        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        metrics["losses"].append(avg_loss)

        agent.decay_epsilon()
        metrics["epsilons"].append(float(agent.epsilon))

        agent.step_scheduler()

        if len(metrics["rewards"]) >= 100:
            avg_100 = float(np.mean(metrics["rewards"][-100:]))
            metrics["avg_rewards_100"].append(avg_100)

        # ---- Meilleur modèle (critère : récompense originale) ----
        if total_reward > best_reward:
            best_reward  = total_reward
            best_episode = episode
            torch.save({
                "model_state_dict": agent.q_net.state_dict(),
                "epsilon":          agent.epsilon,
                "episode":          episode,
                "reward":           total_reward,
                "variant":          variant,
            }, os.path.join(save_path, "best_model.pth"))

        # ---- Affichage tous les 10 épisodes ----
        if (episode + 1) % 10 == 0:
            avg_reward        = np.mean(metrics["rewards"][-10:])
            avg_shaped_reward = np.mean(metrics["shaped_rewards"][-10:])
            avg_loss_10       = np.mean(metrics["losses"][-10:])
            crash_rate_10     = np.mean(metrics["crash_per_episode"][-10:]) * 100
            print(
                f"Ep {episode + 1:3d}/{NUM_EPISODES} | "
                f"Reward: {total_reward:6.2f} | Shaped: {total_shaped_reward:7.2f} | "
                f"Avg(10): {avg_reward:6.2f} | Loss: {avg_loss_10:.4f} | "
                f"ε: {agent.epsilon:.3f} | Crashes(10): {crash_rate_10:4.1f}% | "
                f"Buffer: {len(buffer):6d}"
            )

        # ---- Checkpoint périodique ----
        if (episode + 1) % SAVE_INTERVAL == 0:
            torch.save({
                "model_state_dict": agent.q_net.state_dict(),
                "epsilon":          agent.epsilon,
                "episode":          episode,
                "variant":          variant,
            }, os.path.join(save_path, f"checkpoint_ep{episode + 1}.pth"))
            print(f"  Checkpoint sauvegardé à l'épisode {episode + 1}")

    # ================ SAUVEGARDE FINALE ================
    torch.save({
        "model_state_dict": agent.q_net.state_dict(),
        "epsilon":          agent.epsilon,
        "variant":          variant,
    }, os.path.join(save_path, "last_model.pth"))

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Résumé ----
    crash_rate_global = np.mean(metrics["crash_per_episode"]) * 100
    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT TERMINÉ — {variant.upper()}")
    print(f"{'='*60}")
    print(f"  Épisodes            : {NUM_EPISODES}")
    print(f"  Meilleure récompense: {best_reward:.2f}  (épisode {best_episode + 1})")
    print(f"  Reward moy (100 der): {np.mean(metrics['rewards'][-100:]):.2f}")
    print(f"  Taux de crash global: {crash_rate_global:.1f}%")
    print(f"  Loss moy (10 der)   : {np.mean(metrics['losses'][-10:]):.4f}")
    print(f"  Modèles sauvegardés : {save_path}/")
    print(f"  Métriques           : {save_path}/metrics.json")
    print(f"{'='*60}\n")

    env.close()


# ================ POINT D'ENTRÉE ================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement DQN safety-aware")
    parser.add_argument(
        "--variant",
        choices=["conservative", "moderate", "all"],
        default="all",
        help="Variante de reward shaping à entraîner (défaut: all)",
    )
    args = parser.parse_args()

    variants_to_run = (
        ["conservative", "moderate"] if args.variant == "all" else [args.variant]
    )

    for v in variants_to_run:
        train(v)

    print("  Tous les entraînements sont terminés.")