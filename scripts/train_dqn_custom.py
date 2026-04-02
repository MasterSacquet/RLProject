# train_dqn_custom.py

"""
Script d'entraînement pour l'agent DQN personnalisé
Avec sauvegarde des métriques
"""

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


# ================ CONFIG ================
NUM_EPISODES = 500  # Augmenté de 100 à 500 pour meilleure convergence
MAX_STEPS = 300
SAVE_PATH = "checkpoints_custom"
SAVE_INTERVAL = 50

os.makedirs(SAVE_PATH, exist_ok=True)

# ================ SETUP ================
print("🚀 Préparation de l'environnement...")
env = HighwayV0Env()
obs, _ = env.reset()

state_dim = np.array(obs).flatten().shape[0]
action_dim = env.env.action_space.n

print(f"État dim: {state_dim}, Action dim: {action_dim}")

agent = DQNAgent(state_dim, action_dim)
buffer = ReplayBuffer(capacity=500_000)  # Augmenté de 100k à 500k

best_reward = -float("inf")
best_episode = 0

# Suivi des métriques détaillées
metrics = {
    "episodes": [],
    "rewards": [],
    "losses": [],
    "epsilons": [],
    "avg_rewards_100": []
}

# ================ ENTRAÎNEMENT ================
print(f"\n📚 Entraînement sur {NUM_EPISODES} épisodes...")

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = np.array(state).flatten()
    total_reward = 0
    episode_losses = []

    for step in range(MAX_STEPS):
        action = agent.select_action(state, env.env.action_space)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = np.array(next_state).flatten()

        buffer.push(state, action, reward, next_state, done)
        loss = agent.train(buffer)
        if loss is not None:
            episode_losses.append(loss)

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    # Logging
    metrics["episodes"].append(episode)
    metrics["rewards"].append(float(total_reward))
    avg_loss = np.mean(episode_losses) if episode_losses else 0.0
    metrics["losses"].append(float(avg_loss))
    
    # Décrémenter epsilon UNE FOIS par épisode
    agent.decay_epsilon()
    metrics["epsilons"].append(float(agent.epsilon))
    
    # Étape du learning rate scheduler UNE FOIS par épisode
    agent.step_scheduler()
    
    # Moyenne mobile sur 100 épisodes
    if len(metrics["rewards"]) >= 100:
        avg_100 = np.mean(metrics["rewards"][-100:])
        metrics["avg_rewards_100"].append(float(avg_100))
    
    # Sauvegarde du meilleur modèle
    if total_reward > best_reward:
        best_reward = total_reward
        best_episode = episode
        torch.save({
            "model_state_dict": agent.q_net.state_dict(),
            "epsilon": agent.epsilon,
            "episode": episode,
            "reward": total_reward
        }, os.path.join(SAVE_PATH, "best_model.pth"))
    
    # Affichage des logs
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(metrics["rewards"][-10:])
        avg_loss = np.mean(metrics["losses"][-10:])
        print(f"Ep {episode + 1:3d}/{NUM_EPISODES} | Reward: {total_reward:6.2f} | "
              f"Avg(10): {avg_reward:6.2f} | Loss: {avg_loss:.4f} | "
              f"ε: {agent.epsilon:.3f} | Buffer: {len(buffer):6d}")
    
    # Sauvegarde périodique
    if (episode + 1) % SAVE_INTERVAL == 0:
        torch.save({
            "model_state_dict": agent.q_net.state_dict(),
            "epsilon": agent.epsilon,
            "episode": episode
        }, os.path.join(SAVE_PATH, f"checkpoint_ep{episode + 1}.pth"))
        print(f"💾 Checkpoint sauvegardé à l'épisode {episode + 1}")

# ================ SAUVEGARDE FINALE ================
torch.save({
    "model_state_dict": agent.q_net.state_dict(),
    "epsilon": agent.epsilon
}, os.path.join(SAVE_PATH, "last_model.pth"))

# Sauvegarder les métriques
with open(os.path.join(SAVE_PATH, "metrics.json"), 'w') as f:
    json.dump(metrics, f, indent=2)

# Afficher un résumé
print("\n" + "="*60)
print("✅ ENTRAÎNEMENT TERMINÉ")
print("="*60)
print(f"📊 Épisodes: {NUM_EPISODES}")
print(f"🏆 Meilleure récompense: {best_reward:.2f} (épisode {best_episode + 1})")
print(f"📈 Récompense moyenne (derniers 100): {np.mean(metrics['rewards'][-100:]):.2f}")
print(f"📉 Loss moyenne (derniers 10): {np.mean(metrics['losses'][-10:]):.4f}")
print(f"💾 Modèles sauvegardés dans: {SAVE_PATH}/")
print(f"📋 Métriques sauvegardées dans: {SAVE_PATH}/metrics.json")
print("="*60)

print(f"\n✅ Modèle sauvegardé: {SAVE_PATH}/best_model.pth")
print(f"📊 Métriques sauvegardées: {SAVE_PATH}/metrics.json")
print(f"🏅 Meilleure récompense: {best_reward:.2f}")

env.close()
print("🏁 Entraînement terminé!")
