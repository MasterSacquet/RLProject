# test_dqn_custom.py

"""
Script de test pour modèle DQN personnalisé entraîné
Permet de charger un modèle pré-entraîné et de l'évaluer
"""

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


# ================ CONFIG ================
MODEL_NAME = "last_model.pth"
SAVE_PATH = "checkpoints_custom"
NUM_TEST_EPISODES = 10
RENDER = True  # Afficher la visualisation

# ================ SETUP ================
os.makedirs(SAVE_PATH, exist_ok=True)

# Créer l'environnement
wrapper = HighwayV0Env(render_mode="human" if RENDER else None)
env = wrapper.env  #  Utiliser directement l'env Gymnasium

# Déterminer les dimensions
obs, _ = env.reset()
state_dim = np.array(obs).flatten().shape[0]
action_dim = env.env.action_space.n

# ================ CHARGER LE MODÈLE ================
model_path = os.path.join(SAVE_PATH, MODEL_NAME)

if os.path.exists(model_path):
    print(f"Chargement du modèle: {model_path}")
    agent = DQNAgent(state_dim, action_dim)
    checkpoint = torch.load(model_path, map_location=agent.device, weights_only=False)
    agent.q_net.load_state_dict(checkpoint["model_state_dict"])
    agent.epsilon = 0.0  # Désactiver exploration
else:
    print(f" Erreur: Modèle introuvable à {model_path}")
    exit(1)

# ================ TEST/ÉVALUATION ================
print(f"\nÉvaluation sur {NUM_TEST_EPISODES} épisodes...")

episode_rewards = []

for episode in range(NUM_TEST_EPISODES):
    obs, _ = env.reset()
    obs = np.array(obs)
    episode_reward = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        action = agent.select_action(obs, env.env.action_space)
        obs, reward, done, truncated, _ = env.step(action)
        obs = np.array(obs)
        episode_reward += reward
    
    episode_rewards.append(episode_reward)

mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)

print(f"\n Résultats:")
print(f"   Récompense moyenne: {mean_reward:.2f} ± {std_reward:.2f}")

# ================ VISUALISATION DÉTAILLÉE ================
print(f"\nExécutions détaillées (premiers 3 épisodes):")

for episode in range(3):
    obs, _ = env.reset()
    obs = np.array(obs)
    episode_reward = 0
    steps = 0
    
    done = False
    truncated = False
    
    while not (done or truncated):
        # Prédiction du modèle
        action = agent.select_action(obs, env.env.action_space)
        
        obs, reward, done, truncated, _ = env.step(action)
        obs = np.array(obs)
        
        episode_reward += reward
        steps += 1
        
        if RENDER:
            env.render()
    
    print(f"   Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={steps}")

env.close()
print("\n Test terminé!")