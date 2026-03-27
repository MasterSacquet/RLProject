# test_stable_baselines.py

"""
Script de test pour modèle entraîné avec Stable-Baselines3
Permet de charger un modèle pré-entraîné et de l'évaluer
"""

import numpy as np
from highway_env_wrapper import HighwayV0Env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import os


# ================ CONFIG ================
MODEL_NAME = "dqn_highway"
SAVE_PATH = "checkpoints_sb3"
NUM_TEST_EPISODES = 10
RENDER = True  # Afficher la visualisation

# ================ SETUP ================
os.makedirs(SAVE_PATH, exist_ok=True)

# Créer l'environnement
wrapper = HighwayV0Env(render_mode="human" if RENDER else None)
env = wrapper.env  # ✅ Utiliser directement l'env Gymnasium

model_path = os.path.join(SAVE_PATH, MODEL_NAME)

# ================ CHARGER OU CRÉER LE MODÈLE ================
if os.path.exists(f"{model_path}.zip"):
    print(f"📦 Chargement du modèle: {model_path}")
    model = DQN.load(model_path, env=env)
else:
    print("⚠️  Aucun modèle trouvé. Création d'un nouveau modèle DQN...")
    print("   Pour un vrai test, utilisez un modèle entraîné!")
    
    # Créer un nouveau modèle (à entraîner)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100_000,
        batch_size=64,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        gamma=0.99,
        target_update_interval=1000,
        verbose=1
    )
    
    print("🚀 Entraînement rapide sur 5000 steps...")
    model.learn(total_timesteps=5000)
    model.save(model_path)
    print(f"✅ Modèle sauvegardé: {model_path}")

# ================ TEST/ÉVALUATION ================
print(f"\n🧪 Évaluation sur {NUM_TEST_EPISODES} épisodes...")

mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=NUM_TEST_EPISODES,
    deterministic=True,  # Pas d'exploration, exploitation pure
    render=RENDER
)

print(f"\n📊 Résultats:")
print(f"   Récompense moyenne: {mean_reward:.2f} ± {std_reward:.2f}")

# ================ VISUALISATION DÉTAILLÉE ================
print(f"\n🎬 Exécutions détaillées (premiers 3 épisodes):")

for episode in range(3):
    obs, _ = env.reset()
    obs = np.array(obs)
    episode_reward = 0
    steps = 0
    
    done = False
    truncated = False
    
    while not (done or truncated):
        # Prédiction du modèle
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        obs = np.array(obs)
        
        episode_reward += reward
        steps += 1
        
        if RENDER:
            env.render()
    
    print(f"   Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={steps}")

env.close()
print("\n✅ Test terminé!")
