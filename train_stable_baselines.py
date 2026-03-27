# train_stable_baselines.py

"""
Script d'entraînement pour Stable-Baselines3 DQN
"""

import os
import json
import time  # ⬆️ Ajouté pour mesurer le temps d'entraînement
import numpy as np
from highway_env_wrapper import HighwayV0Env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


# ================ CONFIG ================
TOTAL_TIMESTEPS = 10_000  # Entraînement court
MODEL_NAME = "dqn_highway"
SAVE_PATH = "checkpoints_sb3"
SAVE_INTERVAL = 10_000  # Sauvegarde à la fin

os.makedirs(SAVE_PATH, exist_ok=True)

# ================ CALLBACK PERSONNALISÉ ================
class MetricsCallback(BaseCallback):
    """Callback pour sauvegarder les métriques pendant l'entraînement"""
    
    def __init__(self, save_path, save_interval=500):
        super().__init__()
        self.save_path = save_path
        self.save_interval = save_interval
        self.metrics = {
            "timesteps": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": []
        }
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Accéder à l'infos d'épisode si disponible
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                episode_info = info["episode"]
                self.metrics["timesteps"].append(self.num_timesteps)
                self.metrics["episode_rewards"].append(float(episode_info["r"]))
                self.metrics["episode_lengths"].append(int(episode_info["l"]))
                self.episode_count += 1
        
        # Sauvegarder périodiquement
        if self.n_calls % self.save_interval == 0:
            self.save_metrics()
        
        return True
    
    def save_metrics(self):
        path = os.path.join(self.save_path, "metrics.json")
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


# ================ SETUP ================
print("🚀 Préparation de l'environnement...")
wrapper = HighwayV0Env()
env = wrapper.env

# ================ CRÉER LE MODÈLE ================
print("🤖 Création du modèle DQN Stable-Baselines3...")
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.001,  # Augmenté pour apprendre vite sur peu de steps
    buffer_size=10_000,  # Réduit pour 10k timesteps
    batch_size=32,  # Réduit pour court entraînement
    exploration_fraction=0.2,  # 2000 steps d'exploration (20% de 10k)
    exploration_initial_eps=1.0,  # Début aléatoire complet
    exploration_final_eps=0.05,  # 5% d'exploration à la fin
    gamma=0.99,  # Discount factor standard
    target_update_interval=1000,  # Réduit pour stabilité court terme
    tau=0.001,  # Soft update factor standard
    train_freq=2,  # Update très souvent pour apprendre plus
    gradient_steps=2,  # Réduit pour vitesse
    max_grad_norm=10,  # Clipping standard
    learning_starts=1000,  # 1k steps = 10% de 10k timesteps
    verbose=1
)

# ================ ENTRAÎNEMENT ================
# ================ ENTRAÎNEMENT ================
print(f"\n📚 Entraînement sur {TOTAL_TIMESTEPS} timesteps...")
print("⏱️ Configuration optimisée pour apprentissage rapide et stable...\n")

metrics_callback = MetricsCallback(SAVE_PATH, SAVE_INTERVAL)

# Callback d'évaluation périodique
eval_env = HighwayV0Env().env
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_PATH,
    log_path=SAVE_PATH,
    eval_freq=5000,        # Réduit pour au moins 2 évals
    n_eval_episodes=2,     # Réduit pour vitesse
    deterministic=True,
    render=False
)

# Mesurer le temps d'entraînement
start_time = time.time()

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[metrics_callback, eval_callback],
    progress_bar=True,
    log_interval=50  # Afficher le log tous les 50 updates
)

elapsed_time = time.time() - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\n✅ Entraînement complété en {int(hours)}h {int(minutes)}m {int(seconds)}s")

# ================ SAUVEGARDE ================
model_path = os.path.join(SAVE_PATH, MODEL_NAME)
model.save(model_path)
print(f"\n✅ Modèle sauvegardé: {model_path}.zip")

# Sauvegarder les métriques
metrics_callback.save_metrics()
print(f"📊 Métriques sauvegardées: {SAVE_PATH}/metrics.json")

eval_env.close()
env.close()
print("🏁 Entraînement terminé!")
