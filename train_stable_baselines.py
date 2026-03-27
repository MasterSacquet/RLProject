# train_stable_baselines.py

"""
Script d'entraînement pour Stable-Baselines3 DQN
"""

import os
import numpy as np
from highway_env_wrapper import HighwayV0Env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import json


# ================ CONFIG ================
TOTAL_TIMESTEPS = 20_000  # Entraînement rapide (équivalent ~100 épisodes)
MODEL_NAME = "dqn_highway"
SAVE_PATH = "checkpoints_sb3"
SAVE_INTERVAL = 20_000

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
    learning_rate=1e-4,  # Taux d'apprentissage stable
    buffer_size=50_000,  # Réduit pour accélération (20k timesteps)
    batch_size=32,  # Taille standard
    exploration_fraction=0.1,  # 2k timesteps d'exploration décroissante
    exploration_initial_eps=1.0,  # Début aléatoire complet
    exploration_final_eps=0.05,  # 5% d'exploration à la fin
    gamma=0.99,  # Discount factor long-terme
    target_update_interval=2_000,  # Update réseau cible plus souvent (moins d'instabilité sur periodo court)
    tau=0.001,  # Soft update factor
    train_freq=8,  # Mise à jour tous les 8 steps (2x moins de mises à jour = 2x plus rapide)
    gradient_steps=1,  # 1 seule étape de gradient (4x plus rapide)
    max_grad_norm=10,  # Clipping de gradient
    verbose=1
)

# ================ ENTRAÎNEMENT ================
print(f"\n📚 Entraînement sur {TOTAL_TIMESTEPS} timesteps...")
metrics_callback = MetricsCallback(SAVE_PATH, SAVE_INTERVAL)

# Callback d'évaluation périodique
eval_env = HighwayV0Env().env
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_PATH,
    log_path=SAVE_PATH,
    eval_freq=20_000,      # Évaluation une seule fois à la fin
    n_eval_episodes=1,     # 1 seul épisode (plus rapide)
    deterministic=True,
    render=False
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[metrics_callback, eval_callback],  # Utiliser les deux callbacks
    progress_bar=True
)

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
