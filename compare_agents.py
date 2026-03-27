# compare_agents.py

"""
Script de comparaison entre l'agent DQN personnalisé et Stable-Baselines3
Évalue les deux sur 50 épisodes avec statistiques
"""

import torch
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from highway_env_wrapper import HighwayV0Env
from dqn_agent import DQNAgent
from stable_baselines3 import DQN as DQN_SB3


# ================ CONFIG ================
NUM_EVAL_EPISODES = 50
CUSTOM_MODEL_PATH = "checkpoints_custom/last_model.pth"
SB3_MODEL_PATH = "checkpoints_sb3/dqn_highway.zip"
RESULTS_PATH = "comparison_results"

os.makedirs(RESULTS_PATH, exist_ok=True)

# ================ RÉSULTATS ================
results = {
    "timestamp": datetime.now().isoformat(),
    "num_episodes": NUM_EVAL_EPISODES,
    "custom_dqn": {
        "rewards": [],
        "episode_lengths": [],
        "stats": {}
    },
    "stable_baselines": {
        "rewards": [],
        "episode_lengths": [],
        "stats": {}
    }
}


# ================ FONCTION D'ÉVALUATION ================
def evaluate_agent(agent, env, num_episodes, agent_name="Agent"):
    """Évalue un agent sur plusieurs épisodes"""
    if agent_name == "Custom DQN":
        is_custom = True
    else:
        is_custom = False
    
    rewards = []
    lengths = []
    
    print(f"\n🧪 Évaluation {agent_name} ({num_episodes} épisodes)...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        if is_custom:
            obs = np.array(obs).flatten()
        
        episode_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            if is_custom:
                # DQN personnalisé
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                    q_values = agent.q_net(state_tensor)
                    action = q_values.argmax(dim=1).item()
            else:
                # Stable-Baselines3
                action, _ = agent.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, _ = env.step(action)
            if is_custom:
                obs = np.array(obs).flatten()
            
            episode_reward += reward
            steps += 1
        
        rewards.append(episode_reward)
        lengths.append(steps)
        
        if (episode + 1) % 10 == 0:
            current_mean = np.mean(rewards[-10:])
            print(f"  ├─ Épisodes {episode + 1 - 9:2d}-{episode + 1:2d}: Moyenne = {current_mean:7.2f}")
    
    return rewards, lengths


# ================ SETUP ================
print("=" * 60)
print("COMPARAISON DES DEUX APPROCHES DQN")
print("=" * 60)

wrapper = HighwayV0Env()
env = wrapper.env

state_dim = 50  # 10 véhicules × 5 features
action_dim = env.action_space.n

# ================ CHARGER AGENT CUSTOM ================
print("\n📦 Chargement Agent Custom DQN...")
if os.path.exists(CUSTOM_MODEL_PATH):
    custom_agent = DQNAgent(state_dim, action_dim)
    checkpoint = torch.load(CUSTOM_MODEL_PATH, map_location=custom_agent.device, weights_only=False)
    custom_agent.q_net.load_state_dict(checkpoint["model_state_dict"])
    custom_agent.q_net.eval()  # Mode évaluation (aucune adaptation batch norm)
    custom_agent.epsilon = 0.0  # Pas d'exploration
    
    custom_rewards, custom_lengths = evaluate_agent(
        custom_agent, env, NUM_EVAL_EPISODES, "Custom DQN"
    )
    results["custom_dqn"]["rewards"] = custom_rewards
    results["custom_dqn"]["episode_lengths"] = custom_lengths
    print("   ✅ Agent Custom DQN chargé")
else:
    print(f"   ❌ Modèle non trouvé: {CUSTOM_MODEL_PATH}")
    custom_rewards = None

# ================ CHARGER AGENT STABLE-BASELINES ================
print("\n📦 Chargement Agent Stable-Baselines3...")
if os.path.exists(SB3_MODEL_PATH):
    sb3_agent = DQN_SB3.load(SB3_MODEL_PATH, env=env)
    
    sb3_rewards, sb3_lengths = evaluate_agent(
        sb3_agent, env, NUM_EVAL_EPISODES, "Stable-Baselines3"
    )
    results["stable_baselines"]["rewards"] = sb3_rewards
    results["stable_baselines"]["episode_lengths"] = sb3_lengths
    print("   ✅ Agent Stable-Baselines3 chargé")
else:
    print(f"   ⚠️  Modèle non trouvé: {SB3_MODEL_PATH}")
    sb3_rewards = None

env.close()

# ================ STATISTIQUES ================
print("\n" + "=" * 60)
print("📊 RÉSULTATS ET STATISTIQUES")
print("=" * 60)

if custom_rewards:
    custom_mean = np.mean(custom_rewards)
    custom_std = np.std(custom_rewards)
    custom_min = np.min(custom_rewards)
    custom_max = np.max(custom_rewards)
    
    results["custom_dqn"]["stats"] = {
        "mean": float(custom_mean),
        "std": float(custom_std),
        "min": float(custom_min),
        "max": float(custom_max),
        "median": float(np.median(custom_rewards))
    }
    
    print("\n🤖 Agent Custom DQN:")
    print(f"   ├─ Moyenne:  {custom_mean:7.2f}")
    print(f"   ├─ Std Dev:  {custom_std:7.2f}")
    print(f"   ├─ Min:      {custom_min:7.2f}")
    print(f"   ├─ Max:      {custom_max:7.2f}")
    print(f"   └─ Médiane:  {np.median(custom_rewards):7.2f}")

if sb3_rewards:
    sb3_mean = np.mean(sb3_rewards)
    sb3_std = np.std(sb3_rewards)
    sb3_min = np.min(sb3_rewards)
    sb3_max = np.max(sb3_rewards)
    
    results["stable_baselines"]["stats"] = {
        "mean": float(sb3_mean),
        "std": float(sb3_std),
        "min": float(sb3_min),
        "max": float(sb3_max),
        "median": float(np.median(sb3_rewards))
    }
    
    print("\n🚀 Agent Stable-Baselines3:")
    print(f"   ├─ Moyenne:  {sb3_mean:7.2f}")
    print(f"   ├─ Std Dev:  {sb3_std:7.2f}")
    print(f"   ├─ Min:      {sb3_min:7.2f}")
    print(f"   ├─ Max:      {sb3_max:7.2f}")
    print(f"   └─ Médiane:  {np.median(sb3_rewards):7.2f}")

# ================ COMPARAISON ================
if custom_rewards and sb3_rewards:
    print("\n⚖️  COMPARAISON:")
    diff = sb3_mean - custom_mean
    pct_diff = (diff / abs(custom_mean)) * 100 if custom_mean != 0 else 0
    
    if diff > 0:
        print(f"   Stable-Baselines3 est {abs(diff):6.2f} points ({abs(pct_diff):.1f}%) MEILLEUR")
    elif diff < 0:
        print(f"   Custom DQN est {abs(diff):6.2f} points ({abs(pct_diff):.1f}%) MEILLEUR")
    else:
        print(f"   Les deux agents sont équivalents")
    
    results["comparison"] = {
        "difference": float(diff),
        "percent_difference": float(pct_diff),
        "winner": "Stable-Baselines3" if diff > 0 else "Custom DQN" if diff < 0 else "Tie"
    }

# ================ SAUVEGARDE ================
results_file = os.path.join(RESULTS_PATH, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Résultats sauvegardés: {results_file}")

# ================ VISUALISATION ================
if custom_rewards and sb3_rewards:
    print("\n📈 Génération des graphiques...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparaison DQN Custom vs Stable-Baselines3", fontsize=16, fontweight='bold')
    
    # Graphique 1: Distribution des récompenses
    axes[0, 0].hist(custom_rewards, alpha=0.7, label="Custom DQN", bins=15)
    axes[0, 0].hist(sb3_rewards, alpha=0.7, label="Stable-Baselines3", bins=15)
    axes[0, 0].set_xlabel("Récompense")
    axes[0, 0].set_ylabel("Fréquence")
    axes[0, 0].set_title("Distribution des Récompenses")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Graphique 2: Box plot
    axes[0, 1].boxplot([custom_rewards, sb3_rewards], 
                        labels=["Custom DQN", "Stable-Baselines3"])
    axes[0, 1].set_ylabel("Récompense")
    axes[0, 1].set_title("Box Plot - Comparaison")
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Graphique 3: Récompenses par épisode
    axes[1, 0].plot(custom_rewards, label="Custom DQN", alpha=0.7)
    axes[1, 0].plot(sb3_rewards, label="Stable-Baselines3", alpha=0.7)
    axes[1, 0].axhline(custom_mean, color='blue', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(sb3_mean, color='orange', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel("Épisode")
    axes[1, 0].set_ylabel("Récompense")
    axes[1, 0].set_title("Récompenses par Épisode")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Graphique 4: Moyenne mobile
    window = 10
    custom_moving_avg = np.convolve(custom_rewards, np.ones(window)/window, mode='valid')
    sb3_moving_avg = np.convolve(sb3_rewards, np.ones(window)/window, mode='valid')
    
    axes[1, 1].plot(custom_moving_avg, label="Custom DQN (MA10)", linewidth=2)
    axes[1, 1].plot(sb3_moving_avg, label="Stable-Baselines3 (MA10)", linewidth=2)
    axes[1, 1].set_xlabel("Épisode")
    axes[1, 1].set_ylabel("Récompense (Moyenne Mobile)")
    axes[1, 1].set_title(f"Moyenne Mobile ({window} épisodes)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_PATH, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Graphique sauvegardé: {plot_path}")
    plt.close()

print("\n✅ Comparaison terminée!")
print("=" * 60)
