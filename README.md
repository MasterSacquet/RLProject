# 🚀 Guide Complet - Entraînement et Comparaison des Agents DQN

## 📋 Vue d'ensemble

Ce projet implémente et compare deux approches DQN pour résoudre le problème Highway-v0:
- **Custom DQN**: Implémentation personnalisée avec Double DQN
- **Stable-Baselines3 DQN**: Utilisation de la bibliothèque standard

---

## 📁 Structure des fichiers

```
Projet/
├── shared_core_config.py        # Configuration partagée (NE PAS MODIFIER)
├── highway_env_wrapper.py       # Wrapper pour l'environnement
├── dqn_agent.py                 # Agent DQN personnalisé
├── replay_buffer.py             # Buffer de replay
│
├── train.py                     # Entraîner Custom DQN
├── train_dqn_custom.py          # Entraîner Custom DQN (avec métriques)
├── train_stable_baselines.py    # Entraîner Stable-Baselines3 DQN
│
├── main.py                      # Tester Custom DQN
├── test_stable_baselines.py     # Tester Stable-Baselines3 DQN
├── compare_agents.py            # Comparer les deux agents
│
├── checkpoints_custom/          # Modèles Custom DQN
│   ├── best_model.pth
│   ├── metrics.json
│   └── ...
│
├── checkpoints_sb3/             # Modèles Stable-Baselines3
│   ├── dqn_highway.zip
│   └── metrics.json
│
└── comparison_results/          # Résultats de la comparaison
    ├── comparison_*.json        # Statistiques détaillées
    └── comparison_*.png         # Graphiques
```

---

## 🚀 Workflow Complet

### **Étape 1: Entraîner Custom DQN** (recommandé: ~10 minutes)

```bash
python train_dqn_custom.py
```

**Sortie attendue:**
```
...
Episode 100/500 | Avg Reward (10): 15.42 | Epsilon: 0.368
Episode 200/500 | Avg Reward (10): 22.34 | Epsilon: 0.135
...
✅ Modèle sauvegardé: checkpoints_custom/best_model.pth
🏅 Meilleure récompense: 28.45
```

**Fichiers générés:**
- `checkpoints_custom/best_model.pth` - Meilleur modèle
- `checkpoints_custom/last_model.pth` - Modèle final
- `checkpoints_custom/metrics.json` - Métriques d'entraînement

---

### **Étape 2: Entraîner Stable-Baselines3 DQN** (recommandé: ~15 minutes)

```bash
python train_stable_baselines.py
```

**Sortie attendue:**
```
Using cpu device
Mean Action Q-Value: 5.234 | Std Q-Value: 1.025 | Weight Update: 42
...
✅ Modèle sauvegardé: checkpoints_sb3/dqn_highway.zip
```

**Fichiers générés:**
- `checkpoints_sb3/dqn_highway.zip` - Modèle entraîné
- `checkpoints_sb3/metrics.json` - Métriques

---

### **Étape 3: Tester les modèles individuellement** (optionnel)

**Tester Custom DQN:**
```bash
python main.py
```

**Tester Stable-Baselines3:**
```bash
python test_stable_baselines.py
```

---

### **Étape 4: Comparer les deux approches** ⭐ RECOMMANDÉ

```bash
python compare_agents.py
```

**Sortie complète:**
```
============================================================
COMPARAISON DES DEUX APPROCHES DQN
============================================================

🧪 Évaluation Custom DQN (50 épisodes)...
  ├─ Épisodes 01-10: Moyenne =   18.45
  ├─ Épisodes 11-20: Moyenne =   21.34
  ...

🧪 Évaluation Stable-Baselines3 (50 épisodes)...
  ├─ Épisodes 01-10: Moyenne =   20.12
  ...

============================================================
📊 RÉSULTATS ET STATISTIQUES
============================================================

🤖 Agent Custom DQN:
   ├─ Moyenne:   20.45
   ├─ Std Dev:    6.34
   ├─ Min:        5.21
   ├─ Max:       32.18
   └─ Médiane:   21.45

🚀 Agent Stable-Baselines3:
   ├─ Moyenne:   22.67
   ├─ Std Dev:    5.89
   ├─ Min:        8.34
   ├─ Max:       35.24
   └─ Médiane:   23.12

⚖️  COMPARAISON:
   Stable-Baselines3 est  2.22 points (10.8%) MEILLEUR

💾 Résultats sauvegardés: comparison_results/comparison_20240324_143022.json
📊 Graphique sauvegardé: comparison_results/comparison_20240324_143022.png
```

**Fichiers générés:**
- `comparison_results/comparison_*.json` - Statistiques détaillées
- `comparison_results/comparison_*.png` - 4 graphiques de comparaison

---

## 📊 Analyse des Résultats

### **Fichier JSON de comparaison:**
```json
{
  "custom_dqn": {
    "stats": {
      "mean": 20.45,
      "std": 6.34,
      "min": 5.21,
      "max": 32.18,
      "median": 21.45
    }
  },
  "stable_baselines": {
    "stats": {
      "mean": 22.67,
      "std": 5.89,
      "min": 8.34,
      "max": 35.24,
      "median": 23.12
    }
  },
  "comparison": {
    "difference": 2.22,
    "percent_difference": 10.8,
    "winner": "Stable-Baselines3"
  }
}
```

### **4 Graphiques générés:**

1. **Distribution des Récompenses** - Histogrammes comparatifs
2. **Box Plot** - Visualisation des quartiles
3. **Récompenses par Épisode** - Séries temporelles brutes
4. **Moyenne Mobile (10 épisodes)** - Tendance lissée

---

## 🔧 Configuration des Hyperparamètres

### **Custom DQN** (`train_dqn_custom.py`):
```python
NUM_EPISODES = 500          # Nombre d'épisodes d'entraînement
MAX_STEPS = 300             # Max steps par épisode
learning_rate = 1e-3        # Taux d'apprentissage
gamma = 0.99                # Facteur de discount
epsilon_decay = 0.995       # Décroissance epsilon
```

### **Stable-Baselines3** (`train_stable_baselines.py`):
```python
TOTAL_TIMESTEPS = 100_000   # Total timesteps
learning_rate = 1e-3        # Taux d'apprentissage
buffer_size = 100_000       # Taille du replay buffer
batch_size = 64             # Batch size
exploration_fraction = 0.1  # Fraction d'exploration
```

---

## 📈 Interprétation des Métriques

| Métrique | Signification | Idéal |
|----------|---------------|-------|
| **Moyenne** | Récompense moyenne sur 50 épisodes | Plus haut ✓ |
| **Std Dev** | Variance des récompenses | Plus bas ✓ |
| **Min/Max** | Bornes des récompenses | Max haut, Min stable |
| **Médiane** | Valeur centrale (robuste aux outliers) | Élevée |

---

## 🎯 Cas d'usage

### Scénario 1: **Entraînement et comparaison rapides** (30 min)
```bash
python train_dqn_custom.py     # ~10 min
python train_stable_baselines.py  # ~15 min
python compare_agents.py        # ~5 min
```

### Scénario 2: **Comparer avec modèles existants** (5 min)
```bash
python compare_agents.py        # Charge les meilleurs modèles auto
```

### Scénario 3: **Tester un seul agent** (2 min)
```bash
python main.py                  # ou python test_stable_baselines.py
```

---

## ⚠️ Dépendances Requises

```bash
pip install torch numpy gymnasium highway-env stable-baselines3 matplotlib
```

---

## 🐛 Troubleshooting

### **"Modèle non trouvé"**
→ Lancez d'abord l'entraînement (`train_dqn_custom.py` ou `train_stable_baselines.py`)

### **Erreur CUDA/GPU**
→ Les scripts détectent automatiquement CPU/GPU. Pas d'action requise.

### **Plots matplotlib non affichés**
→ Les graphiques sont sauvegardés dans `comparison_results/`. Ouvrez-les avec un viewer image.

---

## 📝 Notes

- Le fichier `shared_core_config.py` définit les règles du jeu (immuable)
- Les métriques incluent **moyenne + écart-type** sur les 50 runs
- La comparaison est **déterministe** (pas d'exploration, epsilon=0)
- Les graphiques utilisent une **moyenne mobile sur 10 épisodes** pour lisser les récompenses

---

## 🎓 Concepts de renforcement (RL)

### **DQN (Deep Q-Network)**
- Approche hors-politique (off-policy)
- Utilise un réseau de neurones pour approximer Q(s,a)
- Replay buffer + Target network pour stabilité

### **Double DQN** (Custom implementation)
- Réduit la surestimation des Q-values
- Sélection d'action avec main network
- Évaluation avec target network

### **Epsilon-Greedy**
- Équilibre exploration (aléatoire) / exploitation (meilleure action)
- Epsilon décroît pendant l'entraînement (exploitation augmente)

---

**Bon entraînement! 🚀**