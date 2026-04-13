# Projet Reinforcement Learning - Agents DQN sur Highway-v0

Comparaison entre une implémentation personnalisée de DQN et la bibliothèque
Stable Baselines3 sur l'environnement highway-v0, avec une extension orientée
sécurité par reward shaping.

Des résumés détaillés sont disponibles dans :
- [`analysis/CORE_TASK_CustomDQN_vs_StableBaselines.md`](analysis/CORE_TASK_CustomDQN_vs_StableBaselines.md) pour le core task
- [`analysis/EXTENSION_TASK_safety_aware.md`](analysis/EXTENSION_TASK_safety_aware.md) pour l'extension

## Structure du dépôt

```
src/rlproject/
├── dqn_agent.py                    # Agent DQN custom (Dueling + Double DQN)
├── replay_buffer.py                # Buffer d'expérience
├── highway_env_wrapper.py          # Wrapper highway-v0
├── shared_core_config.py           # Configuration partagée
├── reward_shaper.py                # Reward shaping safety-aware (extension)
└── safety_metrics.py               # Métriques de sécurité (extension)

scripts/
├── train_dqn_custom.py             # Entraînement DQN custom
├── train_stable_baselines.py       # Entraînement SB3
├── train_dqn_safety_aware.py       # Entraînement variantes safety-aware
├── evaluate_multiseed.py           # Évaluation multi-seed (custom vs SB3)
├── evaluate_safety_aware.py        # Évaluation safety-aware
├── plot_training_curves.py         # Courbes d'entraînement
├── plot_safety_aware.py            # Courbes extension
├── compare_agents.py               # Comparaison custom vs SB3
└── record_rollout.py               # Enregistrement des rollouts

checkpoints_custom/                 # Sauvegardes DQN custom
checkpoints_sb3/                    # Sauvegardes SB3
checkpoints_safety_aware_conservative/  # Sauvegardes variante conservative
checkpoints_safety_aware_moderate/      # Sauvegardes variante moderate

comparison_results/                 # Résultats d'évaluation et figures
analysis/                           # Documents d'analyse
├── CORE_TASK_CustomDQN_vs_StableBaselines.md
└── EXTENSION_TASK_safety_aware.md

rollouts/                           # Vidéos enregistrées
task.md                             # Répartition des tâches entre membres du groupe
requirements.txt
README.md
```

## Installation

```bash
pip install torch gymnasium highway-env stable-baselines3 numpy
```

## Utilisation

### Core Task — DQN custom vs SB3

```bash
# Entraînement
python scripts/train_dqn_custom.py
python scripts/train_stable_baselines.py

# Évaluation multi-seed
python scripts/evaluate_multiseed.py --seeds 0,1,3 --episodes 50

# Enregistrement d'un rollout
python scripts/record_rollout.py --agent custom \
    --custom-model checkpoints_custom/last_model.pth --episodes 3 --seed 0
python scripts/record_rollout.py --agent sb3 \
    --sb3-model checkpoints_sb3/dqn_highway.zip --episodes 3 --seed 0
```

### Extension — Safety-Aware Reward Shaping

```bash
# Entraînement des variantes
python scripts/train_dqn_safety_aware.py --variant conservative
python scripts/train_dqn_safety_aware.py --variant moderate
python scripts/train_dqn_safety_aware.py --variant all

# Évaluation des trois agents
python scripts/evaluate_safety_aware.py --seeds 0,1,3 --episodes 50
```


## Résultats principaux

### Core Task

| Seed | Custom Mean | Custom Std | SB3 Mean | SB3 Std |
|---|---|---|---|---|
| 0 | 18.76 | 5.07 | 10.24 | 5.66 |
| 1 | 19.37 | 4.47 | 11.28 | 5.94 |
| 3 | 16.94 | 6.47 | 10.59 | 6.80 |
| Overall | **18.36** | 5.50 | 10.70 | 6.17 |

Le DQN custom surpasse SB3 sur toutes les seeds évaluées.

### Extension — Safety-Aware Reward Shaping

| Agent | Reward Mean | Reward Std | Collision Rate |
|---|---|---|---|
| Baseline | 18.36 | 5.50 | 21.33% |
| Conservative (λ=10, μ=50) | 18.06 | 6.28 | 22.00% |
| Moderate (λ=5, μ=30) | **19.54** | **4.40** | **14.67%** |

L'agent Moderate obtient le meilleur compromis sécurité/performance,
avec une réduction de 31% du taux de collision et une amélioration
simultanée du reward moyen par rapport à la baseline.


## Dépendances

- `torch` — deep learning
- `gymnasium` — interface RL
- `highway-env` — environnement de conduite
- `stable_baselines3` — algorithmes RL optimisés
- `numpy` — calcul numérique
