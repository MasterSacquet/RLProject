# Task Definition — RL Project

**Projet** : DQN vs Stable-Baselines Comparison + Safety-Aware Extension


## PART 1 : CORE TASK (Baseline Comparison)

### 1.1 Implémentation du DQN custom
- [x] **Agent DQN custom** — FINI (Simon)
  - Architecture Dueling DQN + Double DQN avec target network
  - Replay buffer
  - Fichier : [src/rlproject/dqn_agent.py](src/rlproject/dqn_agent.py)

### 1.2 Infrastructure d'entraînement
- [x] **Configuration partagée** — FINI (Simon)
  - Environnement : highway-v0, Observation : Kinematics, Actions : DiscreteMetaAction
  - Fichier : [src/rlproject/shared_core_config.py](src/rlproject/shared_core_config.py)

- [x] **Scripts d'entraînement** — FINI (Simon)
  - [scripts/train_dqn_custom.py](scripts/train_dqn_custom.py)
  - [scripts/train_stable_baselines.py](scripts/train_stable_baselines.py)
  - Checkpoints : `checkpoints_custom/` et `checkpoints_sb3/`

### 1.3 Évaluation quantitative (50 épisodes par seed)
- [x] **Évaluation multi-seed** — FINI (Juliette)
  - Seeds : 0, 1, 3 — 50 épisodes par seed
  - Script : [scripts/evaluate_multiseed.py](scripts/evaluate_multiseed.py)
  - Résultats : `comparison_results/multiseed_eval_20260402_164711.md`

| Seed | Custom Mean | Custom Std | SB3 Mean | SB3 Std |
|------|-------------|-----------|----------|---------|
| 0 | 18.76 | 5.07 | 10.24 | 5.66 |
| 1 | 19.37 | 4.47 | 11.28 | 5.94 |
| 3 | 16.94 | 6.47 | 10.59 | 6.80 |
| Overall | 18.36 | 5.50 | 10.70 | 6.17 |

-[x] **Analyse des résultats** — FINI (Juliette)
  - Fichier : [Analyse des résultats quantitatifs](analysis/CORE_TASK_CustomDQN_vs_StableBaselines.md#5-analyse-des-résultats-quantitatifs)

### 1.4 Courbes d'entraînement
- [x] **Génération des courbes** — FINI (Simon + Juliette)
  - Script : [scripts/plot_training_curves.py](scripts/plot_training_curves.py)
  - Résultats : `comparison_results/training_curves_20260402_165013.png`

### 1.5 Analyse qualitative
- [x] **Analyse comportementale** — FINI (Juliette)
  - Fichier : [Partie 6.1 : Analyse globale des résultats qualitatifs dans analysis/CORE_TASK_CustomDQN_vs_StableBaselines.md](analysis/CORE_TASK_CustomDQN_vs_StableBaselines.md#6-analyse-des-résultats-qualitatifs-rollouts)

- [x] **Rollouts enregistrés** — FINI (Juliette)
  - Script : [scripts/record_rollout.py](scripts/record_rollout.py)
  - Seeds : 0, 1, 3 — Vidéos dans `rollouts/`

- [x] **Identification et analyse des failure modes** — FINI (Juliette)
  - Fichier : [Partie 6.2 : Exemples de failure modes dans analysis/CORE_TASK_CustomDQN_vs_StableBaselines.md](analysis/CORE_TASK_CustomDQN_vs_StableBaselines.md#exemples-de-failure-modes)

### 1.7 Documentation core task
- [x] **Document de synthèse** — FINI (Albane)
  - Résumé comparaison, choix de design, résultats clés
  - Fichier : [analysis/CORE_TASK_CustomDQN_vs_StableBaselines.md](analysis/CORE_TASK_CustomDQN_vs_StableBaselines.md)



## PART 2 : EXTENSION — Safety-Aware Reward Shaping

**Hypothèse** : pénaliser explicitement les comportements dangereux pendant
l'entraînement produit des agents avec un taux de collision réduit, sans
dégradation significative du reward moyen.

**Formulation** :
$$r_{safe} = r_{base} - \lambda \cdot c - \mu \cdot \mathbf{1}_{crash}$$

Deux variantes : Conservative (λ=10, μ=50) et Moderate (λ=5, μ=30).

### 2.1 Implémentation du reward shaping
- [x] **Fonction de reward safety-aware** — FINI (Juliette)
  - Fichier : [src/rlproject/reward_shaper.py](src/rlproject/reward_shaper.py)

- [x] **Script d'entraînement safety-aware** — FINI (Manon)
  - Fichier : [scripts/train_dqn_safety_aware.py](scripts/train_dqn_safety_aware.py)

- [x] **Entraînement Conservative** — FINI (Manon)
  - Checkpoints : `checkpoints_safety_aware_conservative/`
- [x] **Entraînement Moderate** — FINI (Albane)
  - Checkpoints : `checkpoints_safety_aware_moderate/`

### 2.2 Métriques de sécurité
- [x] **Implémentation des métriques** — FINI (Albane)
  - Collision rate, mean crashes, mean speed
  - Fichier : [src/rlproject/safety_metrics.py](src/rlproject/safety_metrics.py)

### 2.3 Évaluation des variantes
- [x] **Script d'évaluation** — FINI (Manon)
  - Fichier : [scripts/evaluate_safety_aware.py](scripts/evaluate_safety_aware.py)

- [x] **Évaluation des trois agents** — FINI (Albane)
  - Seeds : 0, 1, 3 — 50 épisodes par seed
  - Résultats : `comparison_results/safety_aware_eval_20260412_153241.md`

- [x] **Plots comparatifs** — FINI (Manon)
  - Script : [scripts/plot_safety_aware.py](scripts/plot_safety_aware.py)
  - Résultats : `comparison_results/safety_aware_training_curves.png`

### 2.4 Analyse quantitative
- [x] **Commentaire des résultats quantitatifs** — FINI (Albane)
  - Fichier : [Analyses des résultats quantitatifs dans analysis/EXTENSION_TASK_safety_aware.md](analysis/EXTENSION_TASK_safety_aware.md#5-analyses-des-résultats-quantitatifs)

### 2.5 Analyse qualitative 
- [ ] **Rollouts variantes safety-aware** — FINI (Manon)
  - Seeds : 0, 1, 3 pour Conservative et Moderate

- [ ] **Analyse qualitative et failure modes** — FINI (Manon)
  - Différences comportementales entre les trois agents
  - Fichier : [Analyse des résultats qualitatifs dans analysis/EXTENSION_TASK_safety_aware.md](analysis/EXTENSION_TASK_safety_aware.md#6-analyse-des-résultats-qualitatifs-rollouts)


### 2.6 Documentation extension task
- [x] **Document de synthèse** — FINI (Albane)
  - Résumé comparaison, choix de design, résultats clés
  - Fichier : [analysis/EXTENSION_TASK_safety_aware.md](analysis/EXTENSION_TASK_safety_aware.md)


## PART 3 : MISE EN FORME DES MARKDOWNS FINAUX

- [x] Mis à jour du README (Simon, Juliette, Manon et Albane)
- [x] Mis à jour du task.md (Juliette et Albane)


## Références

- Dueling DQN : Wang et al., 2015
- Double DQN : Van Hasselt et al., 2015
- Highway-env : https://highway-env.readthedocs.io/