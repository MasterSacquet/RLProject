# Projet Reinforcement Learning - Agents DQN sur Highway-v0

Comparaison entre une implémentation personnalisée de DQN et la bibliothèque Stable Baselines3 sur l'environnement Highway-v0.

## Vue d'ensemble du projet

Le projet compare deux approches pour entraîner un agent de conduite autonome:

1. **DQN Personnalisé** (`dqn_agent.py`): Implémentation manuelle avec Dueling DQN et Double DQN
2. **Stable Baselines3**: Bibliothèque optimisée pour l'apprentissage par renforcement

### Structure des fichiers

```
shared_core_config.py        # Configuration partagée pour l'environnement
highway_env_wrapper.py       # Wrapper pour Highway-v0
dqn_agent.py                 # Agent DQN personnalisé avec architecture Dueling
replay_buffer.py             # Buffer de rejeu pour la mémorisation
train_dqn_custom.py          # Script d'entraînement DQN personnalisé
train_stable_baselines.py    # Script d'entraînement Stable Baselines3
test_dqn_custom.py           # Évaluation de l'agent DQN
test_stable_baselines.py     # Évaluation de Stable Baselines3
compare_agents.py            # Comparaison des performances
checkpoints_custom/          # Sauvegardes du modèle DQN
checkpoints_sb3/             # Sauvegardes du modèle Stable Baselines3
```

---

## 1. Fonctionnement du DQN Personnalisé

### Architecture du réseau Q

Le DQN utilise une **architecture Dueling DQN** avec normalisation par batch:

```
État de l'agent (50 dimensions)
    |
    v
Dense(256) + BatchNorm + ReLU
    |
    v
Dense(256) + BatchNorm + ReLU
    |
    +---> Value Stream --> Dense(128) + ReLU --> Dense(1)
    |
    +---> Advantage Stream --> Dense(128) + ReLU --> Dense(3)
    |
    v
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
```

**Avantages de cette architecture:**
- Le flux Value capture l'intérêt global d'un état
- Le flux Advantage capture la qualité relative de chaque action
- Cela améliore la stabilité et la convergence

### Algorithme d'apprentissage

L'agent utilise **Double DQN** pour éviter la surestimation des Q-values:

1. **Sélection d'action**: Utilise le réseau Q principal discretement ou aléatoirement (epsilon-greedy)
2. **Évaluation**: Utilise le réseau target pour évaluer l'action suivante
3. **Loss**: `SmoothL1Loss` (Huber loss) entre Q(s,a) estimée et Q(s,a) cible
4. **Mise à jour**: Gradient descent avec Adam optimizer et régularisation L2

### Stratégie d'exploration

- **Epsilon-Greedy**: Choisit une action aléatoire avec probabilité ε, sinon la meilleure action
- **Décroissance**: ε démarre à 1.0 et décroît de 0.99 par épisode jusqu'à 0.02
- **Objectif**: Forcer l'exploitation progressive avec exploration initiale

### Stabilisation de l'apprentissage

- **Target Network**: Copie du réseau Q mise à jour tous les 50 pas
- **Experience Replay**: Batch de 32 transitions tirées aléatoirement du buffer (500k capacité)
- **Learning Rate Scheduling**: Cosine annealing avec warm restarts pour ajuster dynamiquement le taux d'apprentissage

---

## 2. Fonctionnement de Stable Baselines3

Stable Baselines3 est une bibliothèque optimisée offrant une implémentation professionnelle de DQN.

### Différences principales avec le DQN personnalisé

|Aspect|DQN Personnalisé|Stable Baselines3|
|---|---|---|
|Architecture réseau|Dueling (custom)|Flexible, optimisée|
|Double DQN|Oui|Oui (par défaut)|
|Priority Experience Replay|Non|Optionnel|
|Exploration strategy|Epsilon-greedy simple|Epsilon-greedy avec options avancées|
|Optimiseur|Adam + LR scheduling|Adam (configurable)|
|Normalisation d'observation|Manuel|Automatique (running normalization)|
|Performance|Bonne|Généralement supérieure|

### Avantages de Stable Baselines3

- Code testé et optimisé par la communauté
- Normalisation automatique des observations
- Gestion robuste de la variance d'entraînement
- Intégration facile avec des callbacks personnalisés

---

## 3. Hyperparamètres clés et leur impact

### Pour l'Agent DQN Personnalisé

#### Apprentissage (`dqn_agent.py`)

| Hyperparamètre | Valeur défaut | Impact |
|---|---|---|
| `learning_rate` | 5e-4 | Réduit pour stabilité. Augmenter si convergence lente, réduire si instable. |
| `gamma` | 0.99 | Facteur de discount. Plus haut = considère plus les récompenses futures (0.99 bon pour highway) |
| `batch_size` | 32 | Plus petit pour mises à jour graduelles. Augmenter si GPU disponible (32→64) |
| `update_frequency` | 50 | Fréquence de copie du target network. Réduire si instabilité (50→100) |

#### Exploration (`dqn_agent.py`)

| Hyperparamètre | Valeur défaut | Impact |
|---|---|---|
| `epsilon` début | 1.0 | Commence par exploration pure. Fixer à 1.0 pour reset à chaque session |
| `epsilon_min` | 0.02 | Exploration minimale (2%). Augmenter pour plus d'aléatoire en fin d'entraînement |
| `epsilon_decay` | 0.99 | Décroissance par épisode. Augmenter (0.99→0.995) pour exploration plus longue |

#### Entraînement (`train_dqn_custom.py`)

| Hyperparamètre | Valeur défaut | Impact |
|---|---|---|
| `NUM_EPISODES` | 500 | Nombre d'épisodes. Plus haut = meilleure convergence (500-1000 recommandé) |
| `MAX_STEPS` | 300 | Longueur d'un épisode. Highway-v0 = 30s, MAX_STEPS=300 donc 0.1s/step |
| `replay_buffer capacity` | 500K | Augmenter pour mieux explorer l'espace d'expériences (500K→1M) |
| `SAVE_INTERVAL` | 50 | Sauvegarde tous les 50 épisodes pour tracking |

#### Planifier l'entraînement optimisé

```python
# Pour rapide mais bruité (debugging)
NUM_EPISODES = 100
learning_rate = 1e-3
epsilon_decay = 0.95

# Pour stable et convergent
NUM_EPISODES = 500
learning_rate = 5e-4
epsilon_decay = 0.99
batch_size = 32

# Pour maximum convergence (long)
NUM_EPISODES = 1000
learning_rate = 3e-4
epsilon_decay = 0.995
batch_size = 64
update_frequency = 50
replay_buffer capacity = 1M
```

### Pour Stable Baselines3 (`train_stable_baselines.py`)

| Hyperparamètre | Configuration |
|---|---|
| `learning_rate` | 3e-5 à 1e-3 (défaut: calculé auto) |
| `gamma` | 0.99 (même que DQN custom) |
| `exploration_fraction` | 0.1 de `total_timesteps` avant full exploitation |
| `exploration_initial_eps` | 1.0 (exploration complète au démarrage) |
| `exploration_final_eps` | 0.05 (5% d'exploration en fin) |
| `target_update_interval` | Tous les 10000 timesteps (à ajuster) |
| `train_freq` | Met à jour le modèle après chaque step (par défaut) |
| `batch_size` | 32 (peut augmenter si GPU) |
| `buffer_size` | 100K (taille du replay buffer) |

#### Optimiser l'entraînement Stable Baselines

```python
# Approche rapide (test)
model = DQN("MlpPolicy", env, 
    learning_rate=1e-3,
    exploration_fraction=0.2,
)
model.learn(50_000)

# Approche balancée
model = DQN("MlpPolicy", env,
    learning_rate=5e-4,
    gamma=0.99,
    exploration_fraction=0.1,
    buffer_size=500_000,
)
model.learn(500_000)

# Approche de convergence maximale
model = DQN("MlpPolicy", env,
    learning_rate=3e-4,
    gamma=0.99,
    exploration_fraction=0.15,
    buffer_size=1_000_000,
    target_update_interval=10_000,
)
model.learn(1_000_000)
```

---

## 4. Configuration de l'Environnement Highway-v0

Les paramètres clés sont dans `shared_core_config.py`:

| Paramètre | Valeur | Signification |
|---|---|---|
| `vehicles_count` | 10 | Véhicules observables (champ de vision) |
| `lanes_count` | 4 | Nombre de voies |
| `vehicles_count` (global) | 45 | Total véhicules sur l'autoroute |
| `duration` | 30 | Durée en secondes |
| `observation features` | position, vitesse | État observé par l'agent |
| `action type` | DiscreteMetaAction | Actions: ralentir, maintenir, accélérer, changer voie |
| `target_speeds` | [20, 25, 30] | Vitesses en m/s |
| `collision_reward` | -1.5 | Pénalité collision |
| `high_speed_reward` | 0.7 | Récompense vitesse élevée |
| `lane_change_reward` | -0.02 | Pénalité changement voie |

Pour tester avec des configurations différentes, modifier ces paramètres via `config_override` dans `HighwayV0Env()`.

---

## 5. Comment utiliser le projet

### Entraîner l'agent DQN personnalisé

```bash
python train_dqn_custom.py
```

**Sorties:**
- `checkpoints_custom/`: Modèles sauvegardés tous les 50 épisodes
- `checkpoints_custom/metrics.json`: Courbes de récompense et loss

### Entraîner avec Stable Baselines3

```bash
python train_stable_baselines.py
```

**Sorties:**
- `checkpoints_sb3/`: Modèles sauvegardés
- `checkpoints_sb3/metrics.json`: Métriques d'entraînement

### Évaluer les modèles

```bash
python test_dqn_custom.py          # Teste le DQN personnalisé
python test_stable_baselines.py    # Teste Stable Baselines3
python compare_agents.py           # Compare les deux approches
```

---

## 6. Conseils d'optimisation

### Si le modèle ne converge pas
- Augmenter `NUM_EPISODES` ou `TOTAL_TIMESTEPS`
- Réduire `learning_rate` (5e-4 → 1e-4)
- Augmenter `gamma` vers 0.995 pour valoriser plus les futures récompenses
- Augmenter `batch_size` pour moins de bruit

### Si le modèle est trop lent
- Réduire `NUM_EPISODES` ou `TOTAL_TIMESTEPS`
- Augmenter `learning_rate` (5e-4 → 1e-3)
- Réduire `batch_size` (32 → 16) pour updates plus fréquentes
- Augmenter `epsilon_decay` pour réduire exploration

### Si le modèle est instable
- Augmenter `update_frequency` (50 → 100) pour target network
- Réduire `learning_rate` davantage
- Augmenter `batch_size` pour généralisation
- Réduire `epsilon_decay` pour exploration plus longue

### Comparaison pratique

Le DQN personnalisé offre plus de contrôle. Stable Baselines3 offre plus de robustesse. Pour choix du meilleur:
- **DQN Custom**: Pédagogique, ajustable, bon pour expérimenter
- **Stable Baselines3**: Production-ready, moins bruité, convergence rapide

---

## Dépendances

- `torch` - Deep learning
- `gymnasium` - Interface RL
- `highway-env` - Environnement de conduite
- `stable_baselines3` - Algorithmes RL optimisés
- `numpy` - Calculs numériques
