# Core Task: Custom DQN vs Stable Baselines3

## Table des matières
- [1. Introduction](#1-introduction)
- [2. Environnement et configuration](#2-environnement-et-configuration)
- [3. Agents implémentés](#3-agents-implémentés)
    - [3.1 DQN personnalisé](#31-dqn-personnalisé)
    - [3.2 Stable Baselines3](#32-stable-baselines3)
- [4. Protocole d'évaluation](#4-protocole-dévaluation)
- [5. Résultats quantitatifs](#5-analyse-des-résultats-quantitatifs)
- [6. Résultats qualitatifs](#6-analyse-des-résultats-qualitatifs-rollouts)
    - [Analyses globales](#analyses-globales)
    - [Exemples de failure case](#exemples-de-failure-modes)
- [7. Conclusion sur les choix des designs et les résultats obtenus](#7-conclusion)


## 1. Introduction

La core task consiste à entraîner et comparer deux agents de conduite autonome
sur l'environnement Highway-v0 : un DQN implémenté manuellement et un modèle
entraîné avec la bibliothèque Stable Baselines3.
 L'objectif est double :
- démontrer une compréhension des mécanismes fondamentaux du DQN à travers une
implémentation from scratch
- évaluer dans quelle mesure une bibliothèque optimisée peut être compétitive face à une implémentation custom sur un
benchmark partagé.

Les deux agents sont entraînés dans des conditions strictement identiques
(même environnement, même configuration, mêmes seeds d'évaluation) afin de
garantir une comparaison fair.


## 2. Environnement et configuration

L'environnement utilisé est highway-v0 de la bibliothèque highway-env.

La configuration partagée est précisée dans `shared_core_config.py` :

| Paramètre | Valeur | Signification |
|---|---|---|
| `vehicles_count` | 10 | Véhicules observables (champ de vision) |
| `lanes_count` | 4 | Nombre de voies |
| `vehicles_count` (global) | 45 | Total véhicules sur l'autoroute |
| `duration` | 30 | Durée en secondes |
| `observation features` | position, vitesse | État observé par l'agent |
| `action type` | DiscreteMetaAction | Actions: ralentir, maintenir, accélérer, changer voie |
| `target_speeds` | [20, 25, 30] | Vitesses cibles en m/s |
| `collision_reward` | -1.5 | Pénalité collision |
| `high_speed_reward` | 0.7 | Récompense vitesse élevée |
| `lane_change_reward` | -0.02 | Pénalité changement voie |

La fonction de récompense originale de Highway-v0 combine ces composantes :

$$r = r_{speed} + r_{collision} + r_{lane\_change}$$

où $r_{speed}$ récompense la vitesse élevée, $r_{collision}$ pénalise les
crashes, et $r_{lane\_change}$ pénalise légèrement les changements de voie.
Cette structure favorise mécaniquement les stratégies agressives orientées
vers la vitesse.


## 3. Agents implémentés

### 3.1 DQN Personnalisé

Le DQN custom (`src/rlproject/dqn_agent.py`) utilise une **Architecture Dueling DQN** avec normalisation par batch :

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

Le flux Value capture l'intérêt global d'un état, le flux Advantage capture
la qualité relative de chaque action. Cette décomposition améliore la
stabilité dans les états où plusieurs actions ont des valeurs similaires,
ce qui est fréquent en trafic fluide sur Highway-v0.

#### Algorithme d'apprentissage (`src/rlproject/dqn_agent.py`)

L'agent utilise **Double DQN** pour éviter la surestimation des Q-values :
- Sélection d'action : Utilise le réseau Q principal discretement ou aléatoirement (epsilon-greedy)
- Évaluation de l'action suivante : Utilise le réseau target pour évaluer l'action suivante
- Loss : SmoothL1Loss (Huber loss) entre Q(s,a) estimée et Q(s,a) cible
- Optimiseur : Adam + régularisation L2

L'agent est entraîné avec un learning rate de 5e-4, un facteur de discount
γ=0.99, un batch size de 32 et une mise à jour du target network tous les
50 pas.

#### Stratégie d'exploration (`src/rlproject/dqn_agent.py`)

- Epsilon-Greedy: Choisit une action aléatoire avec probabilité ε, sinon la meilleure action
- Décroissance: ε démarre à 1.0 et décroît de 0.99 par épisode jusqu'à 0.02
- Objectif: Forcer l'exploitation progressive avec exploration initiale

ε démarre à 1.0 et décroît de 0.99 par épisode jusqu'à un minimum de 0.02,
ce qui assure une exploration complète en début d'entraînement et une
exploitation progressive en fin de training.

#### Entraînement (`scripts/train_dqn_custom.py`)

Pour la stabilisation de l'apprentissage, on utilise :
- Target Network : copie du réseau Q mise à jour tous les 50 pas
- Experience Replay : batch de 32 transitions tirées aléatoirement
  (buffer capacité 500k)
- Learning Rate Scheduling : cosine annealing avec warm restarts

L'entraînement est conduit sur 500 épisodes de 300 pas maximum (soit 30s
de simulation à 0.1s/step), avec un replay buffer de 500k transitions.

### 3.2 Stable Baselines3

Nous comparaons notre DQN personnalisé avec ce qui est proposé par Stable Baseline3 (`scripts/train_stable_baselines.py`).  Stable Baselines3 est une bibliothèque optimisée offrant une implémentation
professionnelle de DQN.

**Différences entre le DQN personnalisé et Stable Baselines**

| Aspect | DQN Personnalisé | Stable Baselines3 |
|---|---|---|
| Architecture réseau | Dueling (custom) | Flexible, optimisée |
| Double DQN | Oui | Oui (par défaut) |
| Priority Experience Replay | Non | Optionnel |
| Exploration strategy | Epsilon-greedy simple | Epsilon-greedy avec options avancées |
| Optimiseur | Adam + LR scheduling | Adam (configurable) |
| Normalisation d'observation | Manuel | Automatique (running normalization) |
| Performance observée (ce projet) | Supérieure en reward moyen | Inférieure en reward moyen |

**Avantages de SB3 :** code testé et optimisé par la communauté, normalisation
automatique des observations, gestion robuste de la variance d'entraînement.

SB3 est configuré avec γ=0.99, un batch size de 32, un buffer de 100k
transitions, et une exploration décroissant de 1.0 à 0.05 sur les 10
premiers pourcents des timesteps totaux.


## 4. Protocole d'évaluation 

- Environnement identique pour les deux agents (`shared_core_config.py`)
- Seeds identiques: 0, 1, 3
- 50 épisodes par seed
- Métrique principale: reward moyen ± écart-type
- Evaluation multi-seed: `comparison_results/multiseed_eval_20260402_164711.md`


## 5. Analyse des résultats quantitatifs 

| Seed | Custom Mean | Custom Std | SB3 Mean | SB3 Std |
|---|---|---|---|---|
| 0 | 18.76 | 5.07 | 10.24 | 5.66 |
| 1 | 19.37 | 4.47 | 11.28 | 5.94 |
| 3 | 16.94 | 6.47 | 10.59 | 6.80 |
| Overall | 18.36 | 5.50 | 10.70 | 6.17 |

Constat principal : dans cette configuration expérimentale, l'agent custom surpasse SB3 sur toutes les seeds en reward moyen.

La variance est non nulle pour les deux agents (std entre ~4.5 et ~6.8 selon seed), ce qui est attendu en RL. Le custom présente globalement un meilleur niveau de reward, avec une dispersion comparable à SB3 à seed donnée. 

Les courbes d'entraînement sont disponibles dans `comparison_results/training_curves_20260402_165013.png`.


## 6. Analyse des résultats qualitatifs (rollouts)

### Analyses globales 

**Custom :** maintien de voie globalement stable, avec quelques changements de voie inutiles. Vitesse souvent élevée, parfois peu de ralentissement lorsqu'un véhicule est juste devant. Le comportement est opportuniste : l'agent privilégie souvent la progression (vitesse/déplacement), au prix d'un risque accru dans les situations serrées. En trafic dense, l'agent est plus en difficulté : temps de réaction parfois trop long et collisions plus probables.

**SB3 :** changements de voie plus fréquents pour contourner les véhicules lents. Conduite visuellement plus fluide dans certaines scènes (décisions plus lisibles), mais pas nécessairement plus performante globalement. Des collisions existent aussi, souvent liées à un changement de voie tardif ou à une anticipation insuffisante. En trafic dense, l'agent tente davantage de se repositionner, ce qui peut aider mais ne supprime pas les erreurs.

**Point important :** un comportement visuellement "plus propre" n'implique pas automatiquement une meilleure reward moyenne. La récompense Highway-v0 valorise fortement la progression et la vitesse (`high_speed_reward = 0.7`), ce qui favorise mécaniquement la stratégie agressive du custom.


### Exemples de failure modes

**Custom - seed 1, épisode 2 :** collision arrière lorsque le trafic ralentit sur la voie courante. Cause probable : stratégie trop orientée vers la vitesse et freinage tardif lorsque l'échappatoire par changement de voie est bloquée.

**SB3 - seed 1, épisode 1 :** collision lors d'un changement de voie. Cause probable : décision de changement de voie déclenchée trop tard, sans freinage compensatoire.

Ces deux failure modes illustrent des stratégies d'évitement différentes mais également limitées : le custom échoue quand le freinage est inévitable, SB3 échoue quand le timing de repositionnement est insuffisant.


## 7. Conclusion

Pour la partie baseline de ce projet, le custom DQN surpasse SB3 sur toutes
les seeds évaluées. Cet écart s'explique en partie par les choix
d'architecture : Dueling + Double DQN produisent des estimations de Q-values
plus stables, favorisant une convergence vers une politique plus performante
sur 500 épisodes. La stratégie agressive du custom est cohérente avec la
structure de la fonction de récompense Highway-v0 qui valorise fortement la
progression.

SB3 reste une baseline robuste et rapide à configurer. La différence observée
ne constitue pas une conclusion générale sur la supériorité des
implémentations manuelles, elle est en partie attribuable au budget
d'entraînement et aux hyperparamètres retenus.

Un DQN vanilla a été implémenté en première approche mais ses performances
se sont révélées insuffisantes, ce qui a motivé l'intégration de Dueling
et Double DQN. Cette progression itérative confirme l'intérêt de ces
améliorations dans ce contexte, même si l'absence de résultats formalisés
du vanilla ne permet pas d'en quantifier précisément le gain.
Qualitativement, les deux
agents gardent des limites de sécurité comparables, et nous avons bien remarqué la différence entre
"qualité visuelle" et "reward optimisée" .

*L'analyse de l'extension safety-aware est développée dans
`analysis/EXTENSION_TASK_safety_aware.md`.*
