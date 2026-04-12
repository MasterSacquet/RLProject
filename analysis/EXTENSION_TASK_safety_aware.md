# Extension Task: Safety-Aware Reward Shaping

## Sommaire
1. [Introduction et objectif](#1-introduction-et-objectif)
2. [Formulation du reward shaping](#2-formulation-du-reward-shaping)
3. [Implémentation](#3-implémentation)
4. [Protocole d'évaluation](#4-protocole-dévaluation)
5. [Analyses des résultats quantitatifs](#5-analyses-des-résultats-quantitatifs)
    * [5.1 Résultats de moderate](#51-résultats-de-moderate)
    * [5.2 Résultats de conservative](#52-résultats-de-conservative)
    * [5.3 Stabilité selon les seeds](#53-stabilité-selon-les-seeds)
    * [5.4 Mean speed comme indicateur comportemental](#54-mean-speed-comme-indicateur-comportemental)
    * [5.5 Conclusion quantitative](#55-conclusion-quantitative)
6. [Analyse des résultats qualitatifs (rollouts)](#6-analyse-des-résultats-qualitatifs-rollouts)


## 1. Introduction et objectif

La core task a mis en évidence que le DQN custom, bien que supérieur à SB3 en reward moyen, présente des collisions récurrentes en trafic dense. 

Dans un scénario de conduite autonome, la sécurité est fondamentale pas un simple bonus de performance. L'extension retenue vise à quantifier le compromis sécurité/performance en modifiant explicitement l'objectif d'apprentissage via un reward shaping orienté sécurité.

L'hypothèse expérimentale est la suivante : en pénalisant explicitement les comportements dangereux pendant l'entraînement, on devrait obtenir des agents avec un taux de collision réduit, potentiellement au prix d'une légère dégradation du reward moyen.


## 2. Formulation du reward shaping

La récompense shapée suit la formulation :

$$r_{safe} = r_{base} - \lambda \cdot c - \mu \cdot \mathbf{1}_{crash}$$

où $r_{base}$ est la récompense originale de Highway-v0, $c$ le coût de collision (magnitude 1.5, cohérente avec le `collision_reward` de la config partagée), et $\mathbf{1}_{crash}$ un indicateur binaire de crash. Deux variantes sont entraînées avec des intensités de pénalité différentes :

| Variante | $\lambda$ | $\mu$ | Interprétation |
|---|---|---|---|
| Baseline | - | - | Reward original Highway-v0 |
| Moderate | 5 | 30 | Pénalité équilibrée |
| Conservative | 10 | 50 | Forte pénalité de sécurité |

Le reward shaping agit uniquement pendant l'entraînement. L'évaluation est conduite sur la récompense originale de Highway-v0 pour garantir la comparabilité entre agents.

## 3. Implémentation

L'extension repose sur trois nouveaux modules :

- `src/rlproject/reward_shaper.py` : implémente la formule de shaping et les
  deux presets de poids (Conservative, Moderate). Le shaping est appliqué
  uniquement pendant l'entraînement, la récompense originale restant inchangée
  pour l'évaluation.
- `src/rlproject/safety_metrics.py` : définit les métriques de sécurité
  (collision rate, mean crashes, mean speed) et la fonction `run_episode()`
  utilisée pour l'évaluation.
- `scripts/train_dqn_safety_aware.py` : reprend la boucle d'entraînement de
  `train_dqn_custom.py` en y branchant `reward_shaper.py`. Les métriques
  de sécurité (crash count, shaped reward) sont trackées en parallèle du
  reward original pendant l'entraînement.
- `scripts/evaluate_safety_aware.py` : évalue les trois agents (baseline,
  conservative, moderate) sur le même protocole multi-seed en utilisant
  `safety_metrics.py`.

## 4. Protocole d'évaluation

- Environnement identique pour les trois agents (`shared_core_config.py`)
- Seeds : 0, 1, 3 - 50 épisodes par seed - 150 épisodes au total par agent
- Métriques : reward moyen ± écart-type, collision rate, mean speed
- Fichiers de résultats : `comparison_results/safety_aware_eval_20260412_153241.md`
- Sélection des modèles : `last_model` pour la baseline (convergence en fin d'entraînement), `best_model` pour conservative et moderate (pic de performance observé avant la fin)




## 5. Analyses des résultats quantitatifs

#### Par seed

| Agent | Seed | Reward Mean | Reward Std | Collision Rate | Mean Speed |
|---|---|---|---|---|---|
| Baseline | 0 | 18.76 | 5.07 | 20.00% | 21.61 |
| Baseline | 1 | 19.37 | 4.47 | 16.00% | 21.12 |
| Baseline | 3 | 16.94 | 6.47 | 28.00% | 21.80 |
| Conservative | 0 | 18.80 | 5.49 | 20.00% | 21.51 |
| Conservative | 1 | 18.97 | 5.73 | 16.00% | 21.64 |
| Conservative | 3 | 16.41 | 7.15 | 30.00% | 21.61 |
| Moderate | 0 | 20.38 | 2.99 | 8.00% | 21.08 |
| Moderate | 1 | 19.46 | 4.81 | 10.00% | 20.90 |
| Moderate | 3 | 18.77 | 4.98 | 26.00% | 21.33 |

#### Overall

| Agent | Reward Mean | Reward Std | Collision Rate | Mean Speed |
|---|---|---|---|---|
| Baseline | 18.36 | 5.50 | 21.33% | 21.51 |
| Conservative | 18.06 | 6.28 | 22.00% | 21.59 |
| Moderate | **19.54** | **4.40** | **14.67%** | 21.10 |



#### 5.1 Résultats de moderate

L'agent Moderate est le seul à avoir de réels meilleurs résultats au niveau de la sécurité mais étonnamment cela ne se fait pas au prix d'une dégradation du reward moyen (la collision est quand même pénalisée de 1,5 dans le reward de la config). En effet, il obtient :
- le meilleur reward moyen (19.54 vs 18.36 pour la baseline, +6.4%)
- le taux de collision le plus bas (14.67% vs 21.33%, réduction de 31%)
- une variance également la plus faible (std 4.40), ce qui indique un comportement plus stable et reproductible. 

Ce résultat suggère qu'une pénalité de sécurité modérée ($\lambda=5$, $\mu=30$) pousse l'agent à éviter les situations à risque sans dégrader sa capacité à optimiser la vitesse et la progression. 

#### 5.2 Résultats de conservative

Le résultat le plus inattendue est celui de Conservative, dont le taux de collision (22.00%) est légèrement supérieur à la baseline (21.33%), malgré une pénalité de sécurité deux fois plus forte. 
Ce résultat est contre-intuitif et nous allons donner des explications qui pourraient être plausible. 

Tout d'abord, une pénalité trop agressive ($\lambda=10$, $\mu=50$) peut induire un comportement excessivement passif : l'agent apprend à éviter toute action risquée, y compris les changements de voie défensifs qui seraient pourtant nécessaires pour éviter une collision imminente. 

Par ailleurs, la sélection du best_model à l'épisode 423 sur 500 peut introduire un biais : ce checkpoint correspond au meilleur épisode individuel, pas nécessairement au comportement le plus sûr en moyenne.     

Enfin, la seed 3 est particulièrement défavorable pour Conservative (30% de collision rate), ce qui suggère une fragilité face à certaines configurations de trafic.

#### 5.3 Stabilité selon les seeds

La seed 3 est systématiquement la plus difficile pour les trois agents, avec des collision rates de 28%, 30% et 26% respectivement. Cela reflète une configuration de trafic particulièrement dense ou défavorable générée par cette seed. Moderate est le seul agent à maintenir des performances acceptables sur seed 3 (18.77 de reward moyen), ce qui renforce l'interprétation d'un comportement plus robuste.

#### 5.4 Mean speed comme indicateur comportemental

Les vitesses moyennes sont très proches entre agents (20.90 à 21.80 m/s), ce qui indique que les trois agents maintiennent une vitesse élevée de façon similaire. La légère vitesse inférieure de Moderate (21.10 vs 21.59 pour Conservative) est cohérente avec un comportement plus prudent, mais l'écart est trop faible pour être conclusif. Cette métrique confirme que le gain de sécurité de Moderate ne provient pas d'un ralentissement global mais d'une meilleure gestion des situations à risque.


#### 5.5 Conclusion quantitative

Les résultats invalident partiellement l'hypothèse initiale : le reward shaping n'améliore pas mécaniquement la sécurité, et une pénalité trop forte peut être contre-productive. L'agent Moderate constitue le meilleur compromis sécurité/performance dans cette configuration expérimentale, avec une réduction de 31% du taux de collision et une amélioration simultanée du reward moyen. Ce résultat suggère l'existence d'un régime optimal de pénalité de sécurité, au-delà duquel la sur-pénalisation dégrade les deux objectifs simultanément.


## 6. Analyse des résultats qualitatifs (rollouts)

*L'analyse qualitative (comportements observés en rollout, failure modes spécifiques) est développée dans la section suivante.*