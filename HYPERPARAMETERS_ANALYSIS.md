# Analyse de Cohérence DQN - dqn_agent.py vs train_dqn_custom.py

## ✅ HYPERPARAMÈTRES COHÉRENTS

| Paramètre | Valeur | Lieu | Cohérence |
|-----------|--------|------|-----------|
| state_dim | Dynamique (50) | Récupéré de l'env | ✓ |
| action_dim | Dynamique (5) | Récupéré de l'env | ✓ |
| learning_rate | 5e-4 | DQNAgent default | ✓ |
| gamma | 0.99 | DQNAgent default | ✓ |
| batch_size | 32 | En dur dans DQNAgent | ✓ |
| epsilon_min | 0.02 | En dur dans DQNAgent | ✓ |
| epsilon_decay | 0.99 | En dur dans DQNAgent | ✓ |
| epsilon appels | 1x/épisode | Appelé dans train_dqn | ✓ |
| buffer_capacity | 500_000 | En dur dans train | ✓ |
| buffer_min (pour train) | 64 (batch*2) | En dur dans DQNAgent | ✓ |
| target_update_freq | 50 steps | En dur dans DQNAgent | ✓ |
| gradient_clip | 10.0 | En dur dans DQNAgent | ✓ |

## ⚠️ PROBLÈME IDENTIFIÉ

### Learning Rate Scheduler (CosineAnnealingWarmRestarts)
- **Paramètres actuels**: `T_0=10, T_mult=2`
- **Problème**: Le scheduler appelle `.step()` à **chaque appel de `train()`**
- **Fréquence**: `train()` est appelé jusqu'à 300 fois/épisode
  - Total: ~500 épisodes × ~150 steps moyen = ~75,000 calls à `train()`
  - Avec T_0=10, il y aura ~7,500 warm restarts!
  - **C'est BEAUCOUP trop fréquent!**

### Conséquence
Le learning rate variera de manière chaotique au lieu de décroître graduellement. Le scheduler est conçu pour des epochs (ce qui serait 1/épisode ici), pas pour chaque step d'optimisation.

## 🔧 SOLUTIONS PROPOSÉES

### Option 1: Scheduler appelé UNE FOIS par épisode (Recommandée)
- Créer une méthode `step_scheduler()` dans DQNAgent
- L'appeler dans train_dqn.py après le decay_epsilon()
- Paramètres idéaux: `T_0=50` (50 warm restarts sur 500 épisodes)

### Option 2: Augmenter T_0
- Changer `T_0=10` → `T_0=300` (équivaut à ~250 steps = plusieurs épisodes)
- Moins idéal mais fonctionne

### Option 3: Scheduler linéaire simple
- Remplacer par `StepLR(step_size=50, gamma=0.95)`
- Appeler une fois par épisode
- Plus simple et prévisible

## 📊 STATISTIQUES D'EXÉCUTION PRÉVUES

### Avec les paramètres actuels:
- **Durée estimée**: 500 épisodes × ~100 steps moyen (avec early stopping) = ~50,000 steps
- **Appels optimizer.step()**: ~50,000 (beaucoup de training!)
- **Buffering**: Bon ratio 32/500_000

### Phases d'epsilon:
- **0-67 épisodes**: Exploration → ε: 1.0 → 0.5
- **68-160 épisodes**: Exploration-Exploitation → ε: 0.5 → 0.2
- **161-500 épisodes**: Quasi-Exploitation → ε: 0.2 → 0.02

### Phases de target network sync:
- Update tous les 50 steps
- Très fréquent (bonus pour stabilité)

## RECOMMANDATIONS FINALES

1. ✅ **Utiliser Option 1** (Scheduler 1x/épisode avec T_0=50)
2. ✅ Vérifier que action_dim=5 est correct (pas 3 comme dans config)
3. ✅ Réduire batch_size à 16 si mémoire/vitesse est problématique
4. ✅ Augmenter NUM_EPISODES à 1000 si les résultats ne convergent pas

