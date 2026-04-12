"""Analyse de l'enregistrement du rollout"""

# Rollout Recording (Notes)

Run scripts/record_rollout.py to generate videos under rollouts/.

commandes:
- python scripts/record_rollout.py --agent custom --episodes 1 --seed 0
- python scripts/record_rollout.py --agent sb3 --episodes 1 --seed 0

# Analyse todo

en faisant plusieurs rollout j'observe :
- custom : changement de voie inutile
- sb3 : collision
