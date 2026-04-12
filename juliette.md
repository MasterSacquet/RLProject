# Synthese du projet RL (Juliette)

## Suivi des taches Juliette (d apres task.md)

### Fait

- [x] 1.3 Multi-Seed Evaluation
   - But: obtenir une comparaison quantitative robuste entre custom et SB3, en reduisant l effet du hasard lie a une seule seed.
   - Realisation: execution de l evaluation sur les seeds 0, 1 et 3, avec 50 episodes par seed et le meme protocole pour les deux agents.
   - Livrables: scripts/evaluate_multiseed.py, comparison_results/multiseed_eval_20260402_164711.json, comparison_results/multiseed_eval_20260402_164711.md.
   - Resultat: custom est meilleur sur toutes les seeds, avec un overall de 18.36 +/- 5.50 contre 10.70 +/- 6.17 pour SB3.

- [x] 1.4 Training Curves (co-realise avec Simon)
   - But: visualiser la dynamique d apprentissage et la stabilite des entrainements dans le temps.
   - Realisation: generation des courbes a partir des metrics.json des deux approches.
   - Livrables: scripts/plot_training_curves.py, comparison_results/training_curves_20260402_165013.png.
   - Resultat: courbes exploitables pour la discussion des performances et de la variance.

- [x] 1.5 Qualitative Analysis
   - But: completer la comparaison chiffrée par une lecture comportementale des agents en situation.
   - Realisation: analyse video des comportements (maintien de voie, vitesse, depassement, reaction au trafic dense) pour custom et SB3.
   - Livrable: analysis/qualitative_behavior.md.
   - Resultat: identification d un ecart entre ressenti visuel et reward moyenne, interprete dans le rapport.

- [x] 1.5 Recorded Rollouts
   - But: produire des evidences visuelles pour l analyse qualitative et les failure modes.
   - Realisation: enregistrement de rollouts pour custom et SB3 sur des seeds comparables.
   - Livrables: scripts/record_rollout.py, videos dans rollouts/.
   - Resultat: base video suffisante pour illustrer les comportements typiques et les erreurs recurrentes.

- [x] 1.6 Failure Mode Analysis
   - But: documenter au moins un mode de defaillance, expliquer le contexte et la cause probable.
   - Realisation: observation d episodes problemes (collision, timing tardif) et formulation d hypotheses explicatives.
   - Livrable: analysis/rollout_notes.md.
   - Resultat: failure mode principal formalise et relie au comportement de l agent sous contrainte.

- [x] 1.7 Fair Comparison Protocol
   - But: garantir une comparaison juste entre les deux approches.
   - Realisation: meme environnement, meme configuration, memes seeds, memes metriques, meme nombre d episodes.
   - Livrables: protocole applique dans les scripts d evaluation et les resultats multi-seed.
   - Resultat: conclusions quantitatives defendables car comparaison controlee.

- [x] 1.8 Documentation (Core Results)
   - But: centraliser les resultats baseline dans une synthese lisible et argumentee.
   - Realisation: ajout d une section Core Results Summary avec protocole, tableau, interpretation, stabilite et conclusion.
   - Livrable: README.md.
   - Resultat: documentation exploitable pour le rendu et la presentation orale.
   - Remarque: task.md contient encore "Status: A FAIRE" sur cette ligne, a corriger pour coherence administrative.

- [x] 2.1 Define Safety-Aware Reward Function
   - But: lancer l extension en ajoutant un objectif securite explicite dans la reward.
   - Realisation: implementation de la formule de reward shaping securite et des presets de penalisation.
   - Livrable: src/rlproject/reward_shaper.py.
   - Detail technique: formule r_safe = r_base - lambda * collision_penalty - mu * crash_indicator.
   - Variantes implementees:
      - conservative: lambda=10, mu=50
      - moderate: lambda=5, mu=30
   - Resultat: brique prete a etre branchee dans la boucle d entrainement safety-aware.

### A faire

- [ ] 2.1 Modify DQN Training Loop
   - Creer scripts/train_dqn_safety_aware.py
   - Entrainer conservative + moderate

- [ ] 2.2 Implement Safety Evaluation Metrics
   - collision_rate, mean_crashes, safety_margin, reward
   - utilitaires prevus dans src/rlproject/safety_metrics.py

- [ ] 2.3 Run Evaluation on All Variants
   - Evaluer baseline, conservative, moderate (seeds 0/1/3, 50 episodes)

- [ ] 2.3 Generate Comparative Plots
   - courbes d entrainement, collision rate, tradeoff reward/securite

- [ ] 2.4 Qualitative Observation (Safety-Aware)
   - rollouts des variantes safety-aware + analyse

- [ ] 2.4 Failure Mode Comparison (Baseline vs Safety-Aware)

- [ ] 2.5 Generalization Experiment (optionnel)

- [ ] 2.6 Extension Results
   - tableau tradeoff + figures dans comparison_results/

- [ ] 2.7 Extension Discussion

- [ ] 3.1 Core Results Summary (etat task.md)
   - Dans les faits, la section existe deja dans README.md.
   - Si necessaire, ajouter un fichier dedie CORE_RESULTS_SUMMARY.md

- [ ] 3.2 Extension Results Summary

- [ ] 3.3 README Update final
   - Integrer les resultats de l extension une fois experimentations terminees

## Resultats quantitatifs references

| seed | custom_mean | custom_std | sb3_mean | sb3_std |
|---|---|---|---|---|
| 0 | 18.76 | 5.07 | 10.24 | 5.66 |
| 1 | 19.37 | 4.47 | 11.28 | 5.94 |
| 3 | 16.94 | 6.47 | 10.59 | 6.80 |
| overall | 18.36 | 5.50 | 10.70 | 6.17 |

## Commandes utiles

- python scripts/evaluate_multiseed.py
- python scripts/plot_training_curves.py
- python scripts/record_rollout.py --agent custom --custom-model checkpoints_custom/best_model.pth --episodes 3 --seed 1
- python scripts/record_rollout.py --agent sb3 --sb3-model checkpoints_sb3/best_model.zip --episodes 3 --seed 1

