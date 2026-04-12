"""qualitative behavior notes from rollout videos."""
 
# Qualitative Behavior Analysis CUSTOM

## Ce qui a été enregistré
- Agent: custom
- Seed: 0, 1
- Episodes recorded: 3

## Comportement global
- Maintien de voie globalement stable, avec quelques changements de voie inutiles.
- Vitesse souvent elevée, parfois peu de ralentissement lorsqu'un vehicule est juste devant.
- Le comportement est opportuniste: l'agent privilégie souvent la progression (vitesse/deplacement), au prix d'un risque accru dans les situations serrées.
- En trafic dense, l'agent est plus en difficulté: temps de reaction parfois trop long et collisions plus probables.

## Exemple representatif: seed 1, episode 2
On observe une collision car la voiture ne ralentit pas a temps. Ce cas met en evidence un defaut recurrent: quand le changement de voie n'est pas possible immediatement, l'agent peut tarder a freiner et percuter le vehicule devant.

## Failure mode
Custom seed 1, episode 2: collision arriere avec une autre voiture.
Contexte: trafic qui ralentit sur la voie courante.
Cause probable: strategie trop orientee vers la vitesse et freinage tardif lorsque l'échappatoire par changement de voie est bloquée.

# Qualitative Behavior Analysis SB3

## Ce qui a été enregistré
- Agent: sb3
- Seed : 1
- Episodes recorded: 3
- Video file(s):

## Comportement global
- Changement de voie plus frequent pour contourner les vehicules lents.
- Conduite visuellement plus fluide dans certaines scenes (decisions plus lisibles), mais pas necessairement plus performante globalement.
- Des collisions existent aussi, souvent liees a un changement de voie tardif ou a une anticipation insuffisante.
- En trafic dense, l'agent tente davantage de se repositionner, ce qui peut aider mais ne supprime pas les erreurs.

## Failure mode
Seed 1, episode 1: collision lors d'un changement de voie.
Contexte: tentative d'évitement d'un vehicule plus lent en face.
Cause probable: décision de changement de voie declenchée trop tard, sans freinage compensatoire.


# Comparaison custom vs sb3
- difference de comportement:
Custom parait plus agressif (vitesse/progression), SB3 parait plus "propre" visuellement sur certaines sequences (plus de repositionnement), mais avec une efficacite moindre selon la recompense moyenne.
- difference de stabilité:
Les deux agents peuvent etre instables dans les situations contraintes. Custom montre des erreurs de freinage tardif; SB3 montre des erreurs de timing de changement de voie.
- difference de sécurité:
Aucun des deux n'est totalement sur. Visuellement, SB3 peut sembler un peu plus prudent, mais les collisions restent presentes des deux cotes.

## Coherence avec les resultats quantitatifs
- Resultats observés: Custom > SB3 sur toutes les seeds (mean reward 18.36 vs 10.70).
- Ce n'est pas contradictoire avec le ressenti visuel: la recompense Highway peut valoriser fortement la progression et la vitesse, meme si le style de conduite parait moins "propre".

