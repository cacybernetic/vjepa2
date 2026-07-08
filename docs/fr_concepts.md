# V-JEPA 2.1, expliqué en douceur (Français)

> Ce document est écrit pour les débutants. Le français est volontairement
> simple. Si un mot semble difficile, on l'explique. Prends ton temps.

## 1. La grande question

Imagine que tu regardes une courte vidéo : une main attrape une tasse. Même si
on cache une partie de l'écran, ton cerveau devine ce qu'il y a derrière. Tu
n'as besoin de personne pour t'expliquer la vidéo. Tu la *comprends*, c'est
tout.

**Un ordinateur peut-il apprendre à comprendre la vidéo pareil, sans
étiquettes ?**

C'est le but de **V-JEPA 2.1**. Il apprend à partir de vidéos brutes, sans
étiquettes, en jouant à un jeu simple avec lui-même : *« Je cache une partie de
la vidéo, et j'essaie de deviner ce que j'ai caché. »*

> **Lecteur naïf :** « Attends, s'il cache les pixels et devine les pixels, il
> dessine juste la partie manquante ? »
>
> Bonne question ! Non. Il ne devine **pas** les pixels. Il devine le *sens* de
> la partie cachée. On verra pourquoi c'est plus malin.

## 2. Les mots que l'on va utiliser

- **Image (frame)** : une seule image de la vidéo.
- **Clip** : un petit paquet d'images (par exemple 16 images).
- **Patch / token** : on découpe chaque image en petits carrés (16x16 pixels).
  Chaque carré devient un « token ». Un token est juste un vecteur de nombres.
- **Encodeur** : un réseau qui lit les tokens et sort une *représentation* (un
  vecteur plus riche qui capture le sens).
- **Prédicteur** : un réseau qui devine la représentation des tokens cachés.
- **Représentation / feature** : les nombres qui décrivent le sens d'un patch.
  Vois ça comme un petit résumé.

## 3. Pourquoi deviner le *sens* et non les *pixels* ?

Les pixels sont bruités. La couleur exacte d'un pixel n'a pas grande
importance. Si le modèle met toute son énergie à dessiner des pixels exacts, il
perd son temps sur des détails (grain, scintillement) qui n'aident pas à
comprendre.

À la place, V-JEPA devine dans l'**espace latent** (l'espace des
représentations). C'est l'idée « JEPA » : **J**oint **E**mbedding **P**redictive
**A**rchitecture (architecture prédictive à plongement joint).

> **Analogie :** Un critique de cinéma ne retient pas chaque pixel d'un film. Il
> retient *ce qui se passe* : « un homme ouvre une porte, a peur, s'enfuit ». Ce
> résumé, c'est la représentation. V-JEPA apprend à prédire le résumé de la
> partie cachée, pas les pixels exacts.

## 4. Les trois réseaux

V-JEPA 2.1 a trois parties qui travaillent ensemble :

1. **L'encodeur de contexte** (encodeur *online*). Il voit la vidéo **avec des
   trous** (des patches enlevés) et encode les patches visibles.
2. **Le prédicteur**. Il prend les patches visibles encodés, plus un marqueur
   pour chaque trou, et prédit la représentation aux trous.
3. **L'encodeur cible** (encodeur *EMA*). Il voit la vidéo **complète** (sans
   trous) et produit la « bonne réponse » que le prédicteur doit viser.

> **Lecteur naïf :** « Si l'encodeur cible donne la réponse, pourquoi le modèle
> ne triche-t-il pas en sortant zéro des deux côtés ? La réponse serait
> toujours zéro et la perte toujours parfaite ! »
>
> Excellent ! Cela s'appelle l'**effondrement** (collapse), le danger classique.
> Deux astuces l'empêchent :
>
> - **Stop-gradient :** on n'entraîne jamais l'encodeur cible directement avec
>   la perte. Le modèle ne peut donc pas tirer les deux côtés vers zéro.
> - **EMA (moyenne mobile exponentielle) :** l'encodeur cible est une copie
>   lente de l'encodeur online. Après chaque pas, il bouge un tout petit peu
>   vers l'online : `cible = 0.99925 * cible + 0.00075 * online`. Il suit, mais
>   lentement. La cible reste stable et pleine de sens.

## 5. Le masquage : cacher de la bonne manière

On cache des patches avec le **masquage en tube**. On choisit un rectangle sur
la grille de patches et on le cache dans **toutes** les images du clip (un
« tube » à travers le temps). Les tokens cachés sont ceux à **prédire** ; les
tokens visibles forment le **contexte**.

```
Grille (chaque case est un patch). X = caché (à prédire), . = visible (contexte)
. . . . . .        Le bloc X est caché dans chaque image,
. . X X . .        il forme donc un tube dans le temps.
. . X X . .
. . . . . .
```

## 6. L'idée clé de la version 2.1 : la perte dense

L'ancienne V-JEPA ne vérifiait la prédiction que sur les patches **cachés**. La
version 2.1 a trouvé un problème : les patches **visibles** n'étaient jamais
vérifiés, donc le modèle était paresseux avec eux. Il s'en servait comme d'un
« brouillon » pour stocker des résumés globaux, et il perdait le détail local.
Les cartes de features étaient bruitées.

**La solution (le cœur de V-JEPA 2.1) :** vérifier aussi les patches visibles.
C'est la **perte prédictive dense** :

```
perte totale = perte_predict  +  lambda * perte_context
               (tokens cachés)          (tokens visibles/contexte)
```

- `perte_predict` : erreur de la prédiction sur les tokens **cachés** (ancienne).
- `perte_context` : erreur du prédicteur sur les tokens **visibles** (nouveau !).
- `lambda` : la confiance donnée à la perte de contexte.

Comme chaque token est maintenant vérifié, le modèle doit garder du vrai détail
local partout. Les cartes de features deviennent nettes et pleines de sens.
C'est ce qui débloque les bonnes tâches **denses** : profondeur, segmentation,
suivi.

### 6.1 Pondérer près des trous

Tous les tokens visibles ne comptent pas pareil. Un token visible **juste à
côté** d'un trou est très utile (il aide à deviner). Un token loin compte moins.
On pondère donc la perte de contexte par la distance :

```
poids_i = lambda / racine(distance au token caché le plus proche)
```

Les tokens visibles proches ont un poids plus grand. Petite idée, grand effet
sur la qualité.

### 6.2 Une montée douce de lambda

Si on active la perte de contexte trop fort dès le début, le modèle peut
trouver une astuce paresseuse (juste copier les features visibles) et perdre la
compréhension globale. On fait donc **monter** lambda doucement : il commence à
0, puis grandit lentement jusqu'à sa valeur finale. Lent et régulier.

## 7. L'auto-supervision profonde

Un transformeur a plusieurs couches. Les premières voient le détail fin ; les
profondes voient le grand sens. V-JEPA 2.1 applique la perte non seulement à la
dernière couche, mais aussi à quelques couches **intermédiaires** (4 niveaux).
Cela pousse le bon détail jusqu'au bout du réseau. Dans le code, c'est
`n_output_distillation: 4`.

## 8. Le tokenizer multi-modal

Images et vidéos sont différentes. Une vidéo a le temps ; une image, non. On
utilise donc deux « découpeurs » :

- une convolution **3D** pour la vidéo (elle regarde l'espace **et** le temps),
- une convolution **2D** pour une image seule.

On ajoute aussi un petit marqueur appris de « modalité » pour que le modèle
sache si l'entrée est une image ou une vidéo. Un seul modèle gère les deux,
proprement.

## 9. Comment marche le programme d'entraînement (pas à pas)

Le côté pratique : le programme de ce dépôt. Voici le voyage de tes données.

1. **Trouver les vidéos.** Ton dataset est un dossier ou un fichier zip. Il peut
   avoir des sous-dossiers. Le chercheur parcourt tout et liste chaque vidéo.
2. **Nettoyer et mettre en cache.** Certains fichiers sont cassés. On essaie
   d'ouvrir chacun ; on ne garde que les bons. On sauve la bonne liste dans un
   fichier `*.cache.json` à côté du dataset. La prochaine fois, on lit la liste
   au lieu de tout re-scanner (rapide !).
3. **Séparer.** Le jeu de test est séparé en une partie **validation** (une
   fraction, `val_prob`) et une partie **test final**. La validation sert à
   chaque époque pour suivre les progrès ; le test final sert une fois à la fin.
4. **Transformer et augmenter.** Chaque clip est redimensionné, recadré et
   normalisé. Pendant l'entraînement, on ajoute des changements aléatoires
   (miroir, couleur, flou) pour que le modèle n'apprenne pas par cœur. Tout se
   contrôle depuis le fichier de config.
5. **(Option) HDF5.** Décoder la vidéo est lent. Tu peux calculer tous les clips
   une fois et les stocker dans `train.h5` / `test.h5`. L'entraînement lit alors
   des clips prêts et va plus vite. On bascule avec `use_hdf5: true`.
6. **Le chargeur reprenable.** C'est le point spécial. C'est un chargeur de
   données qui se souvient **où il en est** dans une époque. Si ta machine plante
   après 3 jours, tu ne recommences pas l'époque à zéro. Il sauve l'ordre de
   mélange et la position, donc il continue au même lot exact.
7. **Accumulation de gradient.** Les GPU ont peu de mémoire. Pour agir comme un
   grand lot sans le coût mémoire, on additionne les gradients sur plusieurs
   petits lots, puis on fait un seul pas d'optimiseur. On s'assure aussi qu'une
   accumulation restante en fin d'époque est bien appliquée.
8. **Points de sauvegarde (checkpoints).** Tous les `ckpt_step` pas
   d'optimiseur, on sauve tout : modèle, optimiseur, scheduler, positions des
   chargeurs, compteurs, et l'état d'entraînement. On ne garde que les
   `max_checkpoint` fichiers les plus récents. Si le run s'arrête, il reprend au
   plus récent. **Priorité : checkpoint d'abord, puis des poids initiaux.**
9. **Meilleur modèle.** Après chaque validation, on regarde une métrique
   choisie. Si c'est la meilleure jusque-là, on sauve `best.pt`. Tu choisis la
   métrique et si « plus haut » ou « plus bas » est mieux.
10. **Courbes.** Après chaque époque, on trace les courbes entraînement vs
    validation pour voir si le modèle apprend ou sur-apprend.

## 10. Le dossier de sortie

Chaque run écrit dans `runs/<run_name>/` :

```
runs/
  mon_run/
    train/          # premier run d'entraînement (train2, train3, ... ensuite)
      history.csv         # métriques par époque
      config_used.yaml    # la config exacte utilisée
      weights/
        best.pt           # meilleur modèle
        last.pt           # modèle de la dernière époque
      checkpoints/
        epoch_000.pth ...
      plotes/
        training_history.jpg
      logs/
        train_AAAA-MM-JJ_HH-MM-SS.log
    eval/           # premier run d'évaluation (eval2, eval3, ...)
      results.csv
      renders/            # images de cartes de features PCA
      plotes/
      logs/
```

Si `resume: true` et qu'un checkpoint existe, le programme réutilise le dossier
de run le **plus récent** et continue dedans. Sinon il crée un nouveau dossier
numéroté.

## 11. Suivre l'entraînement

Le programme affiche deux barres de progression :

- une grande barre **époque** (époques faites, temps restant, meilleur score, lr),
- une petite barre **étape** (progression dans la passe train ou validation).

Les barres utilisent `█` pour fait et `░` pour le fond. Les messages normaux
(les lignes `step 400/553824 | loss=...`) sont écrits par le logger et ne
cassent jamais les barres.

## 12. Lire les métriques

- `loss` : la perte dense totale. Plus bas c'est mieux.
- `predict` : erreur sur les tokens cachés.
- `context` : erreur sur les tokens visibles.
- `lambda` : le poids actuel de la perte de contexte (monte pendant le warm-up).
- `feat_std` : à quel point les features sont étalées. Si ça tombe près de zéro,
  le modèle s'effondre (mauvais). Un entraînement sain le garde bien au-dessus.
- `pred_cos` : similarité cosinus entre la prédiction et la réponse. Proche de 1
  c'est mieux.

## 13. Une petite recette à essayer

```bash
# 1. (option) pré-construire les fichiers HDF5 pour la vitesse
buildh5ds --config cpu/configs/hdf5.yaml

# 2. entraîner
trainvjepa --config cpu/configs/train.yaml

# 3. évaluer le meilleur modèle
evalvjepa --config cpu/configs/eval.yaml

# 4. exporter l'encodeur en ONNX (voir les commentaires de export.yaml)
exportw runs/vjepa2_1_cpu/train/weights/best.pt -o encoder.onnx \
    --model-name vit_tiny --crop-size 128 --num-frames 16 --sdpa --single-file
```

Voilà toute l'idée. Prédire le sens de ce qu'on cache, vérifier chaque token,
pousser le signal à travers chaque couche, et garder l'entraînement en
sécurité pour qu'il tourne des semaines et revienne toujours après un plantage.
