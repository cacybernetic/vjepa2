# Plan de rédaction — Expliquer V-JEPA 2.1 de zéro, jusqu'à la mise en production

> **Papier expliqué** : *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning*
> (Mur-Labadia, Muckley, Bar, Assran, Sinha, Rabbat, LeCun, Ballas, Bardes — FAIR at Meta / Universidad de Zaragoza).
> Code de référence : `https://github.com/facebookresearch/vjepa2`

---

## 1. À qui s'adresse le livre et ce qu'il promet

Le lecteur visé est **un débutant complet en apprentissage automatique**. Il ne connaît ni les réseaux
de neurones, ni les transformeurs, ni l'apprentissage auto-supervisé. Pourtant, il veut trois choses :

1. **Comprendre** ce que fait V-JEPA 2.1 et pourquoi il fait mieux que V-JEPA 2.
2. **Écrire lui-même** le modèle et le programme d'entraînement.
3. **Mettre le modèle en production**, c'est-à-dire le faire tourner pour de vrai, tous les jours,
   au service d'une application.

Le livre doit donc être lisible du début à la fin sans jamais avoir besoin d'ouvrir un autre document.

### Règles de rédaction que chaque chapitre devra respecter

Ces règles ne sont pas des conseils : ce sont des contraintes de fabrication. À la fin de chaque
chapitre, on repasse dessus comme sur une liste de contrôle.

| N° | Règle | Comment on la vérifie |
|----|-------|------------------------|
| R1 | Aucun mot technique n'apparaît sans avoir été défini juste avant | On surligne chaque mot technique du chapitre et on cherche sa définition en amont |
| R2 | On commence toujours par les pré-requis, jamais par la nouveauté | Chaque chapitre s'ouvre sur un encadré « Ce qu'il faut savoir avant » |
| R3 | Un dessin pour chaque idée qui bouge, qui circule ou qui se compare | Aucune section de plus de deux pages sans schéma |
| R4 | Un exemple après chaque notion, avec des chiffres quand c'est possible | Chaque définition est suivie d'un bloc « Exemple » |
| R5 | Des mots simples, ceux de tous les jours | On relit en se demandant : « est-ce que je dirais ça à voix haute ? » |
| R6 | Le texte parle au lecteur, il l'accompagne | Emploi du « nous » et du « vous », questions posées au lecteur |
| R7 | Une phrase de transition à la fin de chaque section, qui annonce la suite | Aucune section ne se termine sur un point sec |
| R8 | Des phrases entières, comme dans un vrai livre, jamais du style « note de code » | Relecture à voix haute obligatoire |

### Un mot d'honnêteté à écrire dans l'avant-propos

Le papier explique **comment entraîner et comment évaluer** le modèle. Il ne parle **presque pas de
mise en production** : pas de serveur, pas de latence, pas de quantification. La partie déploiement du
livre (partie VII) sera donc de deux couleurs, et on le dira franchement au lecteur :

- 🔵 **Ce que le papier dit** : taille des modèles distillés, résolution d'entrée, encodeur gelé,
  vitesse de planification, etc.
- 🟠 **Ce que l'on ajoute en ingénieur** : serveur d'inférence, mesure de latence, surveillance.
  Ce sont des recommandations d'ingénierie classiques, pas des résultats du papier.

> **Transition vers la partie I** : Avant de parler de vidéo, de masques et de transformeurs, il faut
> d'abord se mettre d'accord sur des mots très simples. C'est ce que nous allons faire tout de suite.

---

## PARTIE I — Les fondations : partir vraiment de zéro

*Objectif de la partie : à la fin, le lecteur sait ce qu'est une image pour un ordinateur, ce qu'est
apprendre pour une machine, et il a écrit son premier petit programme d'entraînement.*

### Chapitre 1 — Une image, une vidéo, vues par un ordinateur

- **Ce qu'il faut savoir avant** : rien du tout.
- **Notions à définir, dans cet ordre** : pixel · canaux rouge/vert/bleu · résolution ·
  tableau de nombres (tenseur) · image en niveaux de gris · vidéo = pile d'images · image par seconde
  (fps) · durée d'un clip.
- **Schéma 1.1** : une petite image de 4×4 pixels dessinée en grand, avec le nombre écrit dans chaque
  case, et à côté la même image affichée en couleurs.
- **Schéma 1.2** : une vidéo dessinée comme un empilement de feuilles transparentes, avec l'axe du
  temps qui traverse la pile.
- **Exemple chiffré** : une image de 256 × 256 pixels en couleur contient 256 × 256 × 3 = 196 608
  nombres. Un clip de 16 images à cette taille en contient 16 fois plus, soit 3 145 728 nombres.
  On fait remarquer au lecteur que c'est déjà énorme, et que c'est justement le problème.
- **Vocabulaire à poser ici** : *forme* d'un tenseur, écrite `(temps, hauteur, largeur, canaux)`.
- **Transition** : « Maintenant que nous savons de quoi une image est faite, voyons ce que veut dire
  "une machine apprend". »

### Chapitre 2 — Ce que veut dire « apprendre » pour une machine

- **Pré-requis** : chapitre 1.
- **Notions** : paramètre (ou poids) · modèle · prédiction · erreur (perte) · descente de gradient ·
  taux d'apprentissage · époque · lot (batch) · sur-apprentissage.
- **Schéma 2.1** : la métaphore de la bille dans la vallée. La bille descend la pente ; la pente,
  c'est le gradient ; la taille des pas, c'est le taux d'apprentissage. Trois dessins côte à côte :
  pas trop petit, pas correct, pas trop grand (la bille saute par-dessus la vallée).
- **Application numérique** : on ajuste une droite `y = a·x + b` sur trois points, à la main, sur
  deux pas de descente de gradient. Tous les calculs sont écrits.
- **Exemple de code** : quinze lignes de PyTorch qui font exactement ce calcul, pour que le lecteur
  voie que le code et le calcul à la main donnent le même résultat.
- **Transition** : « Ces exemples avaient une bonne réponse écrite à côté. Mais qui va écrire la bonne
  réponse pour un milliard de vidéos ? Personne. C'est le sujet du chapitre suivant. »

### Chapitre 3 — Apprendre sans étiquettes : l'auto-supervision

- **Pré-requis** : chapitres 1 et 2.
- **Notions** : étiquette · apprentissage supervisé · coût de l'annotation humaine ·
  apprentissage auto-supervisé (SSL) · tâche prétexte · représentation (ou *features*) ·
  espace latent · encodeur.
- **Schéma 3.1** : à gauche, un humain qui colle des étiquettes sur des images une par une ; à droite,
  une machine qui cache elle-même un morceau de l'image et essaie de le deviner. Le message :
  la deuxième colonne passe à l'échelle, pas la première.
- **Schéma 3.2** : l'encodeur dessiné comme un entonnoir. Une image entre, un petit paquet de nombres
  sort. Ce petit paquet, c'est la représentation.
- **Exemple** : le texte à trous. « Le chat boit du ____ ». Pour remplir le trou, il faut avoir compris
  la phrase. C'est exactement l'idée, transposée aux images.
- **Application numérique** : on annonce les chiffres du papier pour donner l'échelle : le jeu de
  données VisionMix163M contient 142 millions d'images et 19 millions de vidéos (soit 1,6 million
  d'heures). Aucune de ces données n'a été étiquetée à la main pour cet entraînement.
- **Transition** : « Il reste une question : dans quel espace la machine doit-elle deviner ? Dans les
  pixels, ou ailleurs ? La réponse à cette question est le cœur de tout le papier. »

### Chapitre 4 — Le transformeur de vision (ViT), brique de base

- **Pré-requis** : chapitres 1 à 3.
- **Notions, dans cet ordre strict** : patch (petit carré d'image) · *tokenisation* (découpage en
  jetons) · vecteur d'embedding · encodage de position · attention · tête d'attention ·
  attention multi-têtes · bloc de transformeur · couche MLP · normalisation de couche (LayerNorm) ·
  connexion résiduelle · profondeur (nombre de blocs) · les tailles ViT-B / ViT-L / ViT-g / ViT-G.
- **Schéma 4.1** : une image découpée en carrés de 16 × 16 pixels, chaque carré numéroté, puis mis à
  la queue leu leu pour former une longue file de jetons.
- **Schéma 4.2** : l'attention expliquée comme une salle de classe. Chaque jeton pose une question
  (*query*), chaque autre jeton présente une étiquette (*key*) et un contenu (*value*). Le jeton
  écoute surtout ceux dont l'étiquette répond à sa question. Les flèches sont plus épaisses là où
  l'attention est plus forte.
- **Schéma 4.3** : un bloc de transformeur ouvert en coupe, avec ses deux étages (attention, puis MLP)
  et ses deux connexions résiduelles.
- **Application numérique, à faire très soigneusement** (c'est la formule que le lecteur réutilisera
  partout ensuite) :
  - Image de 256 × 256 avec des patchs de 16 × 16 → (256 / 16) × (256 / 16) = 16 × 16 = **256 jetons**.
  - Vidéo de 16 images en 256 × 256, patch 16 et *tubelet* de 2 →
    (16 / 2) × (256 / 16) × (256 / 16) = 8 × 16 × 16 = **2 048 jetons**.
  - Vidéo de la phase de refroidissement : 64 images en 384 × 384 →
    (64 / 2) × (384 / 16) × (384 / 16) = 32 × 24 × 24 = **18 432 jetons**. On fait remarquer que le
    coût de l'attention grandit comme le carré de ce nombre : c'est pour cela que la haute résolution
    coûte si cher.
- **Notion à introduire ici et pas ailleurs** : le *tubelet*, c'est-à-dire un patch qui a aussi une
  épaisseur dans le temps (ici 2 images collées ensemble).
- **Encodage de position** : on explique RoPE (encodage de position par rotation) avec un dessin
  d'aiguilles d'horloge que l'on tourne d'un angle proportionnel à la position. On explique tout de
  suite pourquoi ce détail compte pour la production : RoPE permet de changer la résolution d'entrée
  sans réentraîner, en interpolant les fréquences — le papier le fait explicitement pour la
  segmentation et la profondeur.
- **Transition** : « Nous avons l'œil de la machine. Il nous manque son jeu : deviner ce qui est caché.
  C'est le sujet de la partie suivante. »

---

## PARTIE II — Comprendre JEPA et V-JEPA 2, l'ancêtre direct

*Objectif : le lecteur comprend l'architecture qui existait avant, et il pourrait déjà l'implémenter.*

### Chapitre 5 — L'idée JEPA : deviner dans l'espace des idées, pas dans l'espace des pixels

- **Pré-requis** : chapitres 3 et 4.
- **Notions** : architecture prédictive à plongement joint (JEPA) · vue corrompue / vue propre ·
  prédicteur · effondrement des représentations (*collapse*) · gradient bloqué (*stop-gradient*) ·
  moyenne mobile exponentielle (EMA) · encodeur élève (*x-encoder*) et encodeur professeur
  (*y-encoder*).
- **Schéma 5.1** : le schéma central du livre, à soigner. À gauche, l'image masquée entre dans
  l'encodeur élève. À droite, l'image complète entre dans l'encodeur professeur. Au milieu, le
  prédicteur essaie de rejoindre la droite en partant de la gauche. Une barrière dessinée sur la
  flèche de droite représente le gradient bloqué.
- **Schéma 5.2** : comparaison en deux colonnes. Colonne 1 (méthode générative) : le modèle redessine
  les pixels manquants, y compris chaque brin d'herbe, ce qui est un gaspillage. Colonne 2 (JEPA) : le
  modèle ne prédit que *l'idée* de la zone manquante. On dessine littéralement une bulle de pensée.
- **Notion à expliquer avec beaucoup de soin, car c'est le piège classique** : l'effondrement. Si les
  deux encodeurs peuvent tricher, la solution la plus simple est de renvoyer toujours le même vecteur,
  par exemple que des zéros. L'erreur devient nulle et le modèle n'a rien appris. On dessine ce
  scénario, puis on montre les deux garde-fous : le gradient bloqué et la moyenne mobile exponentielle.
- **Application numérique de la EMA** : avec un coefficient de 0,99925, on montre sur trois pas
  successifs comment les poids du professeur suivent ceux de l'élève avec du retard.
  `θ̄ ← 0,99925 · θ̄ + 0,00075 · θ`. On calcule à la main sur un poids valant 1,0 face à un élève à
  2,0, sur trois pas, et on fait remarquer combien c'est lent — et que c'est voulu.
- **Transition** : « Nous savons prédire. Reste à savoir *quoi* prédire exactement, et surtout *où*.
  Voyons la formule que V-JEPA 2 utilisait. »

### Chapitre 6 — La perte de V-JEPA 2, expliquée symbole par symbole

- **Pré-requis** : chapitre 5.
- **Le principe de rédaction ici** : on écrit la formule, puis on la découpe en morceaux, et on
  explique chaque symbole sur une ligne à lui tout seul. Aucun symbole n'est laissé de côté.

  La formule du papier (équation 1) est :

  ```
  L_predict = (1 / |M|) · Σ_{i ∈ M}  ‖ P_φ( E_θ(x), Δ_y )_i  −  sg( E_θ̄(y)_i ) ‖₁
  ```

- **Tableau de décodage à écrire dans le livre** :

  | Symbole | Se lit | Veut dire |
  |---------|--------|-----------|
  | `x` | la vue corrompue | la vidéo dont on a retiré des morceaux |
  | `y` | la vue propre | la même vidéo, complète |
  | `E_θ` | l'encodeur élève | celui qui apprend, ses poids s'appellent θ |
  | `E_θ̄` | l'encodeur professeur | la copie lente de l'élève |
  | `P_φ` | le prédicteur | le petit réseau qui devine |
  | `Δ_y` | les jetons de masque | des jetons vides qui disent « ici il manquait quelque chose, à cette position » |
  | `M` | l'ensemble des positions masquées | la liste des cases cachées |
  | `sg` | *stop-gradient* | « ne fais pas remonter la correction par ce chemin » |
  | `‖ · ‖₁` | la norme L1 | on additionne les écarts en valeur absolue |
  | `1/|M|` | la moyenne | on divise par le nombre de cases cachées |

- **Application numérique complète** : on prend des vecteurs de taille 3 pour que tout tienne sur une
  ligne. Prédiction `(0,5 ; −0,2 ; 1,0)`, cible `(0,4 ; 0,1 ; 0,8)`. Écarts : 0,1 ; 0,3 ; 0,2. Somme :
  0,6. Avec deux jetons masqués et une deuxième erreur de 0,4, la perte vaut (0,6 + 0,4) / 2 = 0,5.
- **Schéma 6.1** : la grille des jetons d'une vidéo, coloriée en deux couleurs. En bleu les jetons
  visibles (le contexte), en gris les jetons masqués. Une accolade indique que **seuls les gris**
  entrent dans la perte. Ce dessin sera repris tel quel au chapitre 8, avec un changement de couleur :
  c'est le pivot de tout le livre.
- **Notion à introduire** : la stratégie de masquage utilisée. Le papier reprend celle de V-JEPA :
  échelle spatiale du masque tirée entre 0,15 et 0,7, échelle temporelle fixée à 1,0 (donc le masque
  traverse toute la durée du clip), rapport d'aspect entre 0,75 et 1,5. On dessine trois exemples de
  masques tirés au sort.
- **Transition** : « Cette formule fonctionne très bien pour reconnaître une action. Nous allons
  maintenant voir qu'elle échoue sur un autre type de tâche — et c'est cet échec qui a donné
  naissance à V-JEPA 2.1. »

---

## PARTIE III — Le problème que V-JEPA 2.1 vient résoudre

*Objectif : le lecteur ressent le problème avant qu'on lui donne la solution. C'est la partie la plus
importante pour la compréhension.*

### Chapitre 7 — Vision « globale » contre vision « dense »

- **Pré-requis** : chapitre 6.
- **Notions** : tâche globale · tâche dense · segmentation sémantique · estimation de profondeur
  monoculaire · suivi d'objet dans une vidéo · sonde linéaire (*linear probe*) · sonde attentive
  (*attentive probe*) · encodeur gelé.
- **Schéma 7.1** : la même photo de rue, avec deux questions posées en dessous.
  Question globale : « que se passe-t-il ici ? » → une seule réponse pour toute l'image.
  Question dense : « à quelle distance est chaque pixel ? » → une réponse par pixel.
- **Schéma 7.2** : le protocole d'évaluation. Un cadenas sur l'encodeur (il est gelé, ses poids ne
  bougent plus), et derrière lui une toute petite couche qu'on entraîne. Message à faire passer : si
  une simple couche linéaire suffit, c'est que l'information était déjà bien rangée dans les
  représentations.
- **Notions de mesure à définir avec un exemple chiffré chacune** :
  - **mIoU** : intersection sur union, moyennée sur les classes. Exemple : la zone prédite couvre 80
    cases, la vraie zone en couvre 100, elles se recouvrent sur 70. Intersection = 70,
    union = 80 + 100 − 70 = 110, donc IoU = 70/110 ≈ 0,636, soit 63,6 %.
  - **RMSE** : racine de la moyenne des carrés des écarts. Exemple sur trois pixels dont les
    profondeurs vraies sont 2,0 / 3,0 / 4,0 m et les prédites 2,2 / 2,7 / 4,3 m. Écarts : 0,2 ; −0,3 ;
    0,3. Carrés : 0,04 ; 0,09 ; 0,09. Moyenne : 0,0733. Racine : ≈ 0,271 m. **Attention** : pour cette
    mesure, plus petit vaut mieux. On le répète, car c'est une source d'erreur fréquente.
  - **J&F-Mean** : moyenne entre la qualité de la zone (J) et la qualité du contour (F).
  - **mAP et Recall@5** : à définir au chapitre 16, quand on en aura besoin.
- **Transition** : « Nous avons les instruments de mesure. Branchons-les sur V-JEPA 2 et regardons ce
  qui se passe. »

### Chapitre 8 — Le diagnostic : pourquoi les cartes de V-JEPA 2 sont bruitées

- **Pré-requis** : chapitres 6 et 7.
- **Notion à introduire d'abord** : l'analyse en composantes principales (PCA), et comment on s'en
  sert pour *voir* des représentations.
  - **Schéma 8.1** : un nuage de points en deux dimensions, avec la direction qui étale le plus les
    points, dessinée comme une flèche. C'est la première composante.
  - **Recette de visualisation, expliquée pas à pas** : on prend la représentation de chaque patch, on
    garde les trois premières composantes principales, on les envoie sur les canaux rouge, vert et
    bleu, et on obtient une image en couleurs. Deux patchs de couleur proche portent une information
    proche. Le papier précise même qu'il teste les six permutations possibles des trois composantes
    sur les trois canaux et garde la plus lisible : on le dit, c'est un détail d'affichage, pas de
    modèle.
- **Le constat, avec les chiffres du papier** : sur une sonde linéaire, V-JEPA 2 obtient
  **22,2 mIoU sur ADE20K** et **0,682 de RMSE sur NYUv2**. À côté, DINOv2 (un modèle d'images) est à
  49,5 mIoU et 0,372 de RMSE. On affiche la comparaison dans un petit tableau et on laisse le lecteur
  conclure lui-même.
- **Schéma 8.2** : reproduction commentée de la figure du papier. Trois lignes : l'image d'origine,
  la carte PCA de V-JEPA 2 (tachetée, brouillonne), la carte PCA une fois la perte de contexte
  ajoutée (nette, les têtes de chiens prennent la même couleur, les roues de voitures aussi).
- **L'hypothèse des auteurs, à expliquer très lentement, car c'est l'idée centrale du papier** :
  1. Le prédicteur reçoit les jetons de contexte **et** les jetons de masque.
  2. Il produit une sortie pour chacun.
  3. Mais la perte ne regarde que les sorties des jetons **masqués**.
  4. Donc les jetons de contexte ne reçoivent **aucune consigne** sur ce qu'ils doivent contenir.
  5. Le modèle en profite : il s'en sert comme d'un cahier de brouillon pour y entasser des
     informations globales, utiles à la prédiction. Ils cessent de décrire *l'endroit* où ils se
     trouvent. Le papier compare ce comportement à celui des « jetons registres ».
  6. Résultat : l'information locale est perdue, et les cartes sont bruitées.
- **Schéma 8.3** : la métaphore à retenir. Chaque jeton de contexte est un élève assis à sa place dans
  une salle. Sans consigne, au lieu de décrire ce qu'il voit depuis sa place, il se met à prendre des
  notes générales sur toute la salle. À la fin, personne ne sait plus ce qu'il y avait à chaque place.
- **Vérification expérimentale** : quand on ajoute la supervision du contexte, ADE20K passe de
  **22,2 à 33,8 mIoU** et NYUv2 de **0,682 à 0,474 de RMSE**. L'hypothèse est confirmée.
- **Transition** : « Le coupable est identifié. Voyons maintenant, ingrédient par ingrédient, comment
  les auteurs le neutralisent. »

---

## PARTIE IV — V-JEPA 2.1, ingrédient par ingrédient

*Objectif : chaque ingrédient est présenté avec le même moule en cinq temps —
(1) le problème restant, (2) l'idée en une phrase, (3) le schéma, (4) la formule décortiquée,
(5) le gain chiffré tiré du tableau d'ablation du papier.*

### Chapitre 9 — Le tableau de bord de la partie IV

On ouvre la partie en donnant au lecteur la carte du chemin : le tableau d'ablation du papier, recopié
en entier, avec une ligne par ingrédient. Le lecteur y reviendra après chaque chapitre pour voir où il
en est.

| Étape ajoutée | IN1K (%) | SSv2 (%) | NYU (RMSE ↓) | ADE20K (mIoU) | Chapitre |
|---|---|---|---|---|---|
| V-JEPA 2 (départ) | 82,2 | 72,8 | 0,682 | 22,2 | ch. 6 |
| + Perte de contexte | 72,6 | 62,5 | 0,474 | 33,8 | ch. 10 |
| + Prédicteur multi-niveaux | 80,8 | 72,1 | 0,463 | 38,6 | ch. 11 |
| + Jeu de données VisionMix163M | 81,6 | 72,6 | 0,418 | 40,8 | ch. 12 |
| + Tokeniseur multi-modal | 81,6 | 72,6 | 0,415 | 41,4 | ch. 13 |
| + Passage à ViT-G (2 milliards) | 84,8 | 76,1 | 0,365 | 47,1 | ch. 14 |
| + Phase de refroidissement | 85,5 | 77,7 | 0,307 | 47,9 | ch. 14 |

**Point de pédagogie à souligner explicitement** : la deuxième ligne fait *chuter* les scores globaux
(72,8 → 62,5 sur SSv2). C'est une régression, et le papier ne la cache pas. On explique au lecteur
que c'est normal en recherche, et que les trois lignes suivantes sont précisément là pour la réparer.
Cela apprend au lecteur à lire une table d'ablation honnêtement.

> **Transition** : « Commençons par l'ingrédient qui donne son nom au papier. »

### Chapitre 10 — Ingrédient 1 : la perte dense (supervision du contexte)

- **L'idée en une phrase** : on applique aussi la perte sur les jetons visibles, pour les forcer à
  rester à leur place au lieu de servir de brouillon.
- **Schéma 10.1** : le schéma 6.1 repris, mais cette fois **les jetons bleus sont eux aussi dans
  l'accolade de la perte**. Le contraste entre les deux dessins doit sauter aux yeux.
- **La formule (équation 2 du papier)**, décortiquée comme au chapitre 6 :

  ```
  L_context = (1 / |C|) · Σ_{i ∈ C}  λ_i · ‖ P_φ( E_θ(x), Δ_y )_i  −  sg( E_θ̄(y)_i ) ‖₁
  ```

  Un seul symbole est nouveau : `C`, l'ensemble des jetons de contexte, et `λ_i`, un poids propre à
  chaque jeton. La perte totale devient `L_dense = L_predict + L_context`.
- **Le piège à expliquer** : si on met un poids trop fort, le modèle trouve une solution paresseuse.
  Il lui suffit de **recopier** ce qu'il a en entrée pour les jetons visibles, puisqu'il les voit déjà.
  Il n'apprend alors plus rien de global. C'est ce qui explique l'effondrement de SSv2.
- **Le tableau des essais de pondération** (recopié du papier), avec un commentaire ligne par ligne :

  | λ | Schéma | Montée en douceur | ADE20K | SSv2 | Lecture |
  |---|--------|-------------------|--------|------|---------|
  | 0 | — | non | 22,2 | 72,8 | le point de départ |
  | 0,05 | constant | non | 26,4 | 71,0 | un peu mieux, presque rien de perdu |
  | 0,2 | constant | non | 29,6 | 62,5 | on gagne, mais on casse |
  | 0,5 | constant | non | 27,5 | 53,8 | trop fort, tout se dégrade |
  | 1,0 | constant | non | 24,6 | 51,1 | le pire des deux mondes |
  | 0,2 | constant | oui | 30,5 | 60,5 | la montée en douceur aide |
  | 0,5 | constant | oui | 32,2 | 61,5 | encore mieux |
  | 0,5 | **pondéré** | oui | **33,8** | **62,5** | la recette retenue |

- **Les deux remèdes, chacun avec son schéma** :
  1. **La montée en douceur** (*warm-up*) : λ démarre à zéro et augmente progressivement, sur les
     époques 50 à 100. Schéma : une courbe qui part à plat puis monte en rampe. Message : on laisse
     d'abord le modèle apprendre à voir globalement, on lui demande le détail ensuite.
  2. **La pondération par la distance** (équation 3) :

     ```
     λ_i = λ / √( d_min(i, M) )
     ```

     où `d_min(i, M)` est la distance, comptée en nombre de blocs, entre le jeton de contexte `i` et
     le jeton masqué le plus proche.
     - **Schéma 10.2** : une grille avec une zone masquée au centre. Autour, des anneaux concentriques
       colorés du plus foncé (juste au bord du masque, poids fort) au plus clair (loin du masque,
       poids faible).
     - **Application numérique, avec λ = 0,5** :
       distance 1 → λ_i = 0,5 / 1 = **0,50** ;
       distance 4 → λ_i = 0,5 / 2 = **0,25** ;
       distance 9 → λ_i = 0,5 / 3 ≈ **0,17** ;
       distance 16 → λ_i = 0,5 / 4 = **0,125**.
     - **En français simple** : plus un jeton visible est proche du trou, plus on lui demande de bien
       décrire son voisinage. C'est logique : c'est lui qui doit assurer la continuité avec la zone
       cachée.
- **Valeurs à retenir pour l'implémentation** : λ = **0,5 pour la vidéo**, λ = **0,7 pour les images**.
- **Gain obtenu** : ADE20K 22,2 → 33,8 ; NYUv2 0,682 → 0,474. Coût : SSv2 72,8 → 62,5.
- **Transition** : « Nous avons gagné en finesse mais perdu en compréhension globale. L'ingrédient
  suivant existe exactement pour récupérer ce que nous venons de perdre. »

### Chapitre 11 — Ingrédient 2 : la supervision profonde (prédicteur multi-niveaux)

- **Le problème restant** : les scores globaux ont chuté, et l'information locale a du mal à remonter
  jusqu'aux dernières couches.
- **Pré-requis à poser avant** : que se passe-t-il dans les couches d'un réseau profond ? Les couches
  du début voient des bords et des textures, celles de la fin voient des objets et des actions. Il
  faut aussi rappeler que dans un réseau convolutif, cette hiérarchie est naturelle, alors qu'un
  transformeur ne l'a pas d'office.
  - **Schéma 11.1** : un empilement de blocs, avec sur le côté des vignettes montrant ce que
    « voit » chaque niveau : traits → coins → roues → voiture.
- **L'idée en une phrase** : au lieu de ne surveiller que la sortie finale, on surveille aussi trois
  couches intermédiaires.
- **La mécanique exacte, telle que décrite dans le papier** :
  1. On prend la sortie de trois blocs intermédiaires de l'encodeur élève, plus la sortie finale.
     Pour un ViT-G, ce sont les blocs **12, 24, 36 et 48**.
  2. On les normalise, puis on les colle bout à bout sur l'axe des canaux (*concaténation*).
  3. Un petit MLP fusionne ce gros vecteur et réduit sa taille.
  4. Le résultat entre dans le prédicteur, avec les jetons de masque.
  5. Le prédicteur ressort **quatre** prédictions, une par niveau surveillé.
  6. On applique les **deux** pertes (masque et contexte) sur **chacun** des quatre niveaux.
- **Schéma 11.2** (le schéma d'architecture le plus important du livre, à dessiner en pleine page) :
  reproduction commentée de la figure d'architecture du papier, avec les quatre branches qui sortent
  de l'encodeur, le MLP de fusion, le prédicteur, et les quatre flèches de perte vers l'encodeur
  professeur.
- **Notions à définir ici** : concaténation sur l'axe des canaux, MLP de fusion, supervision profonde.
- **Application numérique** : si chaque niveau sort des vecteurs de dimension 1408, la concaténation
  des quatre donne 4 × 1408 = 5632 nombres par jeton. C'est le MLP qui ramène ça à une taille
  utilisable. On explique ainsi pourquoi il est là et qu'il n'est pas un détail décoratif.
- **Gain obtenu** : SSv2 62,5 → 72,1 (on récupère presque tout), IN1K 72,6 → 80,8, et **en plus**
  ADE20K 33,8 → 38,6 et NYU 0,474 → 0,463.
- **Une conséquence pratique très importante pour la production, à souligner** : d'après l'annexe du
  papier, sans supervision profonde, il faut aller chercher quatre couches intermédiaires au moment
  de l'évaluation pour avoir de bons résultats (ADE20K : 34,9 avec la dernière couche seule, contre
  39,1 avec quatre couches). Avec la supervision profonde, la dernière couche suffit presque
  (42,0 contre 43,9). **En production, cela veut dire un branchement plus simple et moins de mémoire.**
- **Transition** : « Le modèle est réparé. Passons maintenant à ce qu'on lui donne à manger. »

### Chapitre 12 — Ingrédient 3 : plus de données, et mieux équilibrées

- **Pré-requis** : notion de jeu de données, de mélange, de poids d'échantillonnage.
- **Notions** : curation de données · poids d'échantillonnage · diversité visuelle ·
  vidéo égocentrique / exocentrique.
- **Le tableau de la recette**, recopié du papier avec une colonne de commentaires :

  | Source | Échantillons | Type | Poids V-JEPA 2 | Poids V-JEPA 2.1 | Pourquoi ce changement |
  |---|---|---|---|---|---|
  | SSv2 | 168 K | vidéo égocentrique | 0,056 | **0,170** | beaucoup de mouvement, c'est ce qu'on veut |
  | Kinetics | 733 K | vidéo exocentrique | 0,188 | **0,010** | peu de mouvement utile, on réduit |
  | HowTo100M | 1,1 M | vidéo exocentrique | 0,318 | **0,100** | on réduit aussi |
  | ImageNet | 1 M | images | 0,250 | **0** | remplacé |
  | YT-1B | 19 M | vidéo exocentrique | 0,188 | **0,720** | très varié, on augmente beaucoup |
  | LVD-142M | 142 M | images curées | — | (voie image) | apporte la diversité d'apparence |

- **Le raisonnement à expliquer** : puisque 142 millions d'images apportent déjà toute la richesse
  d'apparence, les vidéos peuvent se spécialiser dans ce qu'elles sont seules à savoir apporter — le
  mouvement. D'où le poids de SSv2 multiplié par trois.
- **Un point technique d'implémentation à ne pas manquer** : les auteurs **ne mélangent pas** images
  et vidéos dans un même lot. Ils dédient des machines aux images et d'autres aux vidéos, puis
  agrègent les gradients avant la mise à jour.
  - **Schéma 12.1** : deux groupes de machines, l'un avec des piles d'images, l'autre avec des
    pellicules de film, et une flèche de fusion des gradients au milieu.
  - **Chiffres du lot global** : **2 304 images** et **128 clips vidéo de 16 images**.
  - **Application numérique** : 128 clips × 2 048 jetons = 262 144 jetons de vidéo par pas, contre
    2 304 × 256 = 589 824 jetons d'image. Le lecteur voit que la charge est du même ordre de grandeur
    des deux côtés, et comprend que ces deux nombres ont été choisis ensemble, pas au hasard.
- **Gain obtenu** : SSv2 72,1 → 72,6, IN1K 80,8 → 81,6, ADE20K 38,6 → 40,8, NYU 0,463 → 0,418.
- **Transition** : « Nous donnons maintenant au modèle des images et des vidéos. Encore faut-il les
  lui présenter correctement. C'est l'objet de l'ingrédient suivant. »

### Chapitre 13 — Ingrédient 4 : le tokeniseur multi-modal

- **Le problème** : V-JEPA 2 n'avait qu'un seul découpeur, en trois dimensions. Pour lui donner une
  image, il fallait la recopier 16 fois pour en faire une fausse vidéo figée. Deux dégâts : un coût de
  calcul multiplié par 8 pour rien, et un mensonge appris au modèle — une image n'est pas une vidéo
  où rien ne bouge.
  - **Schéma 13.1** : à gauche, une photo photocopiée 16 fois avec un gros panneau « gaspillage » ;
    à droite, la même photo traitée une seule fois par une convolution 2D.
- **Pré-requis à poser** : ce qu'est une convolution, avec un schéma de fenêtre qui glisse sur une
  grille, et la différence entre une convolution 2D (qui glisse en hauteur et largeur) et une
  convolution 3D (qui glisse aussi dans le temps).
- **La solution du papier** :
  - convolution **3D de 16 × 16 × 2** pour les vidéos ;
  - convolution **2D de 16 × 16** pour les images ;
  - un **jeton de modalité apprenable**, ajouté à l'entrée de l'encodeur **et** du prédicteur, qui dit
    au modèle « ceci est une image » ou « ceci est une vidéo ».
- **Schéma 13.2** : deux tuyaux d'entrée qui se rejoignent dans un seul encodeur, chacun avec sa
  petite étiquette de modalité.
- **Gain obtenu** : ADE20K 40,8 → 41,4, le reste stable. Plus un gain de calcul qui n'apparaît pas
  dans la table mais qui compte.
- **Conséquence directe pour la production, à mettre en avant** : le même modèle sert pour une photo
  et pour une vidéo, sans bricolage. Le papier s'en sert lui-même pour la profondeur et la
  segmentation (voie image) et pour le suivi d'objet (voie vidéo).
- **Transition** : « Notre recette est complète. Il ne reste plus qu'à la faire grandir. »

### Chapitre 14 — Ingrédients 5 et 6 : la taille du modèle et le refroidissement

- **Notions** : mise à l'échelle (*scaling*) · nombre de paramètres · phase de refroidissement
  (*cool-down*) · calendrier du taux d'apprentissage.
- **Partie A — la taille** : de ViT-L (300 millions de paramètres) à ViT-G (2 milliards).
  Gains : SSv2 72,6 → 76,1 ; IN1K 81,6 → 84,8 ; ADE20K 41,4 → 47,1 ; NYU 0,415 → 0,365.
- **Partie B — le refroidissement** : à la fin, on fait deux choses en même temps. On fait descendre
  le taux d'apprentissage jusqu'à presque zéro, et on augmente la résolution des entrées.
  - Vidéo : de 16 images en 256 × 256 à **64 images en 384 × 384**.
  - Image : de 256 × 256 à **512 × 512**.
  - **Schéma 14.1** : la courbe complète du taux d'apprentissage sur les 147 000 pas, en trois
    couleurs : la montée (0 → 12 000 pas), le plateau (12 000 → 135 000 pas), la descente
    (135 000 → 147 000 pas). Sous la courbe, une bande indiquant la résolution utilisée.
  - Gains : SSv2 76,1 → 77,7 ; NYU 0,365 → **0,307**. Le papier note que c'est la profondeur qui en
    profite le plus, et on explique pourquoi : estimer une distance demande du détail fin, donc des
    pixels.
- **Application numérique de coût** : on reprend le calcul du chapitre 4. Un clip de refroidissement
  fait 18 432 jetons contre 2 048 pour un clip de la phase principale, soit **9 fois plus**. Comme le
  coût de l'attention grimpe au carré, on est autour de 81 fois plus cher par clip. C'est pour cela
  que cette phase ne dure que 12 000 pas contre 135 000.
- **Transition** : « Un modèle de deux milliards de paramètres est magnifique, mais il ne rentrera
  jamais dans votre serveur. Voyons comment le papier le rétrécit. »

### Chapitre 15 — Ingrédient 7 : la distillation, la clé du déploiement

- **Pourquoi ce chapitre est capital pour l'objectif du lecteur** : c'est ici que se joue la mise en
  production. Le papier dit explicitement qu'il a fait grossir le modèle non seulement pour la
  performance, mais aussi **pour pouvoir ensuite le compresser**.
- **Notions** : distillation de connaissances · professeur · élève · moyenne de Polyak.
- **Schéma 15.1** : un grand professeur et un petit élève. L'élève ne regarde pas les étiquettes, il
  regarde les réponses du professeur.
- **Les différences avec l'entraînement principal, à lister clairement** :
  1. La moyenne mobile exponentielle est remplacée par un **professeur gelé** (le ViT-G entraîné).
  2. On garde quand même une moyenne mobile de l'élève, **jamais utilisée dans la perte**, mais qui
     devient **le modèle final publié**. C'est la moyenne de Polyak, à expliquer.
  3. **Pas de supervision profonde** : la perte ne porte que sur la dernière couche du professeur.
  4. Le prédicteur repart de zéro et n'a plus que **12 blocs** au lieu de 24 — le papier précise que
     cela stabilise beaucoup l'entraînement.
  5. La dernière couche du prédicteur est ajustée pour sortir la même dimension que le professeur.
  6. Deux étapes : d'abord basse résolution avec le professeur d'avant refroidissement, puis haute
     résolution avec le professeur d'après refroidissement, l'élève de l'étape 2 partant de la moyenne
     mobile de l'étape 1.
- **Le résultat, en tableau, avec la lecture à faire** :

  | Modèle | Paramètres | Distillé | SSv2 | ADE20K | NYU ↓ | DAVIS |
  |---|---|---|---|---|---|---|
  | ViT-G | 2 milliards | — | 77,7 | 47,9 | 0,307 | 69,0 |
  | ViT-L | 300 millions | non | 74,2 | 42,0 | 0,381 | 64,5 |
  | ViT-L | 300 millions | **oui** | **76,5** | **46,7** | **0,336** | **68,7** |
  | ViT-B | 80 millions | oui | 74,5 | 42,0 | 0,408 | 67,0 |

- **La phrase à retenir pour le lecteur** : un ViT-L distillé, **six fois plus petit**, retrouve
  presque tout le ViT-G. Un ViT-L entraîné seul dans son coin, non. C'est donc le chemin normal vers
  la production.
- **Transition** : « Le modèle est prêt. Regardons ce qu'il sait faire, et jugeons sur pièces. »

---

## PARTIE V — Les résultats, lus avec honnêteté

*Objectif : le lecteur sait quoi attendre du modèle, sur quelles tâches il excelle, et où il est
encore battu. On refuse le ton publicitaire.*

### Chapitre 16 — Les tâches de prédiction du futur

- **Anticipation d'interaction à court terme (Ego4D STA)** : définir la tâche avec un schéma —
  à partir d'une vidéo vue jusqu'à l'instant t, il faut prédire quatre choses à la fois : le cadre
  autour de l'objet qui va être touché, le nom de cet objet, le verbe de l'action, et dans combien de
  temps le contact aura lieu. Définir *bounding box*, *IoU > 0,5*, mAP.
  Résultat : **7,71 mAP All**, contre 5,67 pour la meilleure méthode spécialisée d'avant, soit environ
  **+35 % en relatif**. Le score de localisation seule (50,7 AP) dépasse même DINOv3 à 7 milliards de
  paramètres.
- **Anticipation d'action (EPIC-KITCHENS-100)** : définir *Recall@5* avec un exemple.
  Résultat : **40,8** en action, nouvel état de l'art. À noter honnêtement : en ViT-g (1 milliard),
  V-JEPA 2.1 est **en dessous** de V-JEPA 2 (38,4 contre 39,7). C'est la montée en taille qui fait la
  différence. On le dit.
- **Transition** : « Prédire, c'est bien. Agir dans le monde réel, c'est mieux. »

### Chapitre 17 — Le modèle qui pilote un robot

- **Notions** : modèle du monde · prédicteur conditionné par l'action · commande prédictive (MPC) ·
  méthode d'entropie croisée (CEM) · horizon de planification · zéro-shot.
- **Schéma 17.1** : la boucle de commande prédictive. On imagine plusieurs suites de gestes possibles,
  on simule chacune dans la tête du modèle, on garde celle qui rapproche le plus de l'image but, on
  fait le premier geste, et on recommence.
- **Résultats sur un bras Franka Panda, en zéro-shot** : Reach 100 % dans tous les cas ;
  **Grasp 60 % → 70 %** à réglages égaux, et **jusqu'à 80 %** en planifiant sur 8 pas au lieu de 1 ;
  Pick-and-Place 80 %. Point important à expliquer : V-JEPA 2 se dégradait quand on allongeait
  l'horizon, alors que V-JEPA 2.1 s'améliore. On explique le lien avec la meilleure perception de la
  profondeur.
- **Navigation** : ATE de **5,687 sur Tartan Drive**, et surtout **10,6 secondes de planification au
  lieu de 103,2**, soit dix fois plus vite, grâce à **8 pas de débruitage au lieu de 128**.
  Deux ajustements techniques à mentionner car ils sont indispensables : prédire la représentation
  propre plutôt que le bruit, et échantillonner avec DDIM plutôt que DDPM.
- **Transition** : « Ces succès viennent tous de la même source : des représentations denses de
  meilleure qualité. Mesurons-les directement. »

### Chapitre 18 — Les tâches denses : profondeur, segmentation, suivi

- **Profondeur** : **0,307 de RMSE sur NYUv2** avec une simple sonde linéaire. C'est le meilleur score
  du tableau, devant DINOv3 à 7 milliards de paramètres (0,309), avec un modèle 3,5 fois plus petit.
  KITTI : 2,461.
- **Segmentation** : VOC12 **85,0** ; Cityscapes **73,5** ; ADE20K **47,9**. Face à V-JEPA 2, les
  gains sont énormes : **+20,7 / +27,6 / +23,4 points**.
- **À dire franchement** : sur ADE20K et Cityscapes, DINOv3 reste devant (54,8 et 79,5). Les auteurs
  avancent une explication — VisionMix contient sans doute moins de scènes très encombrées. On garde
  cette nuance, elle apprend au lecteur à choisir son modèle selon sa tâche réelle.
- **Suivi d'objet dans la vidéo** : DAVIS **69,0** et YouTube-VOS **72,7**, deuxième derrière DINOv3.
  Expliquer ici la propagation d'étiquettes par plus proches voisins, qui n'a **aucun paramètre
  appris** — c'est donc une mesure directe de la stabilité des représentations dans le temps.
- **Classification** : SSv2 **77,7** (état de l'art), K400 87,7, IN1K 85,5.
- **Question-réponse sur vidéo** : avec Llama 3.1 8B, PerceptionTest **83,1**. À dire honnêtement :
  le modèle est en retrait sur TemporalBench et TOMATO, et légèrement en dessous des chiffres publiés
  pour V-JEPA 2, qui avait bénéficié de plus de données d'alignement (88,5 millions contre 72,5).
- **Schéma 18.1** : un tableau récapitulatif en couleurs — vert là où V-JEPA 2.1 est premier, orange
  là où il est deuxième, rouge là où il est distancé. Le lecteur doit pouvoir choisir son modèle d'un
  coup d'œil.
- **Transition** : « Vous savez maintenant tout ce que fait ce modèle. Il est temps de le construire
  de vos mains. »

---

## PARTIE VI — Construire le modèle et le programme d'entraînement

*Objectif : à la fin de cette partie, le lecteur a un dépôt de code qui tourne. On construit brique
par brique, et chaque brique est testable seule.*

### Chapitre 19 — Préparer l'atelier

- Environnement : Python, PyTorch, GPU, CUDA. Comment vérifier que tout marche en trois lignes.
- **La conversation honnête à avoir avec le lecteur** : reproduire le ViT-G du papier demande des
  centaines de GPU pendant des semaines. Ce n'est pas à sa portée, et ce n'est pas grave. On lui
  propose donc **trois chemins**, et on lui demande de choisir dès maintenant :
  - **Chemin A — comprendre** : un ViT minuscule sur un tout petit jeu de données, sur un seul GPU.
    Tout le livre est écrit pour que ce chemin fonctionne de bout en bout.
  - **Chemin B — utiliser** : télécharger les poids publiés et n'entraîner que la sonde. C'est le
    chemin réaliste pour 95 % des projets, et celui qui mène le plus vite à la production.
  - **Chemin C — reproduire** : le vrai entraînement, pour information et pour les chiffres exacts.
- **Schéma 19.1** : les trois chemins dessinés comme trois routes, avec le coût en calcul indiqué sur
  chacune.

### Chapitre 20 — Le tokeniseur multi-modal, en code

- Convolution 2D 16 × 16 pour l'image, convolution 3D 16 × 16 × 2 pour la vidéo.
- Jetons de modalité apprenables.
- RoPE 3D et l'interpolation des fréquences quand la résolution change.
- **Test à écrire avec le lecteur** : donner une image 256 × 256, vérifier qu'il sort bien 256 jetons ;
  donner un clip 16 × 256 × 256, vérifier 2 048 jetons. On retrouve les calculs du chapitre 4, ce qui
  boucle la boucle.

### Chapitre 21 — L'encodeur et ses sorties intermédiaires

- Un bloc de transformeur écrit ligne à ligne.
- Comment récupérer proprement les sorties des blocs 12, 24, 36 et 48 sans casser le calcul du
  gradient (les *hooks*, ou plus simplement une liste de retour).
- La normalisation avant concaténation, puis le MLP de fusion.
- **Test** : vérifier qu'on récupère bien quatre tenseurs de la bonne forme.

### Chapitre 22 — Le générateur de masques

- Implémenter la stratégie du papier : échelle spatiale entre 0,15 et 0,7, échelle temporelle à 1,0,
  rapport d'aspect entre 0,75 et 1,5.
- **Le point délicat**, à traiter avec un schéma et un exemple chiffré : calculer `d_min(i, M)`, la
  distance de chaque jeton visible au jeton masqué le plus proche. On montre le calcul sur une petite
  grille 6 × 6 dessinée dans le livre, case par case.
- **Test visuel** : afficher les masques générés, pour que le lecteur *voie* que son code est correct.

### Chapitre 23 — Le prédicteur multi-niveaux

- 24 blocs, dimension d'embedding 384.
- Les jetons de masque apprenables, porteurs de la position.
- Les quatre têtes de sortie.
- **Test** : vérifier qu'on obtient bien quatre sorties, une par niveau.

### Chapitre 24 — Les deux pertes et la boucle d'entraînement

- `L_predict` sur les jetons masqués, `L_context` pondérée sur les jetons visibles, sommées sur les
  quatre niveaux.
- La montée en douceur de λ.
- La mise à jour EMA du professeur, avec le coefficient 0,99925.
- Le gradient bloqué : **où exactement le mettre**, avec un schéma des flèches de gradient, car c'est
  l'erreur la plus fréquente et la plus silencieuse.
- **Schéma 24.1** : un pas d'entraînement complet, numéroté de 1 à 9, du chargement du lot jusqu'à la
  mise à jour du professeur.
- **Tableau complet des hyper-paramètres**, recopié de l'annexe du papier, avec une colonne
  « à quoi ça sert » écrite en français simple :

  | Paramètre | Phase principale | Refroidissement |
  |---|---|---|
  | Nombre d'images par clip | 16 | 64 |
  | Images par seconde | 4,0 | 4,0 |
  | Taille de recadrage vidéo | 256 | 384 |
  | Taille de recadrage image | 256 | 512 |
  | Rapport d'aspect du recadrage | [0,75 ; 1,35] | [0,75 ; 1,35] |
  | Échelle du recadrage | [0,3 ; 1,0] | [0,3 ; 1,0] |
  | Nombre de pas | 135 000 | 12 000 |
  | Pas de montée | 12 000 | — |
  | Lot d'images (global) | 2 304 | 2 304 |
  | Lot de vidéos (global) | 128 | 128 |
  | Taux d'apprentissage de départ | 1e-4 | 6e-4 |
  | Taux d'apprentissage final | 5,25e-4 | 1e-6 |
  | Décroissance des poids | 0,04 | 0,04 |
  | Coefficient EMA | 0,99925 | 0,99925 |
  | Échelle spatiale du masque | [0,15 ; 0,7] | [0,15 ; 0,7] |
  | Échelle temporelle du masque | [1,0 ; 1,0] | [1,0 ; 1,0] |
  | Rapport d'aspect du masque | [0,75 ; 1,5] | [0,75 ; 1,5] |
  | Tubelet | 2 | 2 |
  | Taille de patch | 16 | 16 |
  | Blocs du prédicteur | 24 | 24 |
  | Dimension du prédicteur | 384 | 384 |
  | Blocs intermédiaires (ViT-G) | [12, 24, 36, 48] | [12, 24, 36, 48] |
  | λ de la perte de contexte | 0,5 vidéo / 0,7 image | 0,5 vidéo / 0,7 image |

### Chapitre 25 — Savoir si l'entraînement se passe bien

- Quelles courbes tracer, et surtout **comment les lire**.
- **Le signal d'alarme numéro un : l'effondrement.** Comment le repérer tôt — la perte tombe très vite
  vers zéro, la variance des représentations s'écroule, les cartes PCA deviennent uniformes.
  **Schéma 25.1** : trois courbes de perte côte à côte — saine, effondrée, divergente.
- La visualisation PCA comme outil de diagnostic quotidien : on apprend au lecteur à la regarder
  chaque jour, comme un médecin regarde une radio.
- **Transition** : « Votre modèle est entraîné, et vous savez qu'il est sain. Reste le plus important
  pour vous : le mettre au travail. »

---

## PARTIE VII — De l'entraînement à la production

*Rappel de la règle d'honnêteté : dans cette partie, on marque clairement ce qui vient du papier
(🔵) et ce qui relève de l'ingénierie classique ajoutée par nous (🟠).*

### Chapitre 26 — Choisir son modèle et sa résolution 🔵

- **Le tableau de décision**, construit à partir des chiffres du papier :

  | Votre contrainte | Modèle conseillé | Ce que vous perdez face au ViT-G |
  |---|---|---|
  | Serveur puissant, qualité maximale | ViT-G (2 milliards) | rien |
  | Bon compromis | **ViT-L distillé** (300 millions) | −1,2 sur SSv2, −1,2 sur ADE20K, +0,029 de RMSE |
  | Matériel léger | ViT-B distillé (80 millions) | −3,2 sur SSv2, −5,9 sur ADE20K |

- **Toujours prendre la version distillée** quand elle existe : le ViT-L distillé bat le ViT-L
  entraîné seul sur absolument toutes les mesures.
- **La résolution** : le papier montre qu'entraîner en 384 est meilleur qu'en 256 sur presque tout,
  et surtout sur la profondeur (0,313 → 0,307). Mais en production, la résolution est le premier
  bouton à tourner quand la latence est trop haute. On donne le calcul du coût au lecteur pour qu'il
  arbitre lui-même.
- **Le point RoPE** : grâce à l'interpolation des fréquences, on peut servir une résolution différente
  de celle d'entraînement sans réentraîner. C'est une vraie liberté d'exploitation, à condition de
  mesurer la perte de qualité sur ses propres données.

### Chapitre 27 — Le principe de l'encodeur gelé 🔵

- **L'idée à faire entrer dans la tête du lecteur** : dans tout le papier, l'encodeur ne bouge jamais.
  On lui accroche derrière une petite tête légère, différente selon la tâche.
- **Schéma 27.1** : un tronc commun (l'encodeur gelé) et plusieurs branches (les têtes de tâche).
- **Ce que cela change en production, et c'est énorme** :
  - un seul modèle lourd à charger en mémoire, partagé par toutes les tâches ;
  - les représentations peuvent être **calculées une fois et mises en cache** ;
  - une nouvelle tâche coûte l'entraînement d'une sonde, pas d'un modèle.
- **Les têtes du papier, à décrire chacune** : sonde linéaire pour la profondeur (avec sa BatchNorm
  apprise) et la segmentation ; sonde attentive à 4 blocs pour la classification et l'anticipation ;
  propagation d'étiquettes sans paramètres pour le suivi d'objet ; tête façon Faster R-CNN pour Ego4D.

### Chapitre 28 — Entraîner sa propre sonde sur ses propres données 🔵

- La marche à suivre complète : geler l'encodeur, extraire les représentations, entraîner la sonde,
  balayer les hyper-paramètres.
- **Les plages de balayage du papier, à reprendre telles quelles** :
  profondeur — taux d'apprentissage parmi {1,5e-3 ; 1e-3 ; 8e-4 ; 3e-4 ; 1e-4 ; 8e-5 ; 3e-5} et
  décroissance des poids parmi {1e-3 ; 1e-4} ;
  segmentation — {1,5e-3 ; 8e-4 ; 5e-4 ; 8e-5} et {0,1 ; 0,01 ; 1e-4}, avec AdamW.
- **L'astuce d'ingénierie du papier, à souligner** : on entraîne **plusieurs têtes en parallèle** avec
  des réglages différents (20 têtes = 5 taux × 4 décroissances) et on garde la meilleure. Comme
  l'encodeur est gelé et calculé une seule fois, cela ne coûte presque rien de plus.
  **Schéma 28.1** : un encodeur, vingt petites têtes en éventail, une seule médaille à l'arrivée.

### Chapitre 29 — Servir le modèle 🟠

- **On annonce clairement** : le papier ne traite pas ce sujet. Ce qui suit est de l'ingénierie
  courante, appliquée aux contraintes propres à ce modèle.
- Passer en mode évaluation, désactiver le calcul du gradient, regrouper les requêtes par lots.
- Précision réduite (float16 / bfloat16) et ce qu'il faut vérifier après.
- Export vers un format de service, mise en cache des représentations.
- **Le budget de latence, calculé pour de vrai** : on refait le calcul du nombre de jetons pour la
  configuration visée, on mesure, on affiche le résultat dans un tableau. Le lecteur doit repartir
  avec **sa propre mesure**, pas avec une promesse.
- **Le point le plus important de ce chapitre** : dans un système vidéo qui tourne en continu, l'ennemi
  n'est pas le modèle, c'est le décodage vidéo et le transfert des données. On le dit, avec un schéma
  du tuyau complet, de la caméra à la réponse.

### Chapitre 30 — Surveiller un modèle en service 🟠

- La dérive des données : votre caméra de production ne ressemble pas à YouTube.
- Quoi mesurer en continu : latence, distribution des normes des représentations, taux de rejet.
- **L'outil de surveillance gratuit et spécifique à ce modèle** : afficher régulièrement les cartes
  PCA sur quelques images de production. Si elles redeviennent bruitées comme celles de V-JEPA 2, il
  se passe quelque chose. C'est un contrôle visuel que n'importe qui peut faire, même sans étiquettes.
- Quand faut-il réentraîner la sonde ? Presque toujours la sonde seule, presque jamais l'encodeur.

### Chapitre 31 — Les pièges qui font perdre des semaines 🔵🟠

Un chapitre écrit sous forme de « si ceci, alors cela », chaque piège tenant en une page.

1. Oublier le gradient bloqué → effondrement silencieux, la perte descend à zéro et tout est perdu.
2. Mettre λ trop fort dès le début → les scores globaux s'écroulent (voir la table du chapitre 10).
3. Oublier la montée en douceur de λ → entraînement instable.
4. Traiter une image comme une vidéo figée → gaspillage de calcul et biais appris (chapitre 13).
5. Aller chercher des couches intermédiaires alors qu'on a la supervision profonde → complexité
   inutile en production (chapitre 11).
6. Prendre le ViT-L entraîné de zéro plutôt que le distillé → performances inférieures pour le même
   coût de service.
7. Oublier d'interpoler les fréquences RoPE en changeant de résolution → chute silencieuse de qualité.
8. Choisir V-JEPA 2.1 pour une tâche de segmentation très encombrée sans avoir comparé à DINOv3 → le
   papier lui-même signale ce point faible.
9. Confondre le sens des mesures : mIoU monte, RMSE descend.

---

## PARTIE VIII — Annexes

- **A. Glossaire complet.** Tous les mots techniques du livre, par ordre alphabétique, avec la
  définition en une phrase simple et le numéro du chapitre où ils sont expliqués en entier.
  Cette annexe est la garantie de la règle R1 : si un mot y manque, c'est que le livre a un trou.
- **B. Toutes les formules du papier**, rassemblées, avec le décodage de chaque symbole.
- **C. Toutes les tables de résultats du papier**, recopiées fidèlement, avec la source indiquée.
- **D. Les jeux de données cités**, avec en une ligne ce que chacun contient et à quoi il sert.
- **E. Un cahier d'exercices** : dix questions de calcul (nombre de jetons, valeur de λ_i, mise à jour
  EMA, calcul de mIoU et de RMSE) avec leurs corrigés détaillés.
- **F. Ce que le papier annonce pour la suite** : monter à 7 milliards de paramètres, agrandir encore
  les données, et construire des modèles du monde par-dessus ces représentations. Utile au lecteur
  pour savoir ce qui risque de changer dans un an.

---

## Ordre de rédaction conseillé

Rédiger dans cet ordre, et pas dans l'ordre des chapitres :

1. **Le glossaire (annexe A) en premier.** Il sert de contrat : tant qu'un mot n'y est pas, on ne
   l'écrit pas dans le corps du livre.
2. **Les chapitres 5, 6, 8 et 10** ensuite. C'est le cœur intellectuel du papier. Si ces quatre
   chapitres sont clairs, le livre est réussi ; s'ils sont flous, rien ne rattrapera.
3. **Les schémas 5.1, 6.1, 8.3, 10.2 et 11.2** juste après, avant même de rédiger le reste. Ce sont
   les cinq dessins qui portent tout le livre.
4. **La partie I** en dernier parmi les parties d'explication, une fois qu'on sait exactement quels
   pré-requis sont réellement nécessaires. Cela évite d'expliquer des choses dont on ne se sert jamais.
5. **Les parties VI et VII** en parallèle du code : on écrit le code, on le fait tourner, puis on
   rédige. Jamais l'inverse — sinon on écrit du code qui ne marche pas.

## Contrôle final, avant de considérer le livre terminé

- [ ] Tout mot technique du corps du texte figure au glossaire, et y est arrivé après sa définition.
- [ ] Chaque notion est suivie d'un exemple, et d'une application chiffrée quand c'est possible.
- [ ] Chaque section se termine par une phrase qui annonce la suivante.
- [ ] Aucun passage ne ressemble à un commentaire de code ou à un journal de programme.
- [ ] Les schémas annoncés dans ce plan existent tous et sont commentés dans le texte.
- [ ] Les parties 🔵 et 🟠 sont clairement distinguées dans toute la partie VII.
- [ ] Le code du chemin A tourne du début à la fin sur une machine à un seul GPU.
- [ ] Les faiblesses du modèle sont dites aussi clairement que ses forces.
