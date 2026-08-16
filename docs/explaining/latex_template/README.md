# Machine Learning For Engineer — sources LaTeX

Sources du livre **« Machine Learning For Engineer »** (ML4E), par Doctor Mokira :
un tout-en-un des mathématiques de l'intelligence artificielle et de la science
des données, écrit pour être lu **sans aucun prérequis**.

Le résultat de la compilation est `main.pdf`.

---

## 1. Compiler en une commande

```bash
make          # ou simplement : latexmk
```

`latexmk` enchaîne tout seul les quatre passes nécessaires
(pdflatex → bibtex → pdflatex → pdflatex) et s'arrête dès que les références
et la table des matières sont stables.

> **Important — l'option `-shell-escape`**
> Elle est **obligatoire** : la coloration syntaxique du code (paquet `minted`)
> appelle le programme externe Pygments. Elle est déjà passée par `.latexmkrc`
> et par le `Makefile` ; vous n'avez rien à ajouter. Si vous compilez depuis un
> éditeur (VS Code, TeXstudio…), pensez en revanche à l'activer dans sa
> configuration, sinon la compilation échouera sur la première ligne de code.

---

## 2. Vérifier que la machine est prête

```bash
make check
```

La commande liste les quatre outils requis et dit lesquels manquent :

| Outil        | Rôle                                    | Si manquant                              |
| ------------ | --------------------------------------- | ---------------------------------------- |
| `pdflatex`   | moteur de composition                   | installer TeX Live                       |
| `latexmk`    | orchestrateur des passes de compilation | paquet `latexmk`                         |
| `bibtex`     | bibliographie                           | fourni avec TeX Live                     |
| `pygmentize` | coloration du code (`minted`)           | `pip install Pygments`                   |

### Installation depuis zéro (Debian / Ubuntu)

```bash
sudo apt install texlive-full latexmk python3-pygments
```

`texlive-full` est volumineux (~5 Go) mais évite toute mauvaise surprise. Pour
une installation minimale, il faut au moins : `texlive-latex-extra`,
`texlive-science`, `texlive-fonts-extra`, `texlive-lang-french` et
`texlive-pictures`.

---

## 3. Toutes les commandes disponibles

```bash
make help       # rappelle cette liste
make            # compile le livre complet -> main.pdf
make watch      # recompile automatiquement à chaque sauvegarde (latexmk -pvc)
make manual     # chaîne de compilation « à la main », sans latexmk
make clean      # supprime les fichiers auxiliaires, garde main.pdf
make distclean  # supprime aussi main.pdf
make check      # vérifie la présence des outils
```

`make watch` est le mode de travail recommandé pendant la rédaction : gardez-le
ouvert dans un terminal et le PDF se met à jour à chaque `Ctrl+S`.

### Compiler à la main, si besoin

```bash
pdflatex -shell-escape main
bibtex   main
pdflatex -shell-escape main
pdflatex -shell-escape main
```

Les deux dernières passes ne sont pas du zèle : la première fixe les numéros
de pages, la seconde résout les renvois croisés (`\cref`) et la table des
matières.

### Version de travail filigranée

```latex
\documentclass[brouillon]{ml4ebook}   % dans main.tex
```

Ajoute un filigrane « BROUILLON » sur chaque page — pratique pour faire relire
un chapitre sans qu'on le prenne pour la version finale.

---

## 4. Organisation des fichiers

```
book/
├── main.tex                  ← fichier maître : c'est LUI qu'on compile
├── ml4ebook.cls              ← classe du livre : format, couleurs, titres
├── Makefile                  ← raccourcis de compilation
├── .latexmkrc                ← configuration de latexmk (dont -shell-escape)
│
├── frontmatter/
│   ├── cover.tex             ← page de couverture
│   └── preface.tex           ← avant-propos
│
├── chapters/                 ← un fichier .tex par chapitre
│   └── 01-vecteurs-et-matrices.tex
│
├── style/
│   ├── mb-boxes.sty          ← théorèmes et encadrés pédagogiques
│   ├── mb-figures.sty        ← schémas TikZ et graphiques pgfplots
│   ├── mb-code.sty           ← blocs de code colorés
│   ├── mb-macros.sty         ← raccourcis mathématiques
│   └── mb-bib.sty            ← bibliographie
│
├── bib/references.bib        ← références bibliographiques
└── build/_minted/            ← cache de coloration (généré, non versionné)
```

### Ajouter un chapitre

1. Créer `chapters/07-estimation.tex`, commençant par `\chapter{...}`.
2. Décommenter (ou ajouter) la ligne correspondante dans `main.tex`.

`main.tex` contient déjà le plan complet des cinq parties du livre, chapitres
commentés : il suffit de les activer au fur et à mesure de la rédaction.

### Changer le titre ou l'auteur

Tout est regroupé en tête de `ml4ebook.cls` :

```latex
\newcommand{\mbTitre}{Machine Learning For Engineer}
\newcommand{\mbTitreCourt}{Machine Learning For Engineer}
\newcommand{\mbSousTitre}{...}
\newcommand{\mbAuteur}{Doctor Mokira}
```

Ces macros alimentent d'un coup la couverture, les en-têtes courants et les
métadonnées du PDF.

---

## 5. Écrire dans ce livre

### Les encadrés pédagogiques

Ils portent chacun un pictogramme pour que le lecteur reconnaisse le type d'un
bloc sans le lire.

```latex
\begin{enclair}    ... \end{enclair}       % l'idée en mots de tous les jours
\begin{intuition}  ... \end{intuition}     % l'image mentale à garder
\begin{enia}       ... \end{enia}          % à quoi ça sert en IA ?
\begin{surlesdonnees} ... \end{surlesdonnees}  % appliqué à des données réelles
\begin{prerequis}  ... \end{prerequis}     % ce qu'il faut savoir avant
\begin{piege}      ... \end{piege}         % l'erreur classique
\begin{aretenir}   ... \end{aretenir}      % l'essentiel
\begin{plusloin}   ... \end{plusloin}      % pont vers plus avancé
```

### Les énoncés numérotés

Titre et étiquette sont **tous deux optionnels** :

```latex
\begin{theoreme}                          ... \end{theoreme}
\begin{theoreme}[Titre]                   ... \end{theoreme}
\begin{theoreme}[Titre][mon-label]        ... \end{theoreme}
```

Disponibles : `theoreme`, `definition`, `proposition`, `lemme`, `corollaire`,
`remarque`, `exemple`, `notation`, plus `exercice` et `solution`.

Ils partagent un même compteur par chapitre (Théorème 3.1, Définition 3.2…), ce
qui rend les renvois faciles à retrouver. On y renvoie avec `\cref`, qui écrit
le mot tout seul :

```latex
\begin{theoreme}[Existence et unicité][inversible] ... \end{theoreme}
D'après le \cref{thm:inversible}, ...        % → « le théorème 1.3 »
```

Le préfixe dépend du type : `thm:`, `def:`, `prop:`, `lem:`, `cor:`, `rem:`,
`ex:`, `not:`.

### Les schémas et les graphiques

Le livre s'appuie beaucoup sur l'image : `mb-figures.sty` fournit de quoi
dessiner vite et **de façon homogène** (toutes les figures partagent la palette
et la typographie du livre).

**Un schéma** — dessin TikZ. Le style `mbfig` est déjà appliqué :

```latex
\begin{schema}{La légende de la figure.}[mon-label]
  \axesxy{5}{5}
  \draw[vecteur] (0,0) -- (3,4);
  \node[annotation] at (4,2) {une remarque};
\end{schema}
% renvoi :  \cref{fig:mon-label}
```

**Un graphique de données** — pgfplots, avec le style d'axes `mbaxis` :

```latex
\begin{graphe}{Évolution du coût au fil des itérations.}[courbe-cout]
\begin{axis}[mbaxis, xlabel={Itération}, ylabel={Coût}]
  \addplot coordinates {(0,4.2) (1,2.1) (2,1.3) (3,0.9)};
  \addplot coordinates {(0,4.2) (1,3.0) (2,2.4) (3,2.1)};
  \legend{Pas bien réglé, Pas trop petit}
\end{axis}
\end{graphe}
```

Les couleurs des courbes s'enchaînent toutes seules dans l'ordre de la charte,
et **chaque courbe change aussi de marqueur et de style de trait** : la figure
reste lisible imprimée en noir et blanc, ou lue par une personne daltonienne.

**Un réseau de neurones**, en une ligne :

```latex
\begin{schema}{Un réseau à une couche cachée.}[reseau]
  \reseauneurones[Entrée][Couche cachée][Sortie]{3}{5}{2}
\end{schema}
```

Styles TikZ prêts à l'emploi, à mettre entre crochets sur un `\node` ou un
`\draw` :

| Famille       | Styles                                                                 |
| ------------- | ---------------------------------------------------------------------- |
| Réseaux       | `neurone`, `neurone entree`, `neurone cache`, `neurone sortie`, `lien`  |
| Algorithmes   | `bloc`, `bloc accent`, `bloc calcul`, `donnee`, `decision`              |
| Flèches       | `fleche`, `fleche accent`, `fleche pointillee`, `etiquette`             |
| Géométrie     | `vecteur`, `vecteur accent`, `construction`, `point`, `point vide`      |
| Annotations   | `annotation`, `annotation bulle`, `trait annotation`, `zone`, `accolade`|

Styles d'axes pgfplots : `mbaxis` (défaut, sobre), `mbaxisboite` (encadré, pour
les surfaces), `mbrepere` (axes centrés en $O$, pour la géométrie).

Pour un petit schéma qui doit rester exactement où il est écrit — dans un
encadré, par exemple — utilisez `schemaici`, qui ne flotte pas et n'a pas de
légende.

### Le code

```latex
\begin{code}{python}[Titre optionnel]
import numpy as np
print(np.linalg.norm([3, 4]))
\end{code}

\fichiercode{python}{scripts/descente.py}   % depuis un fichier externe
\co{python}{np.linalg.solve(A, b)}          % code en ligne
```

Les numéros de ligne sont **affichés mais non sélectionnables** : le lecteur
qui copie un extrait obtient le code seul, directement exécutable, sans avoir
à retirer les numéros à la main. C'est le paquet `accsupp` qui déclare au
lecteur PDF un contenu vide à la place du numéro affiché.

### Les notations mathématiques

`mb-macros.sty` fixe les notations **une fois pour tout le livre** — c'est
l'intérêt d'un tout-en-un : le même objet garde le même symbole du chapitre 1
au dernier.

```latex
\R \Rn \Rnn          % ensembles
\vect{x} \transp{A}  % vecteur en gras, transposée
\norm{\vect{x}}      % norme (s'étire : \norm*{...})
\Prob \Esp \Var \Cov % probabilités
\Normale \Bernoulli  % lois usuelles
\params \cout \perte \pas   % apprentissage : θ, J, ℓ, η
\relu \softmax \sigmo       % activations
\grad \argmin \argmax
```

---

## 6. En cas de problème

**« Package minted Error: You must invoke LaTeX with the -shell-escape flag »**
La compilation ne passe pas par `latexmk`/`make`. Utilisez-les, ou ajoutez
`-shell-escape` à la main.

**`pygmentize` introuvable**
`pip install Pygments`, puis vérifiez avec `make check`.

**Des `??` à la place des numéros de théorème ou de figure**
Une seule passe a été faite. `make` en fait autant que nécessaire ; à la main,
il en faut au moins trois.

**Une erreur incompréhensible après un changement de style**
Les fichiers auxiliaires peuvent être désynchronisés :

```bash
make clean && make
```

**La compilation semble bloquée sur le cache de coloration**
Videz-le : `rm -rf build/_minted` puis recompilez.

---

## 7. Notes techniques

- **Format** : 17 × 24 cm, recto-verso, 11 pt. Polices Palatino (texte et
  mathématiques), TeX Gyre Heros (titres), DejaVu Sans Mono (code, fournie par
  le paquet `dejavu` — sur Debian/Ubuntu : `texlive-fonts-extra`).
- **Bibliographie** : `biblatex` avec le moteur **bibtex** (et non biber) —
  c'est voulu, pour ne rien exiger de plus que TeX Live de base.
- **Numéros de ligne non sélectionnables** : obtenu via `/ActualText` (paquet
  `accsupp`). Honoré par Adobe Reader, Okular, pdf.js (Firefox) et poppler
  (`pdftotext`). Un lecteur qui ignorerait cette balise retomberait sur
  l'ancien comportement — le confort est perdu, rien n'est cassé.
- **Correctif babel** : `ml4ebook.cls` contient un contournement d'une
  régression de babel 3.70 (TeX Live 2022) qui cassait `\cref` sur les
  étiquettes contenant un `:`. Le bug est corrigé dans babel ≥ 3.71 ; le
  correctif reste sans effet si votre babel est plus récent, il peut donc être
  laissé en place.
