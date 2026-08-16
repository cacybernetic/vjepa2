%%%=====================================================================%%%
%%%  ml4ebook.cls                                                        %%%
%%%  Classe LaTeX pour « Machine Learning For Engineer » (ML4E)          %%%
%%%  Le tout-en-un des mathématiques de l'IA, accessible à tous.         %%%
%%%                                                                      %%%
%%%  Doctor Mokira.                                                      %%%
%%%  Basée sur la classe `book`. Compilation : pdfLaTeX + shell-escape.  %%%
%%%=====================================================================%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ml4ebook}[2026/07/21 v1.1 Classe pour livre de maths accessible]

%----------------------------------------------------------------------
% Identité du livre : un seul endroit à modifier pour le titre, le
% sous-titre et l'auteur. Ces macros alimentent la couverture, les
% en-têtes courants et les métadonnées PDF.
%----------------------------------------------------------------------
\newcommand{\mbTitre}{Machine Learning For Engineer}
\newcommand{\mbTitreCourt}{Machine Learning For Engineer}
\newcommand{\mbSousTitre}{Les mathématiques de l'intelligence artificielle,
  reprises depuis zéro}
\newcommand{\mbAuteur}{Doctor Mokira}

%----------------------------------------------------------------------
% Options de classe (transmises à `book`)
%----------------------------------------------------------------------
\newif\if@mbdraft\@mbdraftfalse
\DeclareOption{brouillon}{\@mbdrafttrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[11pt,twoside,openright]{book}

%----------------------------------------------------------------------
% Encodage, langue, polices
%----------------------------------------------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[french]{babel}
\RequirePackage{lmodern}          % secours (glyphes manquants)
\RequirePackage{mathpazo}         % texte + math façon Palatino : chaleureux, lisible
\RequirePackage{tgheros}          % sans-serif (Helvetica) pour titres/labels
% \RequirePackage[scale=0.84]{FiraMono} % chasse fixe moderne pour le code
\RequirePackage[scaled=0.82]{DejaVuSansMono} % chasse fixe du code
\RequirePackage[scaled]{helvet}   % secours sans
\RequirePackage{microtype}        % typographie fine (protrusion, expansion)
\RequirePackage{textcomp}
\RequirePackage{eurosym}

%----------------------------------------------------------------------
% Mathématiques
%----------------------------------------------------------------------
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{mathtools}
\RequirePackage{amsthm}
\RequirePackage{bm}
\RequirePackage{siunitx}

%----------------------------------------------------------------------
% Mise en page : format livre 17 x 24 cm, marges confortables
%----------------------------------------------------------------------
\RequirePackage[
  paperwidth=17cm, paperheight=24cm,
  inner=2.3cm, outer=1.9cm,
  top=2.2cm, bottom=2.6cm,
  headsep=7mm, footskip=12mm,
  marginparwidth=1.4cm, marginparsep=2mm,
]{geometry}
\RequirePackage{emptypage}        % vraies pages blanches (sans en-tête)
\linespread{1.06}                 % interligne un poil aéré : plus accessible
\setlength{\parskip}{0.35\baselineskip}
\setlength{\parindent}{0pt}

%----------------------------------------------------------------------
% Couleurs de la charte
%----------------------------------------------------------------------
\RequirePackage{xcolor}
\definecolor{mbInk}{HTML}{1B2A41}    % encre principale (titres)
\definecolor{mbBlue}{HTML}{2B4C7E}   % primaire
\definecolor{mbAzure}{HTML}{3A7CA5}  % secondaire
\definecolor{mbAmber}{HTML}{E0A73E}  % accent chaud
\definecolor{mbTeal}{HTML}{2A9D8F}   % exemples
\definecolor{mbGreen}{HTML}{4C956C}  % « à retenir »
\definecolor{mbRose}{HTML}{C1495B}   % pièges / attention
\definecolor{mbViolet}{HTML}{6D4C8C} % notations
\definecolor{mbSlate}{HTML}{55606E}  % gris texte secondaire
\definecolor{mbPaper}{HTML}{FBFAF7}  % fond page très légèrement crème
\definecolor{mbMist}{HTML}{EDF1F5}   % fonds doux

%----------------------------------------------------------------------
% Graphisme
%----------------------------------------------------------------------
\RequirePackage{graphicx}
\RequirePackage{tikz}
\usetikzlibrary{calc,positioning,shapes.geometric,arrows.meta,decorations.pathmorphing,backgrounds,matrix}
\RequirePackage{pgfplots}
\pgfplotsset{compat=1.17}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{multirow}
\RequirePackage{caption}
\RequirePackage{subcaption}
\captionsetup{font=small,labelfont={bf,sf,color=mbBlue},labelsep=period}

%----------------------------------------------------------------------
% Listes, icônes
%----------------------------------------------------------------------
\RequirePackage{enumitem}
\setlist{itemsep=2pt,topsep=3pt,parsep=0pt}
\setlist[itemize,1]{label={\textcolor{mbAmber}{\small$\blacktriangleright$}}}
\setlist[enumerate,1]{label={\textcolor{mbBlue}{\bfseries\arabic*.}}}
\RequirePackage{fontawesome5}

%----------------------------------------------------------------------
% En-têtes / pieds de page
%----------------------------------------------------------------------
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{titletoc}

\newcommand{\headfont}{\sffamily\footnotesize}
\fancypagestyle{mbmain}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
  \fancyhead[LE]{\headfont\textcolor{mbSlate}{\thepage\quad|\quad\mbTitreCourt}}
  \fancyhead[RO]{\headfont\textcolor{mbSlate}{\nouppercase{\leftmark}\quad|\quad\thepage}}
}
\fancypagestyle{plain}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
  \fancyfoot[C]{\headfont\textcolor{mbSlate}{\thepage}}
}
\pagestyle{mbmain}
\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}

%----------------------------------------------------------------------
% Style des titres de chapitre : grand numéro accent + filets
%----------------------------------------------------------------------
\titleformat{\chapter}[display]
  {\normalfont\sffamily}
  {\filright\bfseries\fontsize{13}{13}\selectfont\textcolor{mbAmber}{CHAPITRE}%
     \ \fontsize{58}{58}\selectfont\textcolor{mbAmber}{\thechapter}}
  {6pt}
  {\color{mbInk}\Huge\bfseries\filright}
  [\vspace{4pt}{\color{mbAmber}\titlerule[2pt]}\vspace{2pt}{\color{mbBlue!40}\titlerule[0.6pt]}]
\titlespacing*{\chapter}{0pt}{6pt}{28pt}

\titleformat{\section}
  {\normalfont\sffamily\Large\bfseries\color{mbBlue}}
  {\textcolor{mbAmber}{\rule[-1pt]{3pt}{16pt}}\ \thesection}{0.6em}{}
\titleformat{\subsection}
  {\normalfont\sffamily\large\bfseries\color{mbInk}}
  {\thesubsection}{0.6em}{}
\titleformat{\subsubsection}
  {\normalfont\sffamily\normalsize\bfseries\color{mbAzure}}
  {\thesubsubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{2.4ex plus 1ex minus .2ex}{1.2ex}

%----------------------------------------------------------------------
% Table des matières soignée
%----------------------------------------------------------------------
\RequirePackage{tocbibind}
\RequirePackage[titles]{tocloft}
\renewcommand{\cfttoctitlefont}{\sffamily\Huge\bfseries\color{mbInk}}
\renewcommand{\cftchapfont}{\sffamily\bfseries\color{mbBlue}}
\renewcommand{\cftchappagefont}{\sffamily\bfseries\color{mbBlue}}
\renewcommand{\cftsecfont}{\normalfont\color{mbInk}}
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
\setlength{\cftbeforechapskip}{5pt}

%----------------------------------------------------------------------
% Hyperliens & renvois intelligents
%----------------------------------------------------------------------
\RequirePackage[
  colorlinks=true,
  linkcolor=mbBlue,
  citecolor=mbTeal,
  urlcolor=mbAzure,
  linktoc=all,
  pdfborder={0 0 0},
  unicode=true,
]{hyperref}
\RequirePackage{bookmark}

% Métadonnées du PDF (titre affiché par les lecteurs, recherche, partage).
% Posées après le début du document : babel a alors fini de rendre actifs
% ses caractères, et l'expansion des accents se fait sans erreur.
\AtBeginDocument{%
  \hypersetup{%
    pdftitle={\mbTitre},
    pdfsubject={\mbSousTitre},
    pdfauthor={\mbAuteur},
    pdfkeywords={mathématiques, intelligence artificielle, science des
      données, algèbre linéaire, probabilités, statistique, optimisation,
      apprentissage automatique, réseaux de neurones},
  }%
}
\RequirePackage[french,nameinlink,capitalise]{cleveref}

% Contournement d'une régression de babel 3.70 (TeX Live 2022) : le « : »
% rendu actif par le module français casse \cref sur les étiquettes du type
% « chap:xxx » (elles s'imprimaient en clair : « :xxx@cref ... ?? »).
% Corrigé dans babel >= 3.71 ; en attendant, on re-détokenise l'argument
% des références. Toutes les variantes (\cref, \Cref, formes étoilées,
% \labelcref, \cpageref...) passent par \@cref : on ne patche que lui.
\AtBeginDocument{%
  \let\mb@orig@cref\@cref
  \def\@cref#1#2{\expandafter\mb@cref@aux\expandafter{\detokenize{#2}}{#1}}%
  \def\mb@cref@aux#1#2{\mb@orig@cref{#2}{#1}}%
}

%----------------------------------------------------------------------
% Modules du template
% (csquotes est chargé dans mb-bib, après fvextra de mb-code)
%----------------------------------------------------------------------
\RequirePackage{style/mb-boxes}   % théorèmes, définitions, encadrés pédagogiques
\RequirePackage{style/mb-figures} % schémas TikZ et graphiques pgfplots
\RequirePackage{style/mb-code}    % coloration syntaxique du code
\RequirePackage{style/mb-bib}     % bibliographie (biblatex + bibtex)
\RequirePackage{style/mb-macros}  % raccourcis mathématiques

%----------------------------------------------------------------------
% Filigrane « BROUILLON » optionnel
%----------------------------------------------------------------------
\if@mbdraft
  \RequirePackage{draftwatermark}
  \SetWatermarkText{BROUILLON}
  \SetWatermarkColor{mbMist}
  \SetWatermarkScale{0.9}
\fi

\endinput
