# Configuration latexmk pour le projet ML4E
# Usage : latexmk        (compile main.pdf)
#         latexmk -pvc   (recompilation continue)
#         latexmk -c     (nettoyage des fichiers auxiliaires)

$pdf_mode = 1;                     # produire un PDF via pdflatex

# -shell-escape est OBLIGATOIRE pour minted (coloration syntaxique).
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -file-line-error -synctex=1 %O %S';

# biblatex avec le moteur bibtex (biber non requis ici).
$bibtex_use = 2;                   # exécuter bibtex même si les .bib « manquent »

# Fichiers auxiliaires supplémentaires à nettoyer (minted, biblatex...).
$clean_ext = 'bbl run.xml bcf synctex.gz nav snm vrb ' .
             'aux.copy figlist makefile fls fdb_latexmk';

# Répertoire de sortie et de cache.
# (main.pdf est produit à la racine ; le cache minted va dans build/_minted)
$out_dir = '.';

# S'assurer que le dossier de cache minted existe avant compilation.
system('mkdir -p build/_minted');
