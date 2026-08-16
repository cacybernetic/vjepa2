# =====================================================================
#  Makefile — livre « Mathématiques pour l'intelligence artificielle »
# =====================================================================
MAIN      := main
LATEXMK   := latexmk
PDFLATEX  := pdflatex -shell-escape -interaction=nonstopmode -file-line-error

.PHONY: all pdf watch clean distclean check help

## all / pdf : compile le livre complet (LaTeX -> bibtex -> LaTeX x2)
all: pdf
pdf:
	@mkdir -p build/_minted
	$(LATEXMK) $(MAIN).tex

## watch : recompilation automatique à chaque sauvegarde
watch:
	@mkdir -p build/_minted
	$(LATEXMK) -pvc $(MAIN).tex

## manual : chaîne de compilation « à la main » (sans latexmk)
manual:
	@mkdir -p build/_minted
	$(PDFLATEX) $(MAIN)
	bibtex $(MAIN)
	$(PDFLATEX) $(MAIN)
	$(PDFLATEX) $(MAIN)

## clean : supprime les fichiers auxiliaires (garde le PDF)
clean:
	$(LATEXMK) -c
	@rm -rf build/_minted _minted-* *.bbl *.bcf *.run.xml

## distclean : supprime aussi le PDF final
distclean: clean
	@rm -f $(MAIN).pdf

## check : vérifie la présence des outils nécessaires
check:
	@echo "Vérification de la chaîne de compilation :"
	@command -v pdflatex   >/dev/null && echo "  [ok] pdflatex"   || echo "  [MANQUE] pdflatex"
	@command -v latexmk    >/dev/null && echo "  [ok] latexmk"    || echo "  [MANQUE] latexmk"
	@command -v bibtex     >/dev/null && echo "  [ok] bibtex"     || echo "  [MANQUE] bibtex"
	@command -v pygmentize >/dev/null && echo "  [ok] pygmentize (minted)" || echo "  [MANQUE] pygmentize -> pip install Pygments"

## help : affiche cette aide
help:
	@grep -E '^##' Makefile | sed 's/## //'
