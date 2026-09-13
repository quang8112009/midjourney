MAIN = main

.PHONY: all pdf clean

all: pdf

pdf: $(MAIN).tex references.bib
	@if command -v tectonic >/dev/null 2>&1; then \
		tectonic $(MAIN).tex; \
	elif command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf $(MAIN).tex; \
	else \
		pdflatex -interaction=nonstopmode $(MAIN).tex; \
		bibtex $(MAIN); \
		pdflatex -interaction=nonstopmode $(MAIN).tex; \
		pdflatex -interaction=nonstopmode $(MAIN).tex; \
	fi

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).xdv $(MAIN).fls $(MAIN).fdb_latexmk
