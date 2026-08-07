pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

rm *.aux *.log *.nav *.out *.snm *.toc
rm *.bbl *.blg