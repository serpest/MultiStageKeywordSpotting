pdflatex -shell-escape main.tex
bibtex main
pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex

rm -rf svg-inkscape
rm *.aux *.log *.out
rm *.bbl *.blg
