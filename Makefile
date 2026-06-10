.PHONY: all data figures clean test fetch eda modeling poster report

all: data figures modeling

fetch:
	jupyter nbconvert --to notebook --execute notebooks/00_fetch_data.ipynb --inplace

data: data/processed/eeg_train.csv

data/processed/eeg_train.csv: scripts/preprocess.py data/raw/eeg_eye_state.csv
	python scripts/preprocess.py

eda: data
	jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace

figures: eda

modeling: data
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb

poster:
	python scripts/build_poster.py

# Build the paper-style PDF from report/final_report.tex (needs a LaTeX
# toolchain, e.g. texlive-latex-base + texlive-latex-extra). Run twice so
# cross-references and the figure/table numbers resolve.
report:
	cd report && pdflatex -interaction=nonstopmode -halt-on-error final_report.tex \
	  && pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
	rm -f report/final_report.aux report/final_report.log report/final_report.out

test:
	pytest tests/

clean:
	rm -rf data/processed/ figures/*.png tables/*.csv poster/poster.pptx poster/poster_preview.png
