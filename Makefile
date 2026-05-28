.PHONY: all data figures clean test fetch eda modeling

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

test:
	pytest tests/

clean:
	rm -rf data/processed/ figures/*.png tables/*.csv
