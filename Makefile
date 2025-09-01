.PHONY: clean run test data features model maze

clean:
	rm -rf data/processed models report/figures logs || true
	mkdir -p data/processed models report/figures logs

data:
	python scripts/01_data_cleaning.py

features:
	python scripts/02_feature_engineering.py

model:
	python scripts/03_train_model.py

maze:
	python scripts/04_maze_model.py

run: data features model maze

test:
	pytest -q
