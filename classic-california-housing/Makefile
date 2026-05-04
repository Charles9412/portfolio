.PHONY: run data train eval

run: data train eval

# Build dataset splits
data:
	python src/data.py

# Train candidate models and save best
train:
	python src/train.py

# Evaluate on held-out test set
eval:
	python src/evaluate.py
