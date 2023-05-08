# Makefile for Vesuvius Challenge Ink Detection

.PHONY: dataset train optimisation

dataset:
	python src/data/make_dataset.py

train:
ifdef dry
	python src/models/models.py
else
	python src/models/train_model.py
endif

optimisation:
ifdef sweepid
	wandb agent winged-bull/vesuvius-challenge-ink-detection/$(sweepid)
endif
