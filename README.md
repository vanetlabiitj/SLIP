# SLIP: Spatio-Temporal Long-Tail Incident Prediction

This repository contains a modular implementation of SLIP, a framework for handling long-tail multi-label classification in spatio-temporal incident prediction.

## Project Structure

- `data_loader.py` — Functions for loading and preprocessing data.
- `model.py` — Model architecture and related components.
- `train.py` — Training and evaluation routines.
- `main.py` — Main script to execute training and experiments.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python main.py --config_filename ./dataset/LA/LA_crime.yaml
```

