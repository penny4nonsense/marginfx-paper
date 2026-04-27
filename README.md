# marginfx-paper

Research code and paper for *marginfx: Average Marginal Effects for Any Machine Learning Model*.

## Structure

- `data/` — raw and processed datasets
- `simulations/` — Monte Carlo simulation studies
- `empirical/` — real data analyses (UCI Adult, UCI Credit Default, Ames Housing)
- `paper.ipynb` — the paper itself
- `environment.yml` — conda environment for reproducibility

## Setup

```bash
conda env create -f environment.yml
conda activate marginfx-paper
pip install marginfx
```

## Data

Download scripts are in `data/raw/`. Processed data is in `data/processed/`.

## Reproducing results

Run simulation scripts first, then open `paper.ipynb`:

```bash
python simulations/sim1_ame_recovery/run_regression.py
python simulations/sim1_ame_recovery/run_classification.py
python simulations/sim2_se_calibration/run_calibration.py
```

Then open `paper.ipynb` and run all cells.
