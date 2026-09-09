# LSTM Time-Series Forecasting — Sunspots (PyTorch)

This notebook (`DL_Assignment_4__2_.ipynb`) trains an LSTM network in PyTorch to
forecast the monthly mean sunspot number, and demonstrates a multi-step recursive
forecast beyond the available data.

## Dataset

- **Source:** [Kaggle — `robervalt/sunspots`](https://www.kaggle.com/datasets/robervalt/sunspots)
- **Content:** Monthly mean total sunspot number from 1749–2017 (~3,265 observations),
  sourced from SIDC (Solar Influences Data Analysis Center, Royal Observatory of Belgium).
- The notebook downloads the dataset **directly from Kaggle in code** via the
  `kagglehub` library — no manual download needed. This requires a one-time Kaggle
  API credential setup (see the first markdown cell in the notebook for details).

## Requirements

```bash
pip install numpy pandas matplotlib torch scikit-learn kagglehub
```

A CUDA GPU is used automatically if available, otherwise the notebook falls back to CPU.

## What the notebook does

1. **Setup & reproducibility** — sets random seeds and selects the compute device.
2. **Data loading** — downloads and caches the Kaggle sunspots CSV, parses dates,
   and sorts chronologically.
3. **Train / val / test split** — a **chronological** 70% / 15% / 15% split (no
   shuffling, since this is time-series data). A `MinMaxScaler` is fit only on the
   training set to avoid data leakage.
4. **Sequence creation** — builds sliding-window supervised samples with a
   24-month look-back window (`WINDOW = 24`) to predict 1 step ahead (`HORIZON = 1`).
5. **Model** — a 2-layer LSTM (`hidden_size=64`, `dropout=0.2`) followed by a fully
   connected output head (`LSTMForecaster` class).
6. **Training** — Adam optimizer with a learning-rate scheduler, gradient clipping,
   and early stopping (up to 100 epochs, patience of 10).
7. **Evaluation** — predictions are inverse-transformed back to the original scale
   and scored with RMSE, MAE, and MAPE (MAPE excludes near-zero sunspot months,
   which are common at solar minimum and destabilize the metric).
8. **Recursive multi-step forecasting** — the model's own predictions are fed back
   in as input to forecast further into the future beyond the test set.
9. **Plots** — training/validation loss curves, test predictions vs. actuals, and
   the extended recursive forecast.
10. **Model saving** — the trained weights, architecture config, and scaler
    parameters are saved to `lstm_forecaster_sunspots.pt` for later reuse.

## Reusing the saved model

```python
import torch
from lstm_forecaster import LSTMForecaster  # or paste the class definition

checkpoint = torch.load("lstm_forecaster_sunspots.pt")
model = LSTMForecaster(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Rebuild the scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.min_ = checkpoint["scaler_min"]
scaler.scale_ = checkpoint["scaler_scale"]
```

## Notes / things to try

- Try a longer look-back window (e.g. `WINDOW = 60` or `120`) to see if the model
  captures more of the ~9–14 year sunspot cycle structure.
- MAPE is unstable near solar minimum (values near 0), so RMSE/MAE are the more
  reliable metrics for this dataset.
