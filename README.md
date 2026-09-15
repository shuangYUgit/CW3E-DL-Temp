# CW3E-DL-Temperature

This repository contains Python code for the deep learning algorithm
discussed in the following publication:

Yu, S., Sengupta, A., Dixon, T., Haleakala, K., Chen, A., Hecht, C.,
Delle Monache, L., 2025: *Improved Probabilistic Temperature Forecasts
with Deep Learning in Snowpack-Sensitive Regimes across the western
United States*, submitted to npj Climate and Atmospheric Science,
under review.

Center for Western Weather and Water Extremes, Scripps Institution of
Oceanography, University of California, San Diego, La Jolla, California.

## Contents

1. **`DL-training_optimization.py`** — Hyperparameter optimization for
   the QRNN model and loss function.
2. **`DL-cross-validation.py`** — Leave-one-day-out cross-validation
   training across all 28 forecast lead times.
3. **`DL-evaluation.py`** — Verification metrics (PIT, CRPS/CRPSS,
   sharpness, BS/BSS/REL/RES) computed from the pooled out-of-sample
   predictions.

## Environment

- Python 3.x
- PyTorch, torchsummary
- NumPy, Pandas, GeoPandas
- Matplotlib
