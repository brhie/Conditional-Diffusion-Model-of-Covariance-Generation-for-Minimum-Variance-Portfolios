# Quant Final Project

This folder contains the final sprint research code for sleeve-level long-only portfolio construction on CRSP daily U.S. equities.

There are two related experiments:

1. **Stabilized conditional diffusion covariance GMV**: generate next-month covariance scenarios with a conditional DDPM, blend them with the historical sample covariance, then solve long-only GMV weights.
2. **Direct MLP softmax weight learner**: skip covariance generation entirely and train an MLP to output long-only sleeve weights directly from Cholesky covariance features plus macro features.

The current documentation has been consolidated here. Long original task/spec files are archived under `docs/archive/` for provenance.

## Project Map

| Path | Purpose |
|---|---|
| `README.md` | Single project guide and navigation point |
| `docs/implementation_notes.md` | Documented deviations, design changes, and limitations |
| `docs/archive/diffusion_original_spec.md` | Original long diffusion implementation spec |
| `docs/archive/mlp_task_spec.md` | Original Codex task spec for `MLP.ipynb` |
| `config/` | YAML configuration and CRSP column mapping |
| `src/` | Reusable pipeline modules |
| `scripts/` | Ordered pipeline scripts |
| `tests/` | Unit and integrity tests |
| `data/raw/` | Raw CRSP input, expected as `crsp_daily.parquet` |
| `data/interim/` | Cleaned CRSP, trading calendar, universes, sleeves, macro features |
| `data/processed/` | Train/validation/test covariance-pair datasets |
| `artifacts/` | Scalers, trained models, selected configs, logs |
| `results/` | Validation/test summaries, portfolios, diagnostics, figures |
| `reports/` | Generated reports and report outputs |
| `pipeline.ipynb` | Main diffusion-pipeline walkthrough |
| `alternative.ipynb` | Macro-conditioning exploratory notebook |
| `MLP.ipynb` | Direct MLP portfolio weight learner |
| `MP-PDF.ipynb` | Marchenko-Pastur covariance denoising benchmark |

## Core Experimental Setup

| Item | Setting |
|---|---|
| Data | CRSP daily U.S. equities |
| Universe | Dynamic market-cap top 500 |
| Portfolio unit | 10 same-industry stocks |
| Lookback window | Previous 126 trading days |
| Holding period | Next 21 trading days |
| Rebalancing | Every 21 trading days |
| Training period | 2000-2013 |
| Validation period | 2014-2020 |
| Test period | 2021-2025 |
| Portfolio type | Long-only, fully invested, sleeve-level |
| Sleeve aggregation | Equal capital across active sleeves |
| Model selection | Validation annualized realized volatility only |

Important split rule: observations are assigned by the complete future holding window, not merely by the formation date. For example, the first test formation date can be in late 2020 if the 21-day holding window belongs to 2021.

## Models

### Diffusion Covariance GMV

At each rebalance date, the diffusion pipeline:

1. Computes the 126-day historical covariance for each 10-stock sleeve.
2. Appends macro conditioning features when enabled.
3. Generates plausible next-month covariance matrices with conditional DDPM.
4. Averages generated covariance matrices in covariance space.
5. Blends the generated covariance with the historical sample covariance:

```text
Sigma_combined = alpha * S_hist + (1 - alpha) * Sigma_diff
```

6. Solves long-only GMV weights and equal-weights capital across active sleeves.

### Direct MLP Weight Learner

`MLP.ipynb` implements a simpler decision-focused model:

1. Converts each 10-by-10 historical covariance matrix into 55 lower-triangular Cholesky features.
2. Appends macro features aligned as of the rebalance date.
3. Fits a feature scaler on training observations only.
4. Trains a PyTorch MLP with a softmax output layer.
5. Optimizes realized next-month portfolio variance plus equal-weight regularization.

The MLP does not forecast a covariance matrix and does not solve a quadratic program for its own weights.

## Macro Features

Macro features are stored in:

```text
data/interim/macro_features.parquet
```

CRSP-derived features are always available when macro conditioning is enabled:

| Feature | Meaning |
|---|---|
| `log_mkt_var_21d` | Log 21-day equal-weight market realized variance |
| `log_mkt_var_126d` | Log 126-day equal-weight market realized variance |
| `mkt_ret_21d` | 21-day equal-weight market cumulative return |
| `avg_pairwise_corr_126d` | 126-day implied average pairwise correlation |

Optional FRED features can be added with `scripts/00_download_macro_data.py`:

| Feature | Source |
|---|---|
| `log_vix` | VIXCLS |
| `term_spread` | DGS10 minus DGS2 |
| `credit_spread` | BAMLC0A0CMEY |

All macro features must be observable at or before the anchor date. Scalers are fit on training data only.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional FRED download support:

```bash
pip install pandas-datareader
```

Expected raw CRSP file:

```text
data/raw/crsp_daily.parquet
```

Minimum expected fields:

```text
PERMNO, date, RET, DLRET, PRC, SHROUT, SHRCD, EXCHCD, SICCD
```

## Run The Diffusion Pipeline

Run scripts from the project root:

```bash
python scripts/00_download_macro_data.py        # optional FRED data
python scripts/01_clean_crsp_data.py
python scripts/02_build_universe_and_groups.py
python scripts/03_build_covariance_datasets.py
python scripts/04_train_diffusion_models.py
python scripts/05_validate_hyperparameters.py
python scripts/06_run_final_test.py
python scripts/07_generate_report.py
```

Main outputs:

| Output | Purpose |
|---|---|
| `artifacts/scalers/conditioning_scaler.pkl` | Training-only conditioning scaler |
| `artifacts/scalers/target_scaler.pkl` | Training-only target scaler |
| `artifacts/models/` | DDPM checkpoints |
| `artifacts/selected_model/` | Validation-selected diffusion config/model |
| `results/validation/` | Validation grid results |
| `results/test/` | Final test results |

## Run The MLP Notebook

Open and run:

```text
MLP.ipynb
```

The notebook uses the processed covariance datasets and existing evaluation sleeves. It saves:

| Output | Purpose |
|---|---|
| `artifacts/scalers/mlp_feature_scaler.pkl` | Training-only MLP feature scaler |
| `artifacts/scalers/mlp_macro_feature_names.json` | Macro feature names used by MLP |
| `artifacts/mlp/selected_mlp_model.pt` | Validation-selected MLP checkpoint |
| `results/mlp/validation_grid_results.csv` | MLP validation table |
| `results/mlp/test_comparison_summary.csv` | MLP and benchmark test comparison |

Quick smoke execution:

```bash
MLP_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute MLP.ipynb \
  --output /private/tmp/MLP_smoke.ipynb \
  --ExecutePreprocessor.timeout=300 \
  --ExecutePreprocessor.kernel_name=python3
```

Normal execution keeps the requested 200-epoch grid and will take longer.

## Run The MP-PDF Notebook

Open and run:

```text
MP-PDF.ipynb
```

The notebook implements Marchenko-Pastur correlation eigenvalue denoising and compares the resulting long-only GMV portfolios against equal weight, sample covariance GMV, Ledoit-Wolf GMV, and EWMA GMV.

It saves:

| Output | Purpose |
|---|---|
| `results/mp_pdf/validation_comparison_summary.csv` | Validation comparison table |
| `results/mp_pdf/test_comparison_summary.csv` | Final test comparison table |
| `results/mp_pdf/test_mp_eigen_diagnostics.csv` | MP eigenvalue threshold diagnostics |

Quick smoke execution:

```bash
MP_PDF_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute MP-PDF.ipynb \
  --output /private/tmp/MP-PDF_smoke.ipynb \
  --ExecutePreprocessor.timeout=300 \
  --ExecutePreprocessor.kernel_name=python3
```

## Benchmarks

The project compares against:

| Benchmark | Description |
|---|---|
| Equal Weight | Equal weights inside each sleeve |
| Sample Covariance GMV | Long-only GMV using the 126-day sample covariance |
| Ledoit-Wolf Linear GMV | Long-only GMV using linear shrinkage covariance |
| EWMA GMV | Included in `MLP.ipynb` when easy to evaluate |
| Diffusion GMV | Validation-selected stabilized conditional diffusion model |
| MLP Softmax | Direct long-only learned weights from `MLP.ipynb` |

## Tests

Run:

```bash
pytest tests/ -v
```

The tests cover covariance transforms, GMV weights, universe/split integrity, group construction, turnover, diffusion shapes, and reproducibility.

## Key Notes

See `docs/implementation_notes.md` for full details. Current notable decisions:

| ID | Decision |
|---|---|
| IN0001 | Daily-sliding training windows for diffusion training pairs |
| IN0002 | CUDA/MPS/CPU device selection |
| IN0003 | Reduced diffusion grid: linear schedule with T in `{400, 800, 1200, 2000}` |
| IN0004 | Macro-augmented conditioning vector |
| IN0005 | `alternative.ipynb` macro-feature end-to-end test notebook |

## Citation

Please cite the original CRSP data source:

> Center for Research in Security Prices (CRSP). CRSP US Stock Database. University of Chicago Booth School of Business.
