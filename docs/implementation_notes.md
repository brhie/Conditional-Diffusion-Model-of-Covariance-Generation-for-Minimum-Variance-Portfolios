# Implementation Notes

## Project
Stabilized Conditional Diffusion Forecasting of Next-Month Covariance Distributions
for Global Minimum-Variance Portfolio Optimization

---

## Format

Each entry follows this structure:

### [INnnn] Short title

| Field | Detail |
|---|---|
| Date | YYYY-MM-DD |
| Component | Affected module or spec section |
| Reason | Why the deviation was necessary |
| Original design | What the spec originally said |
| Implementation | What was actually implemented |
| Impact | Likely effect on results |

---

## Entries

### [IN0001] Daily-sliding window for training covariance pairs

| Field | Detail |
|---|---|
| Date | 2026-05-24 |
| Component | `src/covariance.py`, `src/datasets.py`, `scripts/03_build_covariance_datasets.py`, `config/base_config.yaml` (§10, §12, §18) |
| Reason | The original spec generates one `(S_hist, S_fwd)` training pair per group per 21-day rebalance date, yielding ~4,751 pairs. This is insufficient to train a neural network reliably; modern diffusion models typically require tens of thousands of samples to learn a useful conditional distribution. |
| Original design | Spec §10–12 implies one covariance pair per group at each rebalance date (stride = 21 trading days). Expected training set: ~4,751 pairs over 2000–2013. |
| Implementation | A daily-sliding window is applied to training data only. For each 10-stock group formed at rebalance date `t_k`, covariance pairs are computed for every trading day `d` in `[t_k, t_{k+1})` where (a) a complete 126-day lookback exists, (b) the 21-day forward window ends on or before 2013-12-31 (strictly inside training), and (c) all 10 stocks have non-missing returns in both windows. Validation and test data retain the original 21-day stride (non-overlapping evaluation sleeves). The stride is configurable via `config/base_config.yaml`: `covariance_transform.training_window_stride_days: 1`. Set to 21 to revert to the original behaviour. |
| Impact | Training set grows from ~4,751 to ~99,000 pairs (~21× increase). Consecutive pairs share 125 of 126 lookback return days and 20 of 21 forward return days, introducing autocorrelation. The diffusion model is not tested on autocorrelated data (validation and test use non-overlapping periods), so this cannot inflate out-of-sample metrics. The increased data volume is expected to improve model fitting and reduce underfitting on the 55-dimensional covariance representation. The spec's stated limitation about dependence across overlapping training groups (§39.5) applies equally to this sliding-window extension. |

---

### [IN0002] GPU/MPS device support

| Field | Detail |
|---|---|
| Date | 2026-05-24 |
| Component | `src/utils.py`, `src/train.py`, `scripts/05_validate_hyperparameters.py`, `scripts/06_run_final_test.py` |
| Reason | The spec does not prescribe a compute device. Training 4 models (linear schedule, T ∈ {400, 800, 1200, 2000}) for 200 epochs on CPU is prohibitively slow, especially at T=1200 and T=2000 where each training step requires more reverse-step iterations. GPU/MPS reduces total training time from many hours to under one hour. |
| Original design | Spec §15 (fixed architecture) and §18 (fixed training procedure) are device-agnostic. |
| Implementation | Added `get_device()` to `src/utils.py` selecting CUDA (NVIDIA) → MPS (Apple Silicon) → CPU. `DDPMScheduler` tensors are now created on the same device as the model in scripts 05 and 06, preventing device-mismatch errors that would occur when a CPU-resident scheduler is used with a GPU-resident model. `torch.backends.cudnn.benchmark = True` is set on CUDA. `pin_memory=True` and `non_blocking=True` transfers are used with CUDA. Training results (loss values, final weights) are numerically identical across devices because the same seed and deterministic operations are used. |
| Impact | Purely computational — no effect on model outputs, hyperparameter selection, or results. All unit tests pass on MPS (Apple Silicon). |

---

### [IN0003] Reduced model grid: linear schedule only, T ∈ {400, 800, 1200, 2000}

| Field | Detail |
|---|---|
| Date | 2026-05-26 |
| Component | `config/base_config.yaml`, `scripts/04_train_diffusion_models.py`, `scripts/05_validate_hyperparameters.py`, `scripts/07_generate_report.py` |
| Original design | Spec §17–18 defined a 3 × 3 grid: schedule ∈ {linear, quadratic, logarithmic} × T ∈ {25, 50, 100}, yielding 9 models and 181 effective validation configurations. |
| Reason | For a 55-dimensional covariance vector representation, a higher number of diffusion steps T provides finer-grained denoising at the cost of more reverse-step iterations at inference. The linear schedule is a strong default that does not require tuning the schedule shape. Restricting to one schedule type and using larger T values focuses the search budget on the dimension most likely to matter for covariance diffusion quality. |
| Implementation | `beta_schedule_grid` set to `["linear"]`; `diffusion_steps_grid` set to `[400, 800, 1200, 2000]`. This yields 4 models and 81 effective validation configurations (4 × 4α × 5M + 1 boundary). The quadratic and logarithmic schedule implementations are retained in `src/beta_schedules.py` and their unit tests are preserved. |
| Impact | Fewer model variants but larger diffusion step counts. Validation search space reduced from 181 to 81 configurations, lowering hyperparameter search risk. Higher T values increase inference cost (scenario generation time) linearly with T. |

---

### [IN0004] Macro-augmented conditioning vector (55 + K dimensions)

| Field | Detail |
|---|---|
| Date | 2026-05-30 |
| Component | `src/macro_features.py` (new), `src/covariance.py`, `src/datasets.py`, `src/train.py`, `config/base_config.yaml`, `scripts/00_download_macro_data.py` (new), `scripts/03–06` |
| Original design | Spec §14–15 defines the conditioning vector as the 55-dimensional log-vech of the historical 126-day sample covariance. The denoiser input is `[y_s (55) | e(s) (32) | c̃ (55)] = 142 dimensions`. |
| Reason | A pure covariance-vech conditioning vector tells the model nothing about the macro-financial environment that mediates how covariance regimes shift. Decades of empirical research show that the level of implied volatility (VIX), the yield-curve slope, credit spreads, and realized market correlation are strong predictors of future realized covariance structure. Adding these as conditioning inputs gives the diffusion model information that is economically relevant but absent from the historical covariance matrix itself. |
| Implementation | Seven macro features are concatenated to the 55-dim covariance vech, extending the conditioning vector to 55 + K dimensions. Four features are computed directly from the CRSP return panel at every anchor date (no external data needed); three optional features are downloaded from FRED. |

**CRSP-derived features (K = 4, always available):**

| Feature | Formula | Research basis |
|---|---|---|
| `log_mkt_var_21d` | log(Var(EW market ret, 21-day)) | Andersen et al. (2003 JASA) realized variance; Shephard & Sheppard (2010) HEAVY model |
| `log_mkt_var_126d` | log(Var(EW market ret, 126-day)) | Corsi (2009 JFEC) HAR-RV; Ding, Granger & Engle (1993) long memory |
| `mkt_ret_21d` | Cumulative EW return over 21 days | Black (1976) leverage effect; Christie (1982); Engle & Ng (1993) news impact curve |
| `avg_pairwise_corr_126d` | Implied equicorrelation: `ρ = (N·Var(EW)/avg_Var_i − 1)/(N−1)` over 126 days | Pollet & Wilson (2010 J. Finance); Driessen, Maenhout & Vilkov (2012 RFS) |

**External FRED features (K += 3, optional; run `scripts/00_download_macro_data.py`):**

| Feature | FRED Series | Research basis |
|---|---|---|
| `log_vix` | log(VIXCLS) | Engle & Figlewski (2012 J. Derivatives); Bollerslev, Tauchen & Zhou (2009 RFS) variance risk premium |
| `term_spread` | DGS10 − DGS2 | Estrella & Mishkin (1998 Rev. Econ. Stat.); Wright (2006) recession forecasting |
| `credit_spread` | BAMLC0A0CMEY (IG OAS) | Gilchrist & Zakrajsek (2012 AER) excess bond premium; Collin-Dufresne et al. (2001 J. Finance) |

**Resulting model dimensions:**

| Configuration | K | condition_dim | Denoiser input dim |
|---|---|---|---|
| Macro disabled | 0 | 55 | 142 |
| CRSP features only | 4 | 59 | 146 |
| CRSP + FRED (full) | 7 | 62 | 149 |

**Leakage guarantee:** All features at anchor date t are computed from the window [t−T+1, t] only. External features use the value on day t (or the most recent prior observation via forward-fill). No future information is introduced.

**Implementation notes:**
- The `conditioning_scaler` is refitted on the full (n, 55+K) combined vector from training data only. The covariance-vech and macro feature columns are standardized jointly (StandardScaler is per-column, so scale differences do not cause issues).
- `condition_dim` is saved in every model checkpoint under the key `"condition_dim"`. `load_trained_model()` reads this field so models can be reconstructed without hard-coding the dimension.
- `build_daily_sliding_pairs_for_group` in `covariance.py` was updated to return `(anchor_idx, S_hist, S_fwd)` triples instead of `(S_hist, S_fwd)` pairs, so `datasets.py` can look up the macro feature vector at the correct anchor date for each training sample.
- `macro_features.parquet` is cached in `data/interim/` after the first computation. Delete this file to force recomputation (e.g., after changing the external data file).
- If `macro_external.parquet` is absent, script 03 falls back to K=4 CRSP features with a warning. The pipeline runs end-to-end without external data; only the quality of macro conditioning is reduced.

| Impact | Description |
|---|---|
| Conditioning dimension | Increases from 55 to 59 (CRSP-only) or 62 (full) |
| Model parameter count | Increases by (K × hidden_dim) = 4×128=512 or 7×128=896 input-layer weights |
| Training data | No change; same covariance pairs, wider conditioning vector |
| Leakage | None — all features are strictly observable at the anchor date |
| Expected effect | Macro features should reduce forecast uncertainty in high-vol/high-correlation regimes where conditioning on covariance alone is insufficient |

---

### [IN0005] `alternative.ipynb` — macro-feature end-to-end test notebook

| Field | Detail |
|---|---|
| Date | 2026-05-30 |
| Component | `alternative.ipynb` (new) |
| Original design | Spec §30 repository structure does not include exploratory notebooks; `pipeline.ipynb` covers main-pipeline walkthrough only. |
| Reason | After adding macro-augmented conditioning (IN0004), a standalone notebook was needed to (a) verify the full pipeline path from macro-feature computation through model training and scenario generation, (b) provide a quantitative comparison against the baseline (no-macro) model before committing the full pipeline to a rerun, and (c) document regime-sensitivity evidence for the macro conditioning. |
| Implementation | `alternative.ipynb` is a 13-section, 42-cell self-contained notebook. It produces its own artefact directories (`artifacts/scalers_macro/`, `artifacts/models_macro/`) and writes figures to `results/figures/`, so it does not overwrite or contaminate main-pipeline artefacts. |

**Sections and what each tests:**

| Section | Test |
|---|---|
| 1–2 | Load cleaned CRSP + interim data; confirm availability |
| 3 | Run `build_macro_feature_df`; cache `data/interim/macro_features.parquet` |
| 4 | Plot K-feature time series (shaded GFC/COVID); plot cross-correlation matrix over 2000–2013 |
| 5 | Build `(n, 59)` daily-sliding training dataset with `macro_df` vs `(n, 55)` baseline |
| 6 | Fit new `(n, 59)` scalers from training data; plot per-column mean/std sanity check |
| 7 | Build validation datasets (macro and baseline) from first 50 eval sleeves |
| 8 | Train macro model (condition_dim=59) and baseline (condition_dim=55) for 50 epochs / T=400; plot side-by-side loss curves |
| 9 | Architecture summary: parameter counts, input dimensions, forward-pass shape assertions |
| 10 | Generate M=10 conditional covariance scenarios per model; heatmaps of mean-generated vs realized covariance; per-stock conditional-vol plots with scenario bands |
| 11 | Batch Frobenius-loss evaluation over 50 validation observations; distribution and scatter plots |
| 12 | Regime-sensitivity test: sort observations by `log_mkt_var_21d`; compare high-vol vs low-vol conditional-vol forecasts from each model |
| 13 | Print full numerical summary and next-step instructions |

**Configurable constants (top of relevant cells):**

| Constant | Default | Effect |
|---|---|---|
| `SAMPLE_GROUPS` | 300 | Number of training groups used (increase to use all) |
| `TRAIN_EPOCHS` | 50 | Epochs per model (increase to 200 for full training) |
| `T_STEPS` | 400 | Diffusion steps |
| `M` / `M_EVAL` | 10 / 5 | Scenarios for demo / batch evaluation |
| `N_EVAL` | 50 | Validation observations for batch Frobenius evaluation |

**Figures produced (`results/figures/`):**

| Figure | Content |
|---|---|
| `macro_features_time_series.png` | Time series of all K macro features, 2000–2020, shaded GFC/COVID |
| `macro_feature_correlations.png` | Cross-correlation matrix over training period |
| `macro_vs_baseline_training_loss.png` | Epoch-by-epoch loss curves: macro vs baseline |
| `macro_vs_baseline_scenarios.png` | Heat maps: realized vs macro-mean vs baseline-mean covariance |
| `macro_vs_baseline_vol_comparison.png` | Per-stock conditional vol with scenario bands |
| `macro_vs_baseline_frobenius_loss.png` | Frobenius-loss distribution and per-observation scatter |
| `macro_regime_sensitivity.png` | High-vol vs low-vol conditional-vol comparison |

**Impact:**

| Dimension | Note |
|---|---|
| Main pipeline | No impact — separate artefact directories, no shared state |
| Reproducibility | All random seeds are fixed (seed=42 for training; deterministic per-sleeve seeds for generation) |
| Validation integrity | Notebook uses only training and validation data; test period is not touched |
| Leakage | Macro features at each anchor date use only data observable at that date (same guarantee as IN0004) |

---

## CRSP Column Mapping Resolution

After loading the raw CRSP file, the resolved mapping between canonical fields and raw column
names was inspected and logged. See `config/column_mapping.yaml` for the current mapping.

If any of the following fields were unavailable in the extract, the corresponding entry
in `column_mapping.yaml` was set to `available: false`, and the limitation is documented here:

| Canonical Field | Status | Fallback Used |
|---|---|---|
| date | ✓ | — |
| permno | ✓ | — |
| ret | ✓ | — |
| dlret | ✓ | — |
| prc | ✓ | — |
| shrout | ✓ | — |
| shrcd | ✓ | — |
| exchcd | ✓ | — |
| siccd | ✓ | — |

*Update this table after running script 01 on the actual CRSP extract.*

---

## Ridge Epsilon Used

Primary ridge value: ε = 1e-8 (as specified).

If numerical instability required a larger ridge, document the final value here:

| Context | Ridge Value Used | Original | Reason |
|---|---|---|---|
| Default | 1e-8 | 1e-8 | No instability observed |

---

## GMV Solver Fallbacks

Any GMV optimization failures that triggered equal-weight fallback will be logged to:
`results/diagnostics/covariance_repairs.csv`

---

## Ledoit-Wolf Nonlinear Shrinkage

As specified in `base_config.yaml`:
```yaml
benchmarks:
  ledoit_wolf_nonlinear_shrinkage_gmv: false
```

The nonlinear Ledoit-Wolf estimator was not included as a primary benchmark because a
reliable production-quality Python implementation was not available. If included, it would be
implemented using the analytical nonlinear shrinkage formula from Ledoit & Wolf (2020).
