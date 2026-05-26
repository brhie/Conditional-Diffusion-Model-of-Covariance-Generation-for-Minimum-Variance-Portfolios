# Claude Code Implementation Specification
## Stabilized Conditional Diffusion Forecasting of Next-Month Covariance Distributions for Global Minimum-Variance Portfolio Optimization

---

## 0. Your Role

You are implementing a complete, reproducible quantitative research pipeline.

The project tests whether a **stabilized conditional diffusion covariance estimator** improves the realized out-of-sample volatility of Global Minimum-Variance (GMV) portfolios constructed from CRSP daily U.S. equity data.

Implement the project exactly as specified below. Do not silently change the model, universe, split logic, portfolio construction, validation metric, or benchmark rules.

If an implementation detail must be modified because of unavailable CRSP columns, numerical instability, or infeasible compute cost:

1. Record the change in `reports/implementation_notes.md`.
2. Explain the reason.
3. Preserve the original design where possible.
4. Never inspect the test period to make implementation or model-selection decisions.

---

# 1. Final Research Title

## Stabilized Conditional Diffusion Forecasting of Next-Month Covariance Distributions for Global Minimum-Variance Portfolio Optimization

---

# 2. Core Research Objective

At each portfolio rebalance date \(t\), an investor observes the historical covariance matrix estimated from the previous 126 trading days. However:

1. The historical sample covariance is noisy because it is estimated from finite observations.
2. The covariance matrix relevant for the next monthly holding period may differ from the historical covariance.
3. GMV optimization is highly sensitive to covariance-estimation error.

The project therefore uses a **conditional diffusion model** to generate multiple plausible covariance matrices for the next 21 trading days, conditional on the previous 126-day sample covariance. The mean generated covariance is then blended with the historical sample covariance through a stability weight \(\alpha\).

For a same-industry 10-stock sleeve \(g\) at rebalance date \(t\):

\[
S_{g,t}^{126}
=
\operatorname{Cov}
\left(
r_{g,t-125},\ldots,r_{g,t}
\right)
\]

is the observed historical covariance input.

The model learns to generate:

\[
\widehat{\Sigma}_{g,t+1:t+21}^{(1)},
\widehat{\Sigma}_{g,t+1:t+21}^{(2)},
\ldots,
\widehat{\Sigma}_{g,t+1:t+21}^{(M)}
\]

as plausible covariance scenarios for the subsequent 21-trading-day holding period.

The diffusion-implied expected next-month covariance is:

\[
\widehat{\Sigma}_{g,t+1}^{\text{diff}}(M)
=
\frac{1}{M}
\sum_{m=1}^{M}
\widehat{\Sigma}_{g,t+1:t+21}^{(m)}
\]

The final stabilized covariance estimate is:

\[
\boxed{
\widehat{\Sigma}_{g,t+1}^{\text{combined}}
=
\alpha S_{g,t}^{126}
+
(1-\alpha)
\widehat{\Sigma}_{g,t+1}^{\text{diff}}(M)
}
\]

The combined covariance matrix is used to construct a long-only GMV portfolio.

The central empirical question is:

\[
\boxed{
\text{Does the stabilized conditional diffusion covariance estimator produce lower future realized GMV volatility than conventional estimators?}
}
\]

---

# 3. Locked Final Research Design

| Component | Final Setting |
|---|---|
| Data source | CRSP daily U.S. equity data |
| Parent universe | Dynamic market-cap top 500 at each rebalance date |
| Security type | U.S. ordinary common shares |
| Portfolio unit | 10 same-industry stocks |
| Asset ordering within sleeve | Descending market capitalization at formation date |
| Input window | Previous 126 trading days |
| Forecast / holding horizon | Subsequent 21 trading days |
| Rebalancing | Every 21 trading days; non-overlapping holding periods |
| Conditioning input | Historical 126-day sample covariance |
| Diffusion target | Subsequent 21-day realized covariance proxy |
| Model | Conditional DDPM with conditional MLP denoiser |
| Model-generated object | Distribution of plausible next-month covariance matrices |
| Diffusion expected covariance | Arithmetic mean of generated covariance matrices |
| Stability blend | \(\alpha S_t^{126} + (1-\alpha)\widehat{\Sigma}_{t+1}^{\text{diff}}\) |
| Portfolio rule | Long-only GMV within each sleeve |
| Aggregate investment strategy | Equal capital across non-overlapping evaluation sleeves |
| Training period | 2000-01-01 to 2013-12-31 |
| Validation period | 2014-01-01 to 2020-12-31 |
| Test period | 2021-01-01 to 2025-12-31 |
| Sole hyperparameter-selection metric | Annualized realized validation GMV portfolio volatility |
| Primary final model | Single best validation-selected configuration |
| Required benchmarks | Equal Weight, Sample Covariance GMV, Ledoit-Wolf Linear Shrinkage GMV |
| Preferred benchmark | Ledoit-Wolf Nonlinear Shrinkage GMV if implementation is reliable |

---

# 4. Correct Methodological Interpretation

## 4.1 What the Model Does

The model learns:

\[
p_{\theta}
\left(
\Sigma_{g,t+1:t+21}
\mid
S_{g,t}^{126}
\right)
\]

In words:

> Conditional on a historical 126-day sample covariance matrix, generate plausible covariance matrices for the next 21-trading-day investment period.

The portfolio then uses a stabilized estimate combining:

- observable historical covariance information; and
- the diffusion-generated expected future covariance.

## 4.2 What the Model Does Not Claim

Do **not** state that:

- \(S_{g,t}^{126}\) is the true covariance matrix;
- \(S_{g,t+1:t+21}^{21}\) is the true future covariance matrix;
- the model directly recovers latent true covariance;
- diffusion eliminates all covariance estimation noise;
- validation-period portfolio selection is fully end-to-end decision-focused training.

Correct terminology:

> The model is a conditional diffusion covariance forecaster with portfolio-objective-aligned hyperparameter selection.

The neural model is trained using diffusion loss. Hyperparameters are selected using the downstream GMV realized-volatility objective.

---

# 5. Data Source: CRSP Daily U.S. Equity Data

## 5.1 Expected Raw Variables

The exact raw column names depend on the extracted CRSP file. Implement a configurable mapping from available raw columns into canonical fields.

| Canonical Field | Typical CRSP Source Variable | Use |
|---|---|---|
| `date` | `date` | Trading date |
| `permno` | `PERMNO` | Permanent security identifier |
| `ret` | `RET` | Daily return |
| `dlret` | `DLRET` | Delisting return adjustment |
| `prc` | `PRC` | Market-cap calculation |
| `shrout` | `SHROUT` | Market-cap calculation |
| `shrcd` | `SHRCD` | Common-share filter |
| `exchcd` | `EXCHCD` | Major U.S. exchange filter |
| `siccd` | `SICCD` | Industry construction |

Before proceeding, inspect the raw dataset schema and write the resolved mapping to:

```text
config/column_mapping.yaml
```

## 5.2 Security Eligibility Filter

Retain ordinary common shares only:

\[
\texttt{SHRCD} \in \{10,11\}
\]

If exchange codes are available, retain major U.S. exchanges:

\[
\texttt{EXCHCD} \in \{1,2,3\}
\]

If either field is unavailable in the provided CRSP extract, document this limitation rather than inventing a replacement.

## 5.3 Daily Total Return Including Delisting

Construct a total daily return series that accounts for delisting returns.

If both ordinary return and delisting return exist:

\[
r_{i,t}^{\text{total}}
=
(1+r_{i,t})(1+dlr_{i,t})-1
\]

Rules:

| Available Fields | Total Return Rule |
|---|---|
| `RET` available, `DLRET` missing | Use `RET` |
| `RET` missing, `DLRET` available | Use `DLRET` |
| Both available | Compound using the formula above |
| Both missing | Leave return missing |

Do not replace missing returns with zero.

## 5.4 Market Capitalization

Construct daily market capitalization:

\[
ME_{i,t}
=
|PRC_{i,t}|
\times
SHROUT_{i,t}
\]

Use absolute value of CRSP price because negative CRSP prices can represent bid/ask-derived price quotations rather than negative equity values.

Require:

\[
ME_{i,t} > 0
\]

at a rebalance date for a security to enter the candidate universe.

---

# 6. Date Splits and Strict Leakage Rules

## 6.1 Time Split

| Split | Date Range | Purpose |
|---|---|---|
| Training | 2000-01-01 to 2013-12-31 | Fit conditional diffusion models |
| Validation | 2014-01-01 to 2020-12-31 | Select \(\beta\) schedule, \(\alpha\), \(T\), and \(M\) using GMV realized volatility only |
| Test | 2021-01-01 to 2025-12-31 | Final untouched evaluation |

## 6.2 Observation Assignment Rule

Assign an observation to a split based on the dates of its **future 21-day target / holding window**, not the dates of its input lookback window.

For a rebalance date \(t\):

\[
\text{Input window}
=
[t-125,\ldots,t]
\]

\[
\text{Holding / target window}
=
[t+1,\ldots,t+21]
\]

Rules:

| Split | Eligible Observation Condition |
|---|---|
| Training | Every future holding-window date lies within 2000–2013 |
| Validation | Every future holding-window date lies within 2014–2020 |
| Test | Every future holding-window date lies within 2021–2025 |

The input lookback window may extend into the prior split because those data are observable at the portfolio formation date.

Examples:

- A January 2014 validation holding period may use input returns from late 2013.
- A January 2021 test holding period may use input returns from late 2020.
- A holding period crossing from December 2020 into January 2021 belongs to neither validation nor test and must be dropped.

## 6.3 Prohibited Leakage

Do not use the test period to:

- fit feature scalers;
- choose covariance transforms;
- choose model architecture;
- select hyperparameters;
- select \(\alpha\);
- select \(M\);
- select transaction costs;
- select portfolio constraints;
- decide whether to include or exclude benchmarks;
- choose random seeds;
- repair a poor model after inspecting test performance.

All transformations and model choices must be locked before test evaluation.

---

# 7. Rebalance Date Schedule

## 7.1 Main Schedule

Use non-overlapping 21-trading-day holding periods.

Let valid rebalance dates be:

\[
t_1,t_2,\ldots,t_J
\]

such that:

\[
t_{j+1}=t_j+21 \text{ trading days}
\]

At each rebalance date:

| Date Range | Use |
|---|---|
| \(t-125,\ldots,t\) | Compute historical conditioning covariance |
| \(t+1,\ldots,t+21\) | Compute future realized covariance proxy and portfolio realized returns |

## 7.2 Calendar Construction

Build a canonical trading calendar from the CRSP daily dataset. Use actual trading dates rather than calendar-day arithmetic.

Implement:

```python
def build_non_overlapping_rebalance_dates(
    trading_dates,
    start_target_date,
    end_target_date,
    lookback_days=126,
    horizon_days=21,
) -> list[pd.Timestamp]:
    """
    Return rebalance dates whose complete 126-day input window exists
    and whose complete 21-day future target/holding window is inside
    the specified split.
    """
```

---

# 8. Dynamic Market-Cap Top-500 Universe

## 8.1 Universe Definition

At each rebalance date \(t\):

1. Start from eligible ordinary common shares.
2. Require an observable positive market capitalization at \(t\).
3. Require complete historical daily returns over the previous 126 trading dates.
4. Rank securities by market capitalization at \(t\).
5. Retain the top 500 eligible securities.

\[
\mathcal{U}_t
=
\operatorname{Top500}_{ME}
\left(
\text{eligible securities at }t
\right)
\]

## 8.2 Future Return Availability

For construction of training, validation, and test observations, a sleeve requires realized future returns across its 21-day holding period.

Handle delistings using total returns including `DLRET` when available. If future returns remain unobservable after delisting adjustment, drop that specific sleeve-date observation and log the reason.

## 8.3 Survivorship-Bias Rule

The universe must be rebuilt dynamically at every rebalance date.

Never:

- define the universe using surviving securities as of 2025;
- require a security to remain listed through the full study;
- rank securities using future market caps;
- eliminate past firms because they later delisted.

---

# 9. Same-Industry 10-Stock Sleeves

## 9.1 Industry Assignment

Use contemporaneously available SIC code information.

Primary industry mapping:

\[
\text{industry}_{i,t}
=
\left\lfloor
\frac{SICCD_{i,t}}{100}
\right\rfloor
\]

This produces broad two-digit SIC industries.

Implement the mapping as a configurable function:

```yaml
industry:
  method: "sic_2digit"
  group_size: 10
```

Do not use a future industry classification to assign a stock in a past period.

## 9.2 Fixed Matrix Dimension

Every model input and target concerns exactly 10 stocks.

Thus:

\[
N=10
\]

and each covariance matrix is:

\[
10 \times 10
\]

with:

\[
\frac{10(10+1)}{2}=55
\]

unique entries.

## 9.3 Asset Ordering Within Sleeve

Order stocks within every sleeve by descending market capitalization at the rebalance date:

\[
ME_{1,t} \geq ME_{2,t} \geq \cdots \geq ME_{10,t}
\]

This determines covariance row and column positions.

Without this rule, matrix coordinate \((i,j)\) would refer to arbitrary stock pair positions across samples, making neural learning inconsistent.

---

# 10. Training Group Construction

## 10.1 Purpose

During training, generate many covariance input-target pairs by sampling 10-stock same-industry groups from the dynamic top-500 universe.

Target approximately:

\[
40\text{--}50
\]

training groups per rebalance date, where feasible.

## 10.2 Rules

| Rule | Training Setting |
|---|---|
| Universe | Dynamic top-500 at formation date |
| Industry restriction | All 10 stocks share the same industry classification |
| Sleeve size | 10 |
| Overlap between sampled groups | Allowed during training |
| Target sampled groups per date | 50 where feasible |
| Randomness | Deterministic seed |
| Storage | Save every group membership record |

## 10.3 Sampling Logic

Implement:

```python
def sample_training_groups(
    universe_at_date: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    group_size: int = 10,
    target_groups: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample same-industry 10-stock groups for diffusion training.
    Overlapping groups are allowed in training.
    Each group must have ordered PERMNOs by descending formation-date market cap.
    Return group_id, date, industry, position, PERMNO, market_cap.
    """
```

If fewer than 50 valid groups can be sampled at a date, use all feasible sampled groups and record the count.

## 10.4 Dependence Caveat

Overlapping training groups increase the number of model observations, but they are not fully independent because they may share:

- stocks;
- industries;
- rebalance dates;
- market-wide shocks.

The final report must acknowledge this limitation.

---

# 11. Validation and Test Sleeve Construction

## 11.1 Implementable Evaluation Portfolio

Do not report results by averaging performance across overlapping evaluation groups as if they formed an investable strategy.

For validation and testing, form deterministic **non-overlapping** same-industry sleeves, then allocate equal capital across sleeves.

## 11.2 Deterministic Non-Overlapping Sleeve Rule

At every validation or test rebalance date:

1. Build the dynamic top-500 universe.
2. Group securities by industry.
3. Sort securities within each industry by descending market capitalization.
4. Partition each industry list into sequential non-overlapping blocks of 10.
5. Create one sleeve from each complete block.
6. Drop residual stocks in an industry if fewer than 10 remain.

Example:

| Eligible Firms in Industry | Sleeves Created | Stocks Unused |
|---:|---:|---:|
| 8 | 0 | 8 |
| 14 | 1 | 4 |
| 27 | 2 | 7 |
| 35 | 3 | 5 |

Implement:

```python
def construct_evaluation_sleeves(
    universe_at_date: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    group_size: int = 10,
) -> pd.DataFrame:
    """
    Create deterministic non-overlapping same-industry sleeves for
    validation and testing. Within industry, sort by market cap descending
    and allocate sequential blocks of exactly 10 stocks.
    """
```

## 11.3 Equal Capital Across Sleeves

If \(G_t\) valid sleeves exist at \(t\), allocate:

\[
\omega_{g,t}
=
\frac{1}{G_t}
\]

to each sleeve.

If sleeve \(g\) assigns within-sleeve weight \(w_{g,i,t}\) to stock \(i\), the aggregate stock weight is:

\[
W_{i,t}
=
\frac{1}{G_t}
w_{g,i,t}
\]

The aggregate portfolio must satisfy:

\[
\sum_i W_{i,t}=1
\]

---

# 12. Historical Covariance Input and Future Covariance Target

For each valid group or sleeve \(g\) at rebalance date \(t\):

## 12.1 Historical Conditioning Covariance

Obtain the previous 126 trading-day return matrix:

\[
R_{g,t}^{\text{past}}
=
\left[
r_{g,t-125},\ldots,r_{g,t}
\right]
\]

Compute the sample covariance:

\[
S_{g,t}^{126}
=
\operatorname{Cov}
\left(
R_{g,t}^{\text{past}}
\right)
\]

This is observable at date \(t\) and is the conditioning input.

## 12.2 Future Realized Covariance Proxy

Obtain the subsequent 21 trading-day return matrix:

\[
R_{g,t}^{\text{future}}
=
\left[
r_{g,t+1},\ldots,r_{g,t+21}
\right]
\]

Compute:

\[
S_{g,t+1:t+21}^{21}
=
\operatorname{Cov}
\left(
R_{g,t}^{\text{future}}
\right)
\]

This is the ex-post future covariance proxy used as the diffusion training target.

## 12.3 Required Language in the Final Report

State:

> The 21-day future covariance matrix is an ex-post realized covariance proxy for the holding period, not an observable latent true covariance matrix. It is noisy because it is estimated from a limited number of daily returns.

---

# 13. Positive-Definite Covariance Representation

## 13.1 Requirement

Covariance matrices generated by the diffusion model must be symmetric positive definite before GMV optimization.

Do not train the model on unrestricted raw covariance matrix entries.

## 13.2 Ridge Stabilization

For every covariance matrix \(S\), compute:

\[
S_{\varepsilon}
=
S+\varepsilon I
\]

Primary ridge value:

\[
\varepsilon=10^{-8}
\]

Keep this configurable.

If numerical failure occurs because daily covariance values are too small, increase the ridge minimally and record the final value in `reports/implementation_notes.md`.

## 13.3 Matrix-Log Vector Representation

Compute:

\[
A=\logm(S_{\varepsilon})
\]

Since \(A\) is symmetric, vectorize the lower triangular entries:

\[
x=\operatorname{vech}(A)\in\mathbb{R}^{55}
\]

For each observation:

\[
c_{g,t}
=
\operatorname{vech}
\left[
\logm
\left(
S_{g,t}^{126}+\varepsilon I
\right)
\right]
\]

\[
y_{g,t}
=
\operatorname{vech}
\left[
\logm
\left(
S_{g,t+1:t+21}^{21}+\varepsilon I
\right)
\right]
\]

where:

| Vector | Use |
|---|---|
| \(c_{g,t}\) | Historical covariance conditioning vector |
| \(y_{g,t}\) | Future covariance target vector |

## 13.4 Training-Only Standardization

Fit separate scalers using training vectors only:

\[
\widetilde{c}
=
\frac{c-\mu_c^{\text{train}}}{\sigma_c^{\text{train}}}
\]

\[
\widetilde{y}
=
\frac{y-\mu_y^{\text{train}}}{\sigma_y^{\text{train}}}
\]

Save:

```text
artifacts/scalers/conditioning_scaler.pkl
artifacts/scalers/target_scaler.pkl
```

Apply the fitted training scalers unchanged to validation and test vectors.

## 13.5 Generated Vector to Covariance Reconstruction

For a generated standardized target vector \(\widehat{\widetilde{y}}\):

1. Inverse standardize:

\[
\widehat{y}
=
\widehat{\widetilde{y}}
\odot
\sigma_y^{\text{train}}
+
\mu_y^{\text{train}}
\]

2. Restore the symmetric log-covariance matrix:

\[
\widehat{A}
=
\operatorname{vech}^{-1}(\widehat{y})
\]

3. Reconstruct covariance:

\[
\widehat{\Sigma}
=
\expm(\widehat{A})
\]

4. Symmetrize numerically:

\[
\widehat{\Sigma}
\leftarrow
\frac{\widehat{\Sigma}+\widehat{\Sigma}^{\top}}{2}
\]

5. Confirm all eigenvalues are positive.

Implement:

```python
def covariance_to_log_vech(
    covariance: np.ndarray,
    ridge_epsilon: float,
) -> np.ndarray:
    """Convert covariance matrix to 55-dimensional log-matrix vech vector."""

def log_vech_to_covariance(
    vector: np.ndarray,
) -> np.ndarray:
    """Convert a 55-dimensional log-matrix vech vector to an SPD covariance matrix."""

def fit_training_scalers(
    train_condition_vectors: np.ndarray,
    train_target_vectors: np.ndarray,
):
    """Fit scalers only on training data and persist them."""
```

---

# 14. Conditional Diffusion Model

## 14.1 Learned Distribution

The conditional diffusion model estimates:

\[
p_{\theta}
\left(
\widetilde{y}_{g,t}
\mid
\widetilde{c}_{g,t}
\right)
\]

equivalently:

\[
p_{\theta}
\left(
\Sigma_{g,t+1:t+21}
\mid
S_{g,t}^{126}
\right)
\]

## 14.2 Forward Diffusion During Training

Noise is added to the **future target vector**, not to the conditioning covariance vector.

Let:

\[
y_0=\widetilde{y}_{g,t}
\]

For a randomly sampled diffusion step:

\[
s\sim\{1,\ldots,T\}
\]

construct:

\[
y_s
=
\sqrt{\bar{\alpha}_s}y_0
+
\sqrt{1-\bar{\alpha}_s}\epsilon
\]

where:

\[
\epsilon\sim\mathcal{N}(0,I)
\]

\[
\alpha_s=1-\beta_s
\]

\[
\bar{\alpha}_s=\prod_{j=1}^{s}\alpha_j
\]

The historical covariance conditioning vector:

\[
\widetilde{c}_{g,t}
\]

is not diffused.

## 14.3 Reverse Denoising Network

The model receives:

\[
\left[
y_s,\;e(s),\;\widetilde{c}_{g,t}
\right]
\]

where \(e(s)\) is a sinusoidal time-step embedding.

It predicts the added noise:

\[
\widehat{\epsilon}_{\theta}
=
\epsilon_{\theta}
\left(
y_s,s,\widetilde{c}_{g,t}
\right)
\]

## 14.4 Training Loss

Use DDPM noise prediction loss:

\[
\mathcal{L}_{DDPM}
=
\mathbb{E}
\left[
\left\|
\epsilon-
\epsilon_{\theta}
\left(
y_s,s,\widetilde{c}_{g,t}
\right)
\right\|_2^2
\right]
\]

The model is trained only on observations whose future target windows lie entirely within 2000–2013.

---

# 15. Fixed Neural Architecture

Use a conditional multilayer perceptron denoiser because the covariance-vector representation is only 55-dimensional.

| Component | Fixed Value |
|---|---:|
| Noised target vector dimension | 55 |
| Conditioning vector dimension | 55 |
| Time embedding dimension | 32 |
| Hidden layers | 3 |
| Hidden dimension | 128 |
| Activation | SiLU |
| Output dimension | 55 |
| Dropout | 0.0 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Batch size | 128 |
| Maximum epochs | 200 |
| Primary seed | 42 |

Do not tune architecture in the primary experiment. The tuned hyperparameters are exclusively:

\[
\boxed{
\text{beta schedule type},\;\alpha,\;T,\;M
}
\]

---

# 16. Beta Schedule Hyperparameter

## 16.1 Interpretation

The beta hyperparameter is the **shape of the diffusion noise schedule**, not a free endpoint value.

Fix:

\[
\beta_{\min}=10^{-4}
\]

\[
\beta_{\max}=0.02
\]

Tune:

\[
\text{schedule type}
\in
\{
\text{linear},
\text{quadratic},
\text{logarithmic}
\}
\]

For:

\[
s=1,\ldots,T
\]

define:

\[
u_s=\frac{s-1}{T-1}
\]

## 16.2 Linear Schedule

\[
\beta_s^{\text{linear}}
=
\beta_{\min}
+
u_s(\beta_{\max}-\beta_{\min})
\]

## 16.3 Quadratic Schedule

\[
\beta_s^{\text{quadratic}}
=
\left[
\sqrt{\beta_{\min}}
+
u_s
\left(
\sqrt{\beta_{\max}}-\sqrt{\beta_{\min}}
\right)
\right]^2
\]

## 16.4 Logarithmic Schedule

\[
\beta_s^{\text{logarithmic}}
=
\exp
\left[
\log(\beta_{\min})
+
u_s
\left(
\log(\beta_{\max})-\log(\beta_{\min})
\right)
\right]
\]

## 16.5 Schedule Factory

Implement:

```python
import numpy as np

def make_beta_schedule(
    schedule_type: str,
    T: int,
    beta_min: float = 1.0e-4,
    beta_max: float = 2.0e-2,
) -> np.ndarray:
    """
    Create a beta schedule for DDPM training and generation.
    Supported schedule types: linear, quadratic, logarithmic.
    """
    if T < 2:
        raise ValueError("T must be at least 2.")

    if schedule_type == "linear":
        betas = np.linspace(beta_min, beta_max, T)

    elif schedule_type == "quadratic":
        betas = np.linspace(np.sqrt(beta_min), np.sqrt(beta_max), T) ** 2

    elif schedule_type == "logarithmic":
        betas = np.exp(np.linspace(np.log(beta_min), np.log(beta_max), T))

    else:
        raise ValueError(f"Unsupported beta schedule type: {schedule_type}")

    if betas.shape != (T,):
        raise ValueError("Incorrect beta schedule shape.")

    if np.any(betas <= 0) or np.any(betas >= 1):
        raise ValueError("Every beta value must be strictly between 0 and 1.")

    return betas
```

---

# 17. Hyperparameters and Search Space

## 17.1 Tuned Hyperparameters Only

| Hyperparameter | Meaning | Candidate Values |
|---|---|---|
| `beta_schedule_type` | Shape of diffusion noise schedule | `["linear", "quadratic", "logarithmic"]` |
| \(T\) | Number of diffusion steps | `[25, 50, 100]` |
| \(\alpha\) | Weight on historical sample covariance in final combined covariance | `[0.00, 0.25, 0.50, 0.75, 1.00]` |
| \(M\) | Number of generated future covariance scenarios averaged at inference | `[1, 5, 10, 20, 50]` |

## 17.2 Trained Models Versus Inference Configurations

Only the following affect neural model training:

- `beta_schedule_type`
- \(T\)

Thus, train:

\[
3 \times 3=9
\]

diffusion models.

The following affect inference / portfolio construction only:

- \(\alpha\)
- \(M\)

For each trained model, evaluate all candidate combinations of \(\alpha\) and \(M\) during validation.

## 17.3 Effective Number of Validation Configurations

When:

\[
\alpha=1
\]

the final covariance is:

\[
\widehat{\Sigma}^{\text{combined}}=S^{126}
\]

and the diffusion model, \(T\), schedule type, and \(M\) have no effect.

Therefore:

- Evaluate \(\alpha=1\) only once as the Sample Covariance GMV benchmark.
- Do not create duplicate \(\alpha=1\) configurations across diffusion settings.

Nontrivial diffusion-blend configurations:

\[
3 \times 3 \times 4 \times 5=180
\]

where:

\[
\alpha\in\{0.00,0.25,0.50,0.75\}
\]

Add one sample-covariance boundary configuration:

\[
180+1=181
\]

effective validation configurations.

## 17.4 Honest Warning About Search Breadth

A 181-configuration selection exercise on one validation period creates data-mining risk even though 2014–2020 is seven years long.

Therefore:

- select one primary model using the stated metric only;
- report the top 5 validation configurations;
- report whether the selected result is robust across nearby \(\alpha\), \(M\), and \(T\) choices;
- never choose a different model based on test performance.

---

# 18. Training Procedure

## 18.1 Models to Train

Train exactly nine diffusion models:

```python
beta_schedule_grid = ["linear", "quadratic", "logarithmic"]
diffusion_steps_grid = [25, 50, 100]

for schedule_type in beta_schedule_grid:
    for T in diffusion_steps_grid:
        train_one_conditional_ddpm(
            schedule_type=schedule_type,
            T=T,
            beta_min=1e-4,
            beta_max=0.02,
            hidden_dim=128,
            seed=42,
        )
```

## 18.2 Fixed Training Rule

Use the same fixed training procedure for every model:

```yaml
training:
  epochs: 200
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.00001
  optimizer: "adam"
  fixed_epoch_training: true
  random_seed: 42
```

Primary design rule:

> Train each candidate model for the same number of epochs. Do not repeatedly monitor validation GMV volatility for epoch-by-epoch early stopping.

This prevents excessive adaptation to the validation return stream.

## 18.3 Checkpoint Naming

Save models as:

```text
artifacts/models/ddpm_schedule-{schedule_type}_T-{T}_seed-42.pt
```

Save training histories as:

```text
artifacts/training_logs/ddpm_schedule-{schedule_type}_T-{T}_seed-42.csv
```

---

# 19. Generating Future Covariance Scenarios

## 19.1 Inference Condition

For each validation or test sleeve-date, compute:

\[
S_{g,t}^{126}
\]

then transform and standardize it:

\[
\widetilde{c}_{g,t}
\]

## 19.2 Conditional Diffusion Generation

For every scenario \(m=1,\ldots,M\):

1. Sample terminal Gaussian noise:

\[
y_T^{(m)}\sim\mathcal{N}(0,I_{55})
\]

2. Run the reverse diffusion chain conditional on \(\widetilde{c}_{g,t}\):

\[
y_T^{(m)}
\rightarrow
y_{T-1}^{(m)}
\rightarrow
\cdots
\rightarrow
\widehat{y}_0^{(m)}
\]

3. Inverse-transform the generated vector to produce an SPD covariance matrix:

\[
\widehat{\Sigma}_{g,t+1}^{(m)}
=
\expm
\left[
\operatorname{vech}^{-1}
\left(
\widehat{y}_0^{(m)}
\right)
\right]
\]

## 19.3 Common Random Number Requirement

To compare values of \(M\) fairly:

1. For each model and sleeve-date, generate the maximum required number of scenarios:

\[
M_{\max}=50
\]

2. Save these 50 generated covariance matrices.

3. For each candidate \(M\), use nested prefixes:

| \(M\) | Scenarios Used |
|---:|---|
| 1 | Scenario 1 |
| 5 | Scenarios 1–5 |
| 10 | Scenarios 1–10 |
| 20 | Scenarios 1–20 |
| 50 | Scenarios 1–50 |

This ensures that differences across \(M\) values reflect scenario averaging rather than unrelated random draws.

---

# 20. Diffusion Expected Covariance and Stability Blend

## 20.1 Mean Generated Future Covariance

Compute the diffusion-implied expected future covariance in covariance space:

\[
\widehat{\Sigma}_{g,t+1}^{\text{diff}}(M)
=
\frac{1}{M}
\sum_{m=1}^{M}
\widehat{\Sigma}_{g,t+1}^{(m)}
\]

Important implementation rule:

\[
\boxed{
\text{Average reconstructed covariance matrices, not transformed log-covariance vectors.}
}
\]

Do not compute the primary estimate as:

\[
\expm
\left(
\frac{1}{M}
\sum_m \widehat{A}^{(m)}
\right)
\]

because standard GMV expected variance depends linearly on covariance matrices themselves:

\[
\mathbb{E}
\left[
w^\top \Sigma w
\right]
=
w^\top
\mathbb{E}[\Sigma]
w
\]

## 20.2 Stabilized Combined Covariance Estimate

For each \(\alpha\):

\[
\boxed{
\widehat{\Sigma}_{g,t+1}^{\text{combined}}(\alpha,M)
=
\alpha S_{g,t}^{126}
+
(1-\alpha)
\widehat{\Sigma}_{g,t+1}^{\text{diff}}(M)
}
\]

Interpretation:

| \(\alpha\) | Estimator Interpretation |
|---:|---|
| 0.00 | Pure conditional diffusion expected covariance |
| 0.25 | Mostly diffusion, partially stabilized by historical sample covariance |
| 0.50 | Equal blend |
| 0.75 | Mostly historical sample covariance, partially forward-looking diffusion |
| 1.00 | Pure historical sample covariance boundary benchmark |

The final proposed method is therefore:

> A stabilized conditional diffusion covariance estimator, not a pure diffusion estimator.

Implement:

```python
def combine_covariances(
    sample_cov: np.ndarray,
    generated_covariances: list[np.ndarray],
    alpha: float,
) -> np.ndarray:
    """
    Combine observable historical sample covariance and the arithmetic
    mean of generated next-month covariance scenarios.

    Parameters
    ----------
    sample_cov:
        Historical 126-day sample covariance.
    generated_covariances:
        List of generated next-month covariance matrices.
    alpha:
        Historical covariance stability weight, in [0, 1].

    Returns
    -------
    combined_cov:
        Symmetric combined covariance matrix for GMV optimization.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")

    if alpha == 1.0:
        combined_cov = sample_cov.copy()
    else:
        diffusion_expected_cov = np.mean(
            np.stack(generated_covariances, axis=0),
            axis=0,
        )
        combined_cov = (
            alpha * sample_cov
            + (1.0 - alpha) * diffusion_expected_cov
        )

    combined_cov = 0.5 * (combined_cov + combined_cov.T)
    return combined_cov
```

---

# 21. GMV Portfolio Optimization

## 21.1 Sleeve-Level Long-Only GMV

For each sleeve \(g\), rebalance date \(t\), and candidate configuration, solve:

\[
w_{g,t}^{GMV}
=
\arg\min_w
\quad
w^\top
\widehat{\Sigma}_{g,t+1}^{\text{combined}}
w
\]

subject to:

\[
\mathbf{1}^{\top}w=1
\]

\[
w_i\geq0
\]

Use `cvxpy`.

Implement:

```python
import cvxpy as cp
import numpy as np

def solve_long_only_gmv(covariance: np.ndarray) -> np.ndarray:
    """
    Compute long-only, fully invested GMV weights.
    """
    covariance = 0.5 * (covariance + covariance.T)

    eig_min = np.linalg.eigvalsh(covariance).min()
    if eig_min <= 1e-10:
        covariance = covariance + (abs(eig_min) + 1e-8) * np.eye(covariance.shape[0])

    n_assets = covariance.shape[0]
    w = cp.Variable(n_assets)

    objective = cp.Minimize(cp.quad_form(w, covariance))
    constraints = [cp.sum(w) == 1, w >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if w.value is None:
        raise RuntimeError("GMV optimization failed.")

    weights = np.asarray(w.value).reshape(-1)
    weights[np.abs(weights) < 1e-10] = 0.0
    weights = weights / weights.sum()

    return weights
```

## 21.2 Numerical Repair Logging

For every covariance matrix requiring jitter or repair before optimization, store:

- split;
- rebalance date;
- sleeve id;
- model configuration;
- minimum eigenvalue before repair;
- jitter added.

Save to:

```text
results/diagnostics/covariance_repairs.csv
```

If repairs are frequent, report them as a serious model stability issue.

---

# 22. Aggregate Portfolio Construction

## 22.1 Equal Weight Across Non-Overlapping Sleeves

If \(G_t\) evaluation sleeves exist at date \(t\), each receives:

\[
\omega_{g,t}=\frac{1}{G_t}
\]

Aggregate security weights:

\[
W_{i,t}
=
\omega_{g,t}w_{g,i,t}
\]

for the unique sleeve containing security \(i\).

## 22.2 Daily Portfolio Return During Holding Period

For each future trading day \(d=t+1,\ldots,t+21\):

\[
r_{p,d}
=
\sum_i W_{i,t}r_{i,d}
\]

Save portfolio weights and realized daily returns for every model and benchmark.

---

# 23. Turnover and Transaction Costs

## 23.1 Turnover with Changing Membership

Stock membership may change between rebalance dates. Define turnover using the union of holdings at consecutive rebalance dates.

Let \(W_{i,t-1}^{\text{post}}\) denote drifted pre-trade weights at the next rebalance date after applying holding-period returns.

\[
\text{Turnover}_t
=
\sum_{i\in\mathcal{H}_{t-1}\cup\mathcal{H}_t}
\left|
W_{i,t}-W_{i,t-1}^{\text{post}}
\right|
\]

Implementation must correctly set weights to zero for stocks entering or leaving the portfolio.

## 23.2 Transaction Cost Assumption

Primary reported transaction cost:

\[
10 \text{ bps per one-way turnover}
\]

\[
\text{Cost}_t
=
0.001 \times \text{Turnover}_t
\]

Subtract cost from the first daily return after each rebalance:

\[
r_{p,t+1}^{\text{net}}
=
r_{p,t+1}^{\text{gross}}
-
\text{Cost}_t
\]

Report transaction-cost sensitivity at:

\[
0,\;5,\;10,\;20
\text{ bps}
\]

Transaction costs are **not** used for hyperparameter selection. Validation selection is based on gross annualized realized portfolio volatility only.

---

# 24. Validation Procedure

## 24.1 Validation Period

Use holding periods entirely within:

\[
2014\text{-01-01 to }2020\text{-12-31}
\]

## 24.2 Validation Loop

For each trained model:

\[
(\text{beta schedule type},T)
\]

perform the following:

1. Load the trained conditional diffusion model.
2. For every validation rebalance date, construct deterministic non-overlapping same-industry sleeves.
3. For every sleeve, compute the historical sample covariance \(S_{g,t}^{126}\).
4. Generate 50 conditional next-month covariance scenarios.
5. For each:

\[
M\in\{1,5,10,20,50\}
\]

use the first \(M\) generated scenarios to estimate:

\[
\widehat{\Sigma}_{g,t+1}^{\text{diff}}(M)
\]

6. For each:

\[
\alpha\in\{0.00,0.25,0.50,0.75\}
\]

construct:

\[
\widehat{\Sigma}_{g,t+1}^{\text{combined}}(\alpha,M)
\]

7. Construct long-only GMV weights.
8. Aggregate sleeve portfolios equally.
9. Apply the aggregate weights to subsequent actual 21-day returns.
10. Concatenate the complete validation daily return stream.
11. Compute annualized realized portfolio volatility.

Separately, evaluate sample covariance GMV once as the \(\alpha=1\) boundary case.

## 24.3 Sole Selection Metric

The only metric used to select the primary model is:

\[
\boxed{
\sigma_{\text{val,ann}}
=
\sqrt{252}
\cdot
\operatorname{Std}
\left(
r_{p,d}^{2014:2020}
\right)
}
\]

Do not use any of the following for model selection:

- annualized return;
- Sharpe ratio;
- CVaR;
- maximum drawdown;
- turnover;
- transaction-cost-adjusted performance;
- covariance forecast loss;
- scenario dispersion;
- test-period performance.

## 24.4 Validation Tie Rule

If two configurations have essentially equal validation volatility:

```python
if abs(vol_a - vol_b) < 1e-8:
    choose the simpler configuration in this order:
        1. larger alpha
        2. smaller M
        3. smaller T
        4. beta schedule priority: linear, quadratic, logarithmic
```

## 24.5 Selected Primary Configuration

Save exactly one primary model configuration:

```yaml
selected_model:
  beta_schedule_type: "<selected>"
  diffusion_steps_T: "<selected>"
  alpha: "<selected>"
  scenario_count_M: "<selected>"
  validation_annualized_realized_volatility: "<selected>"
```

Also save the top 5 validation configurations for transparency, but do not select among them using test results.

---

# 25. Final Test Backtest

## 25.1 Test Period

Use holding periods entirely within:

\[
2021\text{-01-01 to }2025\text{-12-31}
\]

## 25.2 Primary Test Procedure

Load the one validation-selected configuration.

At every test rebalance date:

1. Construct dynamic market-cap top-500 universe.
2. Create deterministic non-overlapping same-industry 10-stock sleeves.
3. Compute each sleeve's past 126-day sample covariance.
4. Transform and standardize the conditioning vector using training-fitted scalers.
5. Generate \(M^\ast\) next-month covariance scenarios using the selected diffusion model.
6. Compute the diffusion expected covariance in covariance space.
7. Blend with historical sample covariance using \(\alpha^\ast\).
8. Solve long-only GMV within each sleeve.
9. Allocate equal capital across sleeves.
10. Hold for the subsequent 21 days.
11. Compute gross realized returns.
12. Compute turnover and net realized returns.
13. Store weights, returns, covariance diagnostics, and scenario dispersion.

Do not change:

- schedule type;
- \(T\);
- \(\alpha\);
- \(M\);
- portfolio constraints;
- benchmark definitions;
- cost assumptions;

after viewing test results.

---

# 26. Required Benchmarks

All benchmarks must use identical:

- rebalance dates;
- dynamic top-500 universes;
- deterministic non-overlapping evaluation sleeves;
- asset ordering;
- long-only fully-invested constraints;
- equal sleeve capital allocation;
- turnover computation;
- transaction-cost reporting.

Only the covariance estimator changes.

| Method | Covariance Used in GMV |
|---|---|
| Equal Weight | No covariance estimation; equal weight inside sleeves |
| Sample Covariance GMV | \(S_{g,t}^{126}\) |
| Ledoit-Wolf Linear Shrinkage GMV | Shrinkage covariance fitted from previous 126 daily returns |
| Ledoit-Wolf Nonlinear Shrinkage GMV | Include only if reliably implemented |
| Stabilized Conditional Diffusion GMV | \(\alpha^\ast S_{g,t}^{126}+(1-\alpha^\ast)\widehat{\Sigma}_{g,t+1}^{\text{diff}}(M^\ast)\) |

Implement Ledoit-Wolf linear shrinkage using a reliable existing library where possible, e.g. `sklearn.covariance.LedoitWolf`.

---

# 27. Required Ablations

Run these test-period ablations **after** primary model selection. They are explanatory analyses, not alternative model-selection opportunities.

| Ablation | Setting | Question Answered |
|---|---|---|
| Pure Diffusion | Selected schedule and \(T^\ast\), \(\alpha=0\), \(M=M^\ast\) | Did stability blending help? |
| Single Diffusion Scenario | Selected schedule and \(T^\ast\), \(\alpha=\alpha^\ast\), \(M=1\) | Did averaging scenarios help? |
| Pure Sample Covariance | \(\alpha=1\) | Did the proposed method improve over no diffusion contribution? |

Report the results, but never relabel an ablation as the primary model based on superior test performance.

---

# 28. Evaluation Metrics

## 28.1 Validation Metric Used for Selection

Only:

\[
\text{Annualized Realized Volatility}
=
\sqrt{252}\operatorname{Std}(r_{p,d})
\]

is used to select the primary configuration.

## 28.2 Final Test Reporting Metrics

Report the following for the selected proposed model and all benchmarks:

| Category | Metric |
|---|---|
| Primary risk | Annualized realized volatility |
| Relative risk | Volatility reduction versus Sample Covariance GMV |
| Return | Annualized compounded return |
| Risk-adjusted return | Sharpe ratio with zero risk-free assumption |
| Tail risk | Historical daily CVaR at 95% |
| Drawdown | Maximum drawdown |
| Trading | Average turnover per rebalance |
| Net result | Annualized return after 10 bps turnover cost |
| Cost robustness | Net return at 0, 5, 10, and 20 bps |
| Concentration | Average maximum security weight |
| Concentration | Average aggregate weight HHI |

## 28.3 Metric Formulas

### Annualized Volatility

\[
\sigma_{\text{ann}}
=
\sqrt{252}
\operatorname{Std}(r_{p,d})
\]

### Volatility Reduction versus Sample GMV

\[
\text{Volatility Reduction}
=
1-
\frac{
\sigma_{\text{proposed}}
}{
\sigma_{\text{sample-GMV}}
}
\]

### Annualized Compounded Return

\[
R_{\text{ann}}
=
\left(
\prod_{d=1}^{D}(1+r_{p,d})
\right)^{252/D}
-1
\]

### Sharpe Ratio

Assuming zero risk-free rate:

\[
\text{Sharpe}
=
\frac{
\sqrt{252}\overline{r_p}
}{
\operatorname{Std}(r_p)
}
\]

### Historical Daily CVaR at 95%

Let:

\[
L_d=-r_{p,d}
\]

Then:

\[
\text{CVaR}_{95\%}
=
\mathbb{E}
\left[
L_d\mid L_d\geq\text{VaR}_{95\%}
\right]
\]

### Maximum Drawdown

Let cumulative wealth be:

\[
V_d=\prod_{\tau\leq d}(1+r_{p,\tau})
\]

Then:

\[
MDD
=
\min_d
\left(
\frac{V_d}{\max_{\tau\leq d}V_\tau}-1
\right)
\]

### Aggregate Weight HHI

\[
HHI_t
=
\sum_i W_{i,t}^2
\]

Report average HHI across rebalances.

---

# 29. Diffusion Diagnostics

These are diagnostics only. They must not be used for primary model selection.

## 29.1 Scenario Dispersion

For each sleeve-date:

\[
D_{g,t}
=
\frac{1}{M}
\sum_{m=1}^{M}
\left\|
\widehat{\Sigma}_{g,t+1}^{(m)}
-
\widehat{\Sigma}_{g,t+1}^{\text{diff}}(M)
\right\|_F^2
\]

Report:

- mean scenario dispersion;
- median scenario dispersion;
- scenario dispersion time series;
- scenario dispersion versus subsequent realized portfolio volatility.

## 29.2 Forecast Loss Diagnostics

Using the realized future covariance proxy:

### Frobenius Loss

\[
L_{\text{Frob}}
=
\left\|
\widehat{\Sigma}_{g,t+1}^{\text{combined}}
-
S_{g,t+1:t+21}^{21}
\right\|_F
\]

### Log-Covariance Loss

\[
L_{\log}
=
\left\|
\logm
\left(
\widehat{\Sigma}_{g,t+1}^{\text{combined}}
\right)
-
\logm
\left(
S_{g,t+1:t+21}^{21}+\varepsilon I
\right)
\right\|_F
\]

## 29.3 Numerical Stability Diagnostics

Report:

- generated covariance minimum eigenvalues;
- combined covariance minimum eigenvalues;
- condition numbers;
- number of diagonal-jitter repairs;
- magnitude of repairs.

---

# 30. Repository Structure

Create the following repository:

```text
final_sprint_cov_diffusion/
│
├── README.md
├── research_spec.md
├── requirements.txt
│
├── config/
│   ├── base_config.yaml
│   └── column_mapping.yaml
│
├── data/
│   ├── raw/
│   │   └── crsp_daily.parquet
│   ├── interim/
│   │   ├── cleaned_crsp_daily.parquet
│   │   ├── trading_calendar.parquet
│   │   ├── rebalance_dates.parquet
│   │   ├── dynamic_top500_universe.parquet
│   │   ├── training_groups.parquet
│   │   └── evaluation_sleeves.parquet
│   └── processed/
│       ├── covariance_pairs_train.npz
│       ├── covariance_pairs_validation.npz
│       └── covariance_pairs_test.npz
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_cleaning.py
│   ├── calendar.py
│   ├── universe.py
│   ├── groups.py
│   ├── covariance.py
│   ├── transforms.py
│   ├── datasets.py
│   ├── beta_schedules.py
│   ├── model.py
│   ├── diffusion.py
│   ├── train.py
│   ├── generate.py
│   ├── gmv.py
│   ├── benchmarks.py
│   ├── backtest.py
│   ├── turnover.py
│   ├── metrics.py
│   ├── diagnostics.py
│   ├── plotting.py
│   └── utils.py
│
├── scripts/
│   ├── 01_clean_crsp_data.py
│   ├── 02_build_universe_and_groups.py
│   ├── 03_build_covariance_datasets.py
│   ├── 04_train_diffusion_models.py
│   ├── 05_validate_hyperparameters.py
│   ├── 06_run_final_test.py
│   └── 07_generate_report.py
│
├── artifacts/
│   ├── scalers/
│   ├── models/
│   ├── training_logs/
│   └── selected_model/
│
├── results/
│   ├── validation/
│   ├── test/
│   ├── benchmarks/
│   ├── ablations/
│   ├── diagnostics/
│   ├── portfolios/
│   └── figures/
│
├── reports/
│   ├── implementation_notes.md
│   ├── final_results.md
│   └── final_results_tables.xlsx
│
└── tests/
    ├── test_data_cleaning.py
    ├── test_split_integrity.py
    ├── test_universe_no_lookahead.py
    ├── test_groups.py
    ├── test_covariance_transform.py
    ├── test_beta_schedules.py
    ├── test_diffusion_shapes.py
    ├── test_covariance_combination.py
    ├── test_gmv_weights.py
    ├── test_turnover.py
    └── test_reproducibility.py
```

---

# 31. Configuration File

Create `config/base_config.yaml`:

```yaml
project:
  name: "stabilized_conditional_diffusion_covariance_gmv"
  random_seed: 42
  output_dir: "results"

data:
  raw_file: "data/raw/crsp_daily.parquet"
  cleaned_file: "data/interim/cleaned_crsp_daily.parquet"
  use_common_shares_only: true
  eligible_share_codes: [10, 11]
  restrict_major_us_exchanges: true
  eligible_exchange_codes: [1, 2, 3]
  include_delisting_returns: true
  market_cap_top_n: 500

periods:
  train:
    start: "2000-01-01"
    end: "2013-12-31"
  validation:
    start: "2014-01-01"
    end: "2020-12-31"
  test:
    start: "2021-01-01"
    end: "2025-12-31"

rolling_windows:
  lookback_days: 126
  horizon_days: 21
  rebalance_every_trading_days: 21
  non_overlapping_holding_periods: true
  assign_split_by_complete_target_window: true

industry:
  method: "sic_2digit"
  group_size: 10

groups:
  training:
    allow_overlap: true
    target_groups_per_rebalance_date: 50
    deterministic_seed: 42
  evaluation:
    allow_overlap: false
    construction: "industry_then_market_cap_descending_sequential_blocks"
    capital_allocation: "equal_weight_across_sleeves"

covariance_transform:
  method: "ridge_matrix_log_vech"
  vector_dimension: 55
  ridge_epsilon: 1.0e-8
  standardize_condition_vectors: true
  standardize_target_vectors: true
  fit_scalers_on_training_only: true
  average_generated_outputs_in_covariance_space: true

model:
  type: "conditional_ddpm"
  noised_target_dim: 55
  conditioning_dim: 55
  time_embedding_dim: 32
  hidden_dim: 128
  num_hidden_layers: 3
  activation: "silu"
  dropout: 0.0

training:
  epochs: 200
  batch_size: 128
  optimizer: "adam"
  learning_rate: 0.001
  weight_decay: 0.00001
  fixed_epoch_training: true
  beta_min: 0.0001
  beta_max: 0.02
  beta_schedule_grid:
    - "linear"
    - "quadratic"
    - "logarithmic"
  diffusion_steps_grid:
    - 25
    - 50
    - 100
  random_seed: 42

generation:
  scenario_count_grid:
    - 1
    - 5
    - 10
    - 20
    - 50
  maximum_scenarios_generated: 50
  use_common_random_numbers: true
  use_nested_scenario_prefixes: true

covariance_combination:
  enabled: true
  formula: "alpha * sample_covariance + (1 - alpha) * diffusion_expected_covariance"
  alpha_grid:
    - 0.00
    - 0.25
    - 0.50
    - 0.75
    - 1.00
  alpha_one_is_single_sample_covariance_boundary: true

portfolio:
  objective: "long_only_gmv"
  long_only: true
  fully_invested: true
  sleeve_capital_allocation: "equal"
  optimization_solver: "clarabel"
  transaction_cost_bps_main: 10
  transaction_cost_bps_sensitivity: [0, 5, 10, 20]

validation:
  selection_metric: "gross_annualized_realized_gmv_portfolio_volatility"
  use_only_selection_metric_for_model_choice: true
  select_single_primary_model: true
  report_top_k_configurations: 5
  tie_tolerance: 1.0e-8
  tie_breaking_order:
    - "larger_alpha"
    - "smaller_M"
    - "smaller_T"
    - "linear_then_quadratic_then_logarithmic"

benchmarks:
  equal_weight: true
  sample_covariance_gmv: true
  ledoit_wolf_linear_shrinkage_gmv: true
  ledoit_wolf_nonlinear_shrinkage_gmv: false

ablations:
  pure_diffusion_alpha_zero: true
  single_scenario_M_one: true
  pure_sample_covariance_alpha_one: true

reporting:
  report_gross_results: true
  report_net_results: true
  report_transaction_cost_sensitivity: true
  report_scenario_dispersion: true
  report_covariance_forecast_diagnostics: true
  report_weight_concentration: true
  report_numerical_repairs: true
```

---

# 32. Script Responsibilities

## `scripts/01_clean_crsp_data.py`

Responsibilities:

1. Read raw CRSP daily file.
2. Inspect and map columns.
3. Filter ordinary common shares and eligible exchanges.
4. Construct delisting-adjusted daily return.
5. Construct daily market capitalization.
6. Remove or flag malformed records.
7. Ensure uniqueness by `date` and `permno`.
8. Save cleaned panel.

Output:

```text
data/interim/cleaned_crsp_daily.parquet
```

## `scripts/02_build_universe_and_groups.py`

Responsibilities:

1. Build canonical CRSP trading calendar.
2. Generate non-overlapping rebalance dates for each split.
3. Construct dynamic top-500 universe at every rebalance date.
4. Construct overlapping sampled training groups.
5. Construct deterministic non-overlapping validation sleeves.
6. Construct deterministic non-overlapping test sleeves.
7. Save all universe and group membership tables.

Outputs:

```text
data/interim/trading_calendar.parquet
data/interim/rebalance_dates.parquet
data/interim/dynamic_top500_universe.parquet
data/interim/training_groups.parquet
data/interim/evaluation_sleeves.parquet
```

## `scripts/03_build_covariance_datasets.py`

Responsibilities:

1. Build historical 126-day covariance inputs.
2. Build future 21-day covariance targets.
3. Assign observations to splits by complete future holding window.
4. Apply matrix-log vectorization.
5. Fit scalers on training data only.
6. Apply training-fitted scalers to validation and test.
7. Save processed arrays and metadata.

Outputs:

```text
data/processed/covariance_pairs_train.npz
data/processed/covariance_pairs_validation.npz
data/processed/covariance_pairs_test.npz
artifacts/scalers/conditioning_scaler.pkl
artifacts/scalers/target_scaler.pkl
```

## `scripts/04_train_diffusion_models.py`

Responsibilities:

1. Train exactly nine models:
   - schedule type in `["linear", "quadratic", "logarithmic"]`
   - \(T\) in `[25, 50, 100]`
2. Use training data only.
3. Use fixed architecture and fixed epochs.
4. Save checkpoints and training histories.

## `scripts/05_validate_hyperparameters.py`

Responsibilities:

1. Load each of the nine trained models.
2. Generate 50 scenarios per validation sleeve-date using common random-number logic.
3. Evaluate each candidate \(M\) and each candidate \(\alpha<1\).
4. Evaluate sample covariance GMV once for \(\alpha=1\).
5. Construct combined portfolios and realized daily returns.
6. Compute annualized realized volatility.
7. Rank configurations using that metric only.
8. Save one primary selected configuration.
9. Save top-five validation table.

Outputs:

```text
results/validation/validation_grid_results.csv
results/validation/top5_validation_configurations.csv
artifacts/selected_model/selected_model_config.yaml
artifacts/selected_model/selected_model.pt
```

## `scripts/06_run_final_test.py`

Responsibilities:

1. Load the primary validation-selected model.
2. Run the untouched 2021–2025 proposed-method backtest.
3. Run all required benchmarks.
4. Run required explanatory ablations.
5. Compute gross metrics, turnover, and cost-adjusted metrics.
6. Save portfolio returns, weights, diagnostics, and summary metrics.

## `scripts/07_generate_report.py`

Responsibilities:

1. Generate result tables.
2. Generate figures.
3. Generate a final Markdown report.
4. Include limitations and implementation deviations.

---

# 33. Validation Pseudocode

```python
beta_schedule_grid = ["linear", "quadratic", "logarithmic"]
T_grid = [25, 50, 100]
alpha_grid = [0.00, 0.25, 0.50, 0.75]
M_grid = [1, 5, 10, 20, 50]
M_max = 50

validation_results = []

# Evaluate diffusion-blend configurations.
for schedule_type in beta_schedule_grid:
    for T in T_grid:
        model = load_trained_model(schedule_type=schedule_type, T=T)

        # Cache maximum scenario generation once per model and sleeve-date.
        scenario_cache = {}

        for date in validation_rebalance_dates:
            sleeves = load_non_overlapping_evaluation_sleeves(date)

            for sleeve in sleeves:
                sample_cov = compute_historical_covariance(
                    sleeve=sleeve,
                    rebalance_date=date,
                    lookback_days=126,
                )

                condition_vector = transform_condition_with_training_scaler(sample_cov)

                scenarios_50 = generate_covariance_scenarios(
                    model=model,
                    condition_vector=condition_vector,
                    num_scenarios=M_max,
                    seed=deterministic_scenario_seed(
                        schedule_type=schedule_type,
                        T=T,
                        rebalance_date=date,
                        sleeve_id=sleeve.id,
                    ),
                )

                scenario_cache[(date, sleeve.id)] = {
                    "sample_cov": sample_cov,
                    "scenarios_50": scenarios_50,
                }

        for M in M_grid:
            for alpha in alpha_grid:
                daily_returns = []

                for date in validation_rebalance_dates:
                    sleeves = load_non_overlapping_evaluation_sleeves(date)
                    aggregate_weights = {}

                    for sleeve in sleeves:
                        cached = scenario_cache[(date, sleeve.id)]
                        sample_cov = cached["sample_cov"]
                        scenarios_M = cached["scenarios_50"][:M]

                        combined_cov = combine_covariances(
                            sample_cov=sample_cov,
                            generated_covariances=scenarios_M,
                            alpha=alpha,
                        )

                        sleeve_weights = solve_long_only_gmv(combined_cov)

                        add_scaled_sleeve_weights_to_aggregate(
                            aggregate_weights=aggregate_weights,
                            sleeve=sleeve,
                            sleeve_weights=sleeve_weights,
                            sleeve_capital=1.0 / len(sleeves),
                        )

                    holding_returns = calculate_future_portfolio_returns(
                        aggregate_weights=aggregate_weights,
                        rebalance_date=date,
                        horizon_days=21,
                    )

                    daily_returns.extend(holding_returns)

                annualized_vol = annualized_realized_volatility(daily_returns)

                validation_results.append({
                    "beta_schedule_type": schedule_type,
                    "diffusion_steps_T": T,
                    "alpha": alpha,
                    "scenario_count_M": M,
                    "validation_annualized_realized_volatility": annualized_vol,
                    "is_sample_covariance_boundary": False,
                })

# Evaluate alpha=1 exactly once.
sample_gmv_validation_returns = run_sample_covariance_gmv_validation_backtest()
sample_gmv_validation_vol = annualized_realized_volatility(sample_gmv_validation_returns)

validation_results.append({
    "beta_schedule_type": "not_applicable",
    "diffusion_steps_T": None,
    "alpha": 1.0,
    "scenario_count_M": None,
    "validation_annualized_realized_volatility": sample_gmv_validation_vol,
    "is_sample_covariance_boundary": True,
})

rank_and_select_single_primary_configuration(validation_results)
```

---

# 34. Test Backtest Pseudocode

```python
selected = load_selected_model_config()
selected_model = load_selected_model()

test_returns_gross = []
test_weights = []
test_turnover = []
test_diagnostics = []

previous_post_return_weights = None

for date in test_rebalance_dates:
    sleeves = load_non_overlapping_evaluation_sleeves(date)
    aggregate_weights = {}

    for sleeve in sleeves:
        sample_cov = compute_historical_covariance(
            sleeve=sleeve,
            rebalance_date=date,
            lookback_days=126,
        )

        condition_vector = transform_condition_with_training_scaler(sample_cov)

        scenarios = generate_covariance_scenarios(
            model=selected_model,
            condition_vector=condition_vector,
            num_scenarios=selected.scenario_count_M,
            seed=deterministic_scenario_seed(
                schedule_type=selected.beta_schedule_type,
                T=selected.diffusion_steps_T,
                rebalance_date=date,
                sleeve_id=sleeve.id,
            ),
        )

        combined_cov = combine_covariances(
            sample_cov=sample_cov,
            generated_covariances=scenarios,
            alpha=selected.alpha,
        )

        sleeve_weights = solve_long_only_gmv(combined_cov)

        add_scaled_sleeve_weights_to_aggregate(
            aggregate_weights=aggregate_weights,
            sleeve=sleeve,
            sleeve_weights=sleeve_weights,
            sleeve_capital=1.0 / len(sleeves),
        )

        save_covariance_and_scenario_diagnostics(
            date=date,
            sleeve=sleeve,
            sample_cov=sample_cov,
            generated_scenarios=scenarios,
            combined_cov=combined_cov,
        )

    if previous_post_return_weights is not None:
        turnover = compute_turnover_on_union_of_permnos(
            target_weights=aggregate_weights,
            pretrade_weights=previous_post_return_weights,
        )
    else:
        turnover = sum(abs(w) for w in aggregate_weights.values())

    gross_holding_returns, end_of_period_drifted_weights = calculate_holding_returns_and_drifted_weights(
        aggregate_weights=aggregate_weights,
        rebalance_date=date,
        horizon_days=21,
    )

    test_returns_gross.extend(gross_holding_returns)
    test_weights.append((date, aggregate_weights))
    test_turnover.append((date, turnover))

    previous_post_return_weights = end_of_period_drifted_weights

test_returns_net = apply_transaction_costs_to_rebalance_dates(
    gross_returns=test_returns_gross,
    turnovers=test_turnover,
    cost_bps=10,
)

compute_and_save_all_test_metrics()
```

---

# 35. Required Unit Tests

## 35.1 Data Cleaning

Test:

- unique `(date, permno)` after cleaning;
- positive market cap where retained;
- correct delisting-return compounding;
- common-share filtering;
- exchange filtering where applicable.

## 35.2 Split Integrity

Test:

```python
def test_all_training_target_dates_are_within_2000_2013():
    ...

def test_all_validation_target_dates_are_within_2014_2020():
    ...

def test_all_test_target_dates_are_within_2021_2025():
    ...

def test_no_target_window_crosses_split_boundary():
    ...

def test_no_observation_is_assigned_to_multiple_splits():
    ...

def test_scalers_are_fitted_on_training_only():
    ...
```

## 35.3 No-Lookahead Universe

For every rebalance date, ensure market-cap ranking and SIC classification use only information available at that date.

## 35.4 Group Construction

Test:

- every sleeve contains exactly 10 stocks;
- every sleeve contains one industry only;
- positions are ordered by descending formation-date market cap;
- validation and test sleeves never overlap at a date;
- training overlap is allowed but deterministic under the seed.

## 35.5 Transformation

For multiple sample covariance matrices:

```python
vector = covariance_to_log_vech(covariance, ridge_epsilon)
reconstructed = log_vech_to_covariance(vector)

assert np.all(np.linalg.eigvalsh(reconstructed) > 0)
assert np.allclose(covariance + ridge_epsilon * np.eye(10), reconstructed, atol=1e-8)
```

## 35.6 Beta Schedules

Test:

- all three schedule types create length \(T\) arrays;
- values lie in \((0,1)\);
- first and last values match fixed endpoints;
- unknown schedule raises `ValueError`.

## 35.7 Diffusion Shapes

Test that:

- model input and output dimensions are correct;
- generated target vectors have dimension 55;
- reconstructed scenario covariance has shape `(10, 10)`;
- generated covariance is symmetric positive definite.

## 35.8 Covariance Combination

Test:

```python
combined_alpha_zero = combine_covariances(sample_cov, scenarios, alpha=0.0)
combined_alpha_one = combine_covariances(sample_cov, scenarios, alpha=1.0)

assert np.allclose(combined_alpha_zero, np.mean(scenarios, axis=0))
assert np.allclose(combined_alpha_one, sample_cov)
```

## 35.9 GMV Optimization

Test:

```python
weights = solve_long_only_gmv(covariance)

assert abs(weights.sum() - 1.0) < 1e-6
assert weights.min() >= -1e-8
```

## 35.10 Reproducibility

Running validation twice with the same data and seeds must produce identical:

- training sampled groups;
- generated scenario sets;
- validation ranking;
- selected configuration;
- test return series;
- test metrics.

---

# 36. Required Outputs

## 36.1 Validation Grid Results

Create:

```text
results/validation/validation_grid_results.csv
```

Columns:

| Column | Description |
|---|---|
| `beta_schedule_type` | `linear`, `quadratic`, `logarithmic`, or `not_applicable` for sample boundary |
| `diffusion_steps_T` | Diffusion steps or null for sample boundary |
| `alpha` | Sample covariance blend weight |
| `scenario_count_M` | Generated scenario count or null for sample boundary |
| `validation_annualized_realized_volatility` | Sole selection metric |
| `is_sample_covariance_boundary` | Boolean |
| `rank` | Rank by validation volatility |
| `selected_primary_model` | Boolean |

## 36.2 Selected Configuration

Create:

```text
artifacts/selected_model/selected_model_config.yaml
```

Contents:

```yaml
selected_model:
  beta_schedule_type: "<value>"
  diffusion_steps_T: "<value>"
  alpha: "<value>"
  scenario_count_M: "<value>"
  validation_metric: "gross_annualized_realized_gmv_portfolio_volatility"
  validation_annualized_realized_volatility: "<value>"
  validation_period: ["2014-01-01", "2020-12-31"]
  test_period_locked: ["2021-01-01", "2025-12-31"]
```

## 36.3 Test Performance Table

Create:

```text
results/test/test_performance_summary.csv
```

Rows:

- Equal Weight
- Sample Covariance GMV
- Ledoit-Wolf Linear Shrinkage GMV
- Ledoit-Wolf Nonlinear Shrinkage GMV, if available
- Stabilized Conditional Diffusion GMV
- Pure Diffusion Ablation
- Single Scenario Ablation

Columns:

| Column |
|---|
| `annualized_volatility_gross` |
| `volatility_reduction_vs_sample_gmv` |
| `annualized_return_gross` |
| `sharpe_gross` |
| `cvar_95_daily_gross` |
| `maximum_drawdown_gross` |
| `average_turnover` |
| `annualized_return_net_10bps` |
| `annualized_volatility_net_10bps` |
| `average_max_stock_weight` |
| `average_weight_hhi` |

## 36.4 Diagnostics

Create:

```text
results/diagnostics/scenario_dispersion.csv
results/diagnostics/covariance_forecast_losses.csv
results/diagnostics/covariance_condition_numbers.csv
results/diagnostics/covariance_repairs.csv
```

---

# 37. Required Figures

Save:

```text
results/figures/validation_configuration_volatility_ranking.png
results/figures/validation_alpha_sensitivity.png
results/figures/validation_M_sensitivity.png
results/figures/test_cumulative_wealth_gross.png
results/figures/test_cumulative_wealth_net_10bps.png
results/figures/test_rolling_63day_volatility.png
results/figures/test_turnover_comparison.png
results/figures/test_weight_concentration_comparison.png
results/figures/test_scenario_dispersion_over_time.png
```

Plot rules:

- Label validation plots explicitly as validation-period results.
- Label test plots explicitly as final untouched out-of-sample results.
- Do not combine validation and test wealth curves as one continuous performance record.
- Do not highlight test-period alternative configurations as superior to the primary validation-selected model.

---

# 38. Final Report Structure

Create:

```text
reports/final_results.md
```

with the following sections:

```markdown
# Final Results: Stabilized Conditional Diffusion Forecasting for GMV Portfolio Optimization

## 1. Research Question

## 2. Motivation and Methodological Framing

## 3. CRSP Daily Data and Dynamic Top-500 Universe

## 4. Same-Industry Sleeve Construction

## 5. Covariance Input and Future Target Construction

## 6. Conditional Diffusion Model

## 7. Stabilized Covariance Estimator and the Role of Alpha

## 8. GMV Portfolio Construction

## 9. Training Design: 2000–2013

## 10. Validation and Hyperparameter Selection: 2014–2020

## 11. Selected Primary Configuration

## 12. Final Untouched Test Results: 2021–2025

## 13. Benchmark Comparison

## 14. Required Ablation Results

## 15. Transaction-Cost Sensitivity

## 16. Diffusion Scenario Diagnostics

## 17. Numerical Stability Diagnostics

## 18. Limitations

## 19. Conclusion
```

---

# 39. Mandatory Limitations in the Final Report

The report must state all of the following limitations clearly.

## 39.1 Noisy Future Covariance Proxy

The subsequent 21-day realized covariance matrix is estimated from only 21 daily observations for 10 assets and is therefore a noisy proxy for the future covariance relevant to the holding period.

## 39.2 Diffusion Is Not Directly Recovering True Covariance

The model learns the conditional distribution of realized future covariance proxies. It does not directly observe or recover a latent true covariance matrix.

## 39.3 Standard GMV Uses Only the Scenario Mean

Although diffusion generates multiple covariance scenarios, the main GMV portfolio uses their arithmetic mean before blending with historical sample covariance. Therefore, the main strategy uses the diffusion distribution through its implied expected covariance, not through an explicit tail-risk optimization criterion.

## 39.4 Alpha Makes the Method a Hybrid Estimator

Because the selected covariance estimator includes:

\[
\alpha S_t^{126}
+
(1-\alpha)\widehat{\Sigma}_{t+1}^{\text{diff}}
\]

any performance improvement should be attributed to a **stabilized conditional diffusion estimator**, not necessarily to pure diffusion alone.

## 39.5 Dependence Across Training Groups

Training observations produced from overlapping groups are cross-sectionally dependent because groups may share stocks and common shocks.

## 39.6 Hyperparameter Search Risk

Even with a seven-year validation period, selecting among 181 effective configurations creates a nontrivial risk of validation overfitting. The test period must remain fully untouched, and nearby hyperparameter robustness should be disclosed.

## 39.7 Strong Benchmark Possibility

Ledoit-Wolf shrinkage is a strong conventional covariance benchmark. If it outperforms the diffusion method in the final test period, report that result honestly.

---

# 40. Implementation Order

Execute in this order:

## Phase 1: Data Preparation

1. Load configuration and raw CRSP dataset.
2. Resolve raw-column mapping.
3. Clean returns and incorporate delisting returns.
4. Construct market capitalization.
5. Build trading calendar.
6. Generate split-safe rebalance dates.

## Phase 2: Universe and Sleeve Formation

7. Construct dynamic top-500 universe at every rebalance date.
8. Construct overlapping training groups.
9. Construct non-overlapping validation and test sleeves.
10. Save all memberships.

## Phase 3: Covariance Dataset

11. Construct historical 126-day covariance matrices.
12. Construct future 21-day covariance proxies.
13. Assign observations to splits by target window.
14. Transform covariance matrices using ridge-log-`vech`.
15. Fit and save training-only scalers.
16. Save train, validation, and test processed datasets.

## Phase 4: Conditional Diffusion Training

17. Implement beta schedules.
18. Implement conditional MLP denoiser.
19. Implement DDPM training and reverse generation.
20. Train nine candidate models.
21. Save checkpoints and loss histories.

## Phase 5: Validation and Selection

22. Generate 50 covariance scenarios per validation sleeve-date and trained model.
23. Evaluate \(\alpha\) and \(M\) combinations.
24. Evaluate sample covariance boundary once.
25. Construct validation GMV portfolios.
26. Compute annualized realized volatility.
27. Rank and select exactly one primary model.

## Phase 6: Final Test and Benchmarks

28. Run selected model on untouched test set.
29. Run required benchmarks.
30. Run explanatory ablations.
31. Compute transaction-cost sensitivity.
32. Compute portfolio and diffusion diagnostics.

## Phase 7: Reporting and Verification

33. Generate final tables and figures.
34. Write final results report.
35. Run all unit tests.
36. Record implementation deviations.
37. Verify no test data were used in selection.

---

# 41. Definition of Done

The project is complete only when:

- [ ] CRSP daily dataset is cleaned and mapped.
- [ ] Delisting-adjusted returns and daily market caps are constructed.
- [ ] Split-safe non-overlapping rebalance schedules exist.
- [ ] Dynamic market-cap top-500 universes are saved.
- [ ] Overlapping training groups are saved.
- [ ] Non-overlapping validation and test sleeves are saved.
- [ ] Covariance input-target pairs are built.
- [ ] SPD-preserving transformations and training-only scalers are implemented.
- [ ] All nine diffusion models are trained and checkpointed.
- [ ] All 181 effective validation configurations are evaluated.
- [ ] One primary configuration is selected using validation realized volatility only.
- [ ] The untouched 2021–2025 test backtest is run exactly once for the primary model.
- [ ] Required benchmark portfolios are run.
- [ ] Required ablations are reported.
- [ ] Gross and net results are saved.
- [ ] Diffusion diagnostics and numerical repair logs are saved.
- [ ] Figures and final Markdown report are generated.
- [ ] All unit tests pass.
- [ ] Implementation deviations and limitations are documented.

---

# 42. Final One-Paragraph Research Summary

This study evaluates a stabilized conditional diffusion covariance estimator for global minimum-variance portfolio construction. Using CRSP daily U.S. equity data, a dynamic market-cap top-500 universe is formed at each non-overlapping 21-trading-day rebalance date. Same-industry 10-stock sleeves are constructed, and each sleeve's covariance matrix estimated from the previous 126 trading days serves as the conditioning input. A conditional diffusion model is trained on 2000–2013 data to generate multiple plausible covariance matrices for the subsequent 21-trading-day holding period. The generated covariance scenarios are averaged in covariance-matrix space to obtain a diffusion-implied expected future covariance, which is then blended with the observed historical sample covariance through a stability weight \(\alpha\). The beta schedule family, number of diffusion steps \(T\), scenario count \(M\), and stability weight \(\alpha\) are selected solely according to annualized realized GMV portfolio volatility during 2014–2020. The single best validation-selected specification is evaluated once on an untouched 2021–2025 test period against equal-weight, raw sample-covariance GMV, and Ledoit-Wolf shrinkage benchmarks.

---

# 43. Final Non-Negotiable Rules

1. Train only on 2000–2013 future-target windows.
2. Select hyperparameters only on 2014–2020 realized GMV portfolio volatility.
3. Keep 2021–2025 fully untouched until final evaluation.
4. Add diffusion noise to the future covariance target, not to the historical conditioning matrix.
5. Generate covariance scenarios conditionally on the historical 126-day covariance.
6. Average generated outputs in covariance-matrix space.
7. Blend diffusion expected covariance with sample covariance using \(\alpha\).
8. Construct long-only GMV portfolios from the combined covariance.
9. Use non-overlapping evaluation sleeves and equal capital allocation.
10. Evaluate \(\alpha=1\) once as the sample covariance boundary benchmark.
11. Never replace the primary selected model based on test-period performance.
12. Report negative results honestly if shrinkage benchmarks outperform diffusion.
