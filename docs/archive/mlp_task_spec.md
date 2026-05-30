# Codex Task: Create `MLP.ipynb` for Direct MLP Portfolio Weight Learning

> Archived original task prompt. The maintained project guide is `README.md`; the implemented notebook is `MLP.ipynb`.

I want to create a new notebook called:
```text
MLP.ipynb

The goal is to simplify my final sprint project by replacing the diffusion covariance model with a direct MLP portfolio-weight learner.

The model should not generate covariance matrices, should not denoise covariance matrices, and should not use DDPM. It should directly output long-only portfolio weights through a softmax layer and train those weights to minimize next-month realized portfolio variance.

I already have macro features ready. Use those macro features as part of the MLP input.

⸻

1. Existing Project Context

The existing project already has some or all of the following:

* CRSP daily U.S. equity data
* Dynamic market-cap top 500 universe
* 10-stock same-industry sleeves
* Previous 126-trading-day return windows
* Next 21-trading-day holding windows
* Train / validation / test split
* Covariance benchmark backtests
* Existing conditional covariance forecasting model
* Macro features

Use the existing codebase as much as possible. Do not rewrite the whole project unless necessary.

Use the following experimental setting:

Item	Setting
Data	CRSP daily U.S. equities
Universe	Dynamic market-cap top 500
Portfolio unit	10 same-industry stocks
Lookback window	Previous 126 trading days
Holding period	Next 21 trading days
Training period	2000–2013
Validation period	2014–2020
Test period	2021–2025
Portfolio type	Long-only, fully invested, sleeve-level portfolio
Sleeve aggregation	Equal capital across active sleeves

The MLP model must use the same train / validation / test split and the same portfolio evaluation framework as the existing benchmark code.

⸻

2. New Model: Direct MLP Softmax Weight Learner

For each 10-stock sleeve at rebalance date (t):

1. Compute historical features using only information available at date (t).
2. Feed those features into an MLP.
3. The MLP outputs 10 logits.
4. Apply softmax to produce long-only fully invested portfolio weights:

[
w_t = \operatorname{softmax}(f_\theta(x_t))
]

This guarantees:

[
w_i \geq 0
]

and:

[
\sum_{i=1}^{10} w_i = 1
]

The model directly learns portfolio weights. It does not estimate a covariance matrix and does not solve a quadratic program for the MLP model.

⸻

3. Input Features

Use a combined feature vector:

[
x_t =
[
\text{Cholesky covariance features},
\text{macro features}
]
]

The Cholesky covariance features capture sleeve-level historical covariance structure. The macro features capture market-wide state information.

⸻

4. Cholesky Covariance Features

For each 10-stock sleeve at rebalance date (t), compute the previous 126-day sample covariance matrix:

[
S_t^{126}
]

Add a small diagonal jitter:

[
S_{t,\epsilon}^{126}

S_t^{126}
+
\epsilon I
]

Use:

eps = 1e-8

Then compute the Cholesky decomposition:

[
S_{t,\epsilon}^{126}

L_t L_t^\top
]

Use the lower-triangular entries of (L_t) as covariance features:

[
x_t^{cov}

\operatorname{vech}(L_t)
]

For a 10-asset sleeve:

[
\dim(x_t^{cov})

\frac{10(10+1)}{2}

55
]

Implement robustly:

import numpy as np
def covariance_to_cholesky_features(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert a 10x10 covariance matrix into 55 Cholesky lower-triangle features.
    Parameters
    ----------
    S:
        10x10 sample covariance matrix.
    eps:
        Initial diagonal jitter.
    Returns
    -------
    features:
        55-dimensional vector containing the lower-triangular entries
        of the Cholesky factor.
    """
    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)
    n = S.shape[0]
    if S.shape != (10, 10):
        raise ValueError(f"Expected 10x10 covariance matrix, got {S.shape}")
    jitter = eps
    for _ in range(8):
        try:
            L = np.linalg.cholesky(S + jitter * np.eye(n))
            return L[np.tril_indices(n)]
        except np.linalg.LinAlgError:
            jitter *= 10
    # Fallback eigenvalue repair
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, eps)
    S_repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
    S_repaired = 0.5 * (S_repaired + S_repaired.T)
    L = np.linalg.cholesky(S_repaired + eps * np.eye(n))
    return L[np.tril_indices(n)]

If adaptive jitter or eigenvalue repair is used, log it if the project has a logging mechanism. Otherwise, count the number of repairs and print a short summary in the notebook.

⸻

5. Macro Features

I already have macro features ready in the project.

Inspect the repository and identify where macro features are stored or created. Use them as part of the MLP input.

Requirements:

1. Macro features must be aligned to the rebalance date (t).
2. Use only macro information available at or before (t).
3. Do not use future macro values.
4. If macro features are monthly or lower frequency, forward-fill only from past available values.
5. Standardize macro features using training-period statistics only.
6. Do not fit macro scalers on validation or test data.

If the macro feature file has a date column, merge by rebalance date using an as-of merge:

import pandas as pd
merged = pd.merge_asof(
    sleeve_df.sort_values("rebalance_date"),
    macro_df.sort_values("date"),
    left_on="rebalance_date",
    right_on="date",
    direction="backward"
)

If the current project already has a macro feature alignment utility, reuse it instead of creating a duplicate.

⸻

6. Final MLP Feature Vector

For each sleeve-date:

features = np.concatenate([cholesky_features, macro_features])

If there are K macro features, then:

input_dim = 55 + K

Standardize the full feature vector using training-period statistics only.

Use something like:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

Save the feature scaler if the project has an artifacts directory:

artifacts/scalers/mlp_feature_scaler.pkl

If the notebook is self-contained and artifact saving is not already used, at least keep the scaler object in memory and clearly state that it was fit only on training data.

⸻

7. Training Target

For each sleeve-date, compute the realized covariance matrix over the next 21 trading days:

[
S_{t+1:t+21}^{21}
]

This future realized covariance is used only for the MLP training loss.

The MLP does not forecast this covariance matrix. It uses it only to evaluate the realized variance of its chosen weights during training.

The model loss is:

[
L_{\text{var}}

w_t^\top S_{t+1:t+21}^{21} w_t
]

Implement batch variance loss:

# weights: shape (batch, 10)
# future_cov: shape (batch, 10, 10)
var = torch.einsum("bi,bij,bj->b", weights, future_cov, weights)
loss_var = var.mean()

⸻

8. Equal-Weight Regularization

Add equal-weight regularization to prevent unstable corner solutions:

[
L =
w_t^\top S_{t+1:t+21}^{21} w_t
+
\lambda_{eq}
|w_t - w^{EW}|_2^2
]

where:

[
w^{EW} = (0.1,\ldots,0.1)
]

Implementation:

equal_weight = torch.full_like(weights, 1.0 / weights.shape[1])
loss_eq = ((weights - equal_weight) ** 2).sum(dim=1).mean()
loss = loss_var + lambda_eq * loss_eq

This regularization is important because the 21-day realized covariance target is noisy and the MLP can otherwise overfit by learning unstable concentrated allocations.

⸻

9. MLP Architecture

Implement a simple PyTorch model inside the notebook, unless the project already has a clean model module where it belongs.

import torch
import torch.nn as nn
class MLPWeightModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 10))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        logits = self.net(x)
        weights = torch.softmax(logits, dim=-1)
        return weights

The output must always be 10 weights.

Add checks during development:

weights = model(batch_x)
assert weights.shape[-1] == 10
assert torch.all(weights >= 0)
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(weights.shape[0], device=weights.device),
    atol=1e-5
)

⸻

10. Dataset Objects

Create a simple PyTorch dataset for the MLP:

from torch.utils.data import Dataset
class MLPPortfolioDataset(Dataset):
    def __init__(self, X, future_covs, metadata=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.future_covs = torch.tensor(future_covs, dtype=torch.float32)
        self.metadata = metadata
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.future_covs[idx]

Where:

* X has shape (num_observations, input_dim)
* future_covs has shape (num_observations, 10, 10)

For validation and test backtesting, also keep metadata that maps each observation to:

* rebalance date
* sleeve id
* ordered PERMNOs
* future return matrix over the next 21 days

This metadata is necessary to reconstruct portfolio returns.

⸻

11. Hyperparameter Grid

Use this primary grid:

hidden_dim_grid = [64, 128]
num_layers_grid = [2, 3]
lambda_eq_grid = [0.001, 0.01, 0.1]
learning_rate_grid = [1e-3, 3e-4]
weight_decay_grid = [0.0, 1e-5]

Training settings:

batch_size = 256
max_epochs = 200
patience = 20
optimizer = "Adam"

If the full grid is too slow, use this smaller grid first:

hidden_dim_grid = [64, 128]
num_layers_grid = [2]
lambda_eq_grid = [0.001, 0.01, 0.1]
learning_rate_grid = [1e-3]
weight_decay_grid = [1e-5]

Do not tune on the test period.

⸻

12. Training Loop

Implement a standard PyTorch training loop.

Pseudo-code:

def train_one_mlp_model(
    model,
    train_loader,
    optimizer,
    lambda_eq,
    device,
):
    model.train()
    total_loss = 0.0
    total_var_loss = 0.0
    total_eq_loss = 0.0
    n_obs = 0
    for X_batch, future_cov_batch in train_loader:
        X_batch = X_batch.to(device)
        future_cov_batch = future_cov_batch.to(device)
        weights = model(X_batch)
        var = torch.einsum("bi,bij,bj->b", weights, future_cov_batch, weights)
        loss_var = var.mean()
        equal_weight = torch.full_like(weights, 1.0 / weights.shape[1])
        loss_eq = ((weights - equal_weight) ** 2).sum(dim=1).mean()
        loss = loss_var + lambda_eq * loss_eq
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = X_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_var_loss += loss_var.item() * batch_size
        total_eq_loss += loss_eq.item() * batch_size
        n_obs += batch_size
    return {
        "loss": total_loss / n_obs,
        "loss_var": total_var_loss / n_obs,
        "loss_eq": total_eq_loss / n_obs,
    }

⸻

13. Validation Rule

Select the model only by validation-period annualized realized portfolio volatility over 2014–2020.

For each trained model:

1. Generate MLP weights for every validation sleeve-date.
2. Equal-weight capital across sleeves.
3. Hold weights for the next 21 trading days.
4. Concatenate validation daily portfolio returns.
5. Compute:

[
\sigma_{\text{ann}}

\sqrt{252}
\operatorname{Std}(r_p)
]

Select the model with the lowest validation annualized volatility.

Do not select using:

* return
* Sharpe ratio
* CVaR
* maximum drawdown
* turnover
* HHI
* test performance
* training loss

Training loss is only a diagnostic. Validation realized volatility is the selection metric.

⸻

14. Early Stopping

Use early stopping based on validation annualized realized portfolio volatility, not validation training loss.

If full portfolio backtesting every epoch is too slow, use one of the following:

Option A: Simple and Correct

Train for fixed max_epochs = 200 for each configuration, then evaluate the full validation backtest once at the end.

This is slower but less complicated.

Option B: Efficient Early Stopping

Every 5 or 10 epochs, run validation backtest and track validation annualized volatility.

Use:

patience = 20

Select the checkpoint with the lowest validation annualized volatility.

Do not use test data for early stopping.

⸻

15. Backtest Procedure for MLP

Use the same evaluation sleeve construction already used in the project.

At each rebalance date:

1. Construct non-overlapping 10-stock same-industry sleeves.
2. For each sleeve:
    * compute Cholesky covariance features from previous 126 returns;
    * merge macro features available at that date;
    * standardize features using the training scaler;
    * generate softmax weights using the MLP.
3. Equal-weight capital across all active sleeves.
4. Hold for the next 21 trading days.
5. Record daily aggregate portfolio returns.

If there are (G_t) active sleeves at rebalance date (t), each sleeve receives:

[
\frac{1}{G_t}
]

The aggregate stock weight should be:

[
W_{i,t}

\frac{1}{G_t}w_{g,i,t}
]

where (w_{g,i,t}) is the MLP-generated weight inside sleeve (g).

⸻

16. Benchmarks to Compare

In MLP.ipynb, compare the MLP model against all available benchmarks:

1. Equal Weight
2. Raw Sample Covariance Minimum-Variance
3. Ledoit-Wolf Shrinkage Minimum-Variance
4. EWMA Minimum-Variance, if already available or easy to add
5. Existing conditional covariance forecasting model, if available
6. New MLP Softmax Weight Learner

Use the same:

* rebalance dates
* sleeve construction
* return data
* transaction-cost assumptions
* metric functions
* train / validation / test periods

Only the portfolio-weight generation method should differ.

Do not modify the existing conditional covariance forecasting model just to make it look better or worse.

⸻

17. Metrics

For validation selection, use only annualized volatility.

For final test reporting, compute:

annualized_volatility
volatility_reduction_vs_sample_cov_mv
annualized_return
sharpe_ratio
cvar_95
maximum_drawdown
average_turnover
average_hhi
average_max_weight
net_return_after_transaction_costs

HHI:

[
HHI_t

\sum_i W_{i,t}^2
]

CVaR 95%:

losses = -returns
var95 = np.quantile(losses, 0.95)
cvar95 = losses[losses >= var95].mean()

Annualized volatility:

ann_vol = np.sqrt(252) * np.std(returns, ddof=1)

Annualized return:

ann_return = np.prod(1 + returns) ** (252 / len(returns)) - 1

Sharpe ratio with zero risk-free rate:

sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns, ddof=1)

Maximum drawdown:

wealth = np.cumprod(1 + returns)
running_peak = np.maximum.accumulate(wealth)
drawdown = wealth / running_peak - 1
mdd = drawdown.min()

⸻

18. Transaction Costs

If the existing project already implements transaction costs, reuse that code.

If not, add a simple turnover-based cost calculation.

At rebalance date (t), compute turnover using aggregate stock weights:

[
\text{Turnover}_t

\sum_i
|W_{i,t} - W_{i,t-1}^{post}|
]

Use the union of current and previous holdings.

If transaction cost is 10 bps per one-way turnover:

[
\text{Cost}_t

0.001 \times \text{Turnover}_t
]

Subtract the cost from the first daily return after rebalance.

Report net return after transaction costs if feasible.

⸻

19. Notebook Structure

Create:

MLP.ipynb

The notebook should have the following sections:

1. Research Motivation

Explain that the previous covariance-generation / denoising approach is simplified into a direct decision-focused weight learner.

Include this sentence:

The MLP does not forecast a covariance matrix. It directly learns long-only portfolio weights using a softmax output layer and is trained to minimize next-month realized portfolio variance.

2. Experimental Setup

Show this table:

Item	Setting
Data	CRSP daily equities
Universe	Dynamic market-cap top 500
Sleeve size	10 same-industry stocks
Lookback	126 trading days
Holding period	21 trading days
Train	2000–2013
Validation	2014–2020
Test	2021–2025
Features	Cholesky covariance features + macro features

3. Feature Construction

Show code and explanation for:

* sample covariance;
* Cholesky feature extraction;
* macro feature merge;
* feature standardization.

4. Dataset Construction

Show:

* number of training observations;
* number of validation observations;
* number of test observations;
* number of macro features;
* final input dimension;
* example feature vector shape;
* example future covariance shape.

5. MLP Architecture

Show the model class and confirm that softmax output weights are long-only and sum to one.

6. Training Loss

Explain:

[
L =
w^\top S_{future} w
+
\lambda_{eq}|w-w^{EW}|^2
]

7. Training and Validation

Show:

* hyperparameter grid;
* training loss curves;
* validation volatility table;
* selected configuration.

8. Final Test Comparison

Show final 2021–2025 comparison table across benchmarks.

9. Plots

Include:

* cumulative wealth;
* rolling 63-day volatility;
* turnover comparison;
* HHI / concentration comparison.

10. Interpretation

Use conditional logic:

If MLP beats benchmarks:

The result suggests that direct decision-focused weight learning added value beyond explicit covariance estimation in this experimental setting.

If MLP loses:

The result suggests that conventional covariance-based estimators remained more robust than direct neural weight learning, likely because the MLP overfit noisy realized variance targets or failed to learn stable allocation structure.

Do not overclaim.

⸻

20. Files to Add if Needed

Inspect the existing project first.

If useful, add:

src/mlp_features.py
src/mlp_weight_model.py
src/mlp_train.py
src/mlp_backtest.py
MLP.ipynb

But if the notebook can be self-contained using existing utilities, keep it simple.

Do not delete existing diffusion or forecasting files.

⸻

21. Required Checks

Add checks in the notebook or helper functions:

1. Cholesky feature vector has dimension 55.
2. Final feature vector has dimension 55 + K, where K is the number of macro features.
3. Feature scaler is fit only on training data.
4. MLP output has shape (batch_size, 10).
5. MLP weights are nonnegative.
6. MLP weights sum to one.
7. Training observations use only 2000–2013.
8. Validation selection uses only 2014–2020.
9. Test evaluation uses only 2021–2025.
10. Same sleeves are used across MLP and benchmark comparisons.
11. Aggregate portfolio weights sum to one.

⸻

22. Definition of Done

The implementation is complete when:

* MLP.ipynb runs top to bottom.
* Cholesky + macro features are built correctly.
* Feature scaler is fit only on training-period observations.
* MLP outputs valid long-only weights that sum to one.
* Training uses only 2000–2013.
* Hyperparameters are selected only by 2014–2020 annualized realized volatility.
* Final test is evaluated only on 2021–2025.
* Test comparison includes the MLP and all available benchmarks.
* Notebook includes tables, plots, and honest interpretation.
* Existing covariance forecasting code remains intact.
