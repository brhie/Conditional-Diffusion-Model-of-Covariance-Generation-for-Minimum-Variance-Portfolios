"""
macro_features.py
-----------------
Macro/market-regime conditioning features for the conditional diffusion model.

The conditioning vector is extended from the 55-dim covariance log-vech to
include K macro features, giving a (55 + K)-dim input to the denoiser.

Feature inventory
-----------------
CRSP-derived (always available, computed from cleaned daily return panel):

  log_mkt_var_21d
    Log realized equal-weighted (EW) market variance over the previous 21
    trading days.  Captures the short-horizon volatility regime.
    Research: Andersen, Bollerslev, Diebold & Labys (2003, JASA);
              Shephard & Sheppard (2010, J. Econometrics) HEAVY model.

  log_mkt_var_126d
    Log realized EW market variance over the previous 126 trading days.
    Captures the long-run persistent volatility component.
    Research: Ding, Granger & Engle (1993, J. Empirical Finance) long memory;
              Corsi (2009, JFEC) HAR-RV model.

  mkt_ret_21d
    Cumulative EW market return over the previous 21 trading days.
    Captures the leverage effect: large negative returns predict higher future
    covariance (asymmetric response of volatility to returns).
    Research: Black (1976) leverage hypothesis;
              Christie (1982, J. Fin. Econ.);
              Engle & Ng (1993, J. Finance) news impact curve.

  avg_pairwise_corr_126d
    Implied equicorrelation across the eligible equity universe, estimated over
    the previous 126 trading days via:
        rho = (N * Var(EW) / avg_Var_i - 1) / (N - 1)
    High values indicate systematic risk dominates idiosyncratic risk, which
    changes the structure of the next-period covariance matrix.
    Research: Pollet & Wilson (2010, J. Finance) average correlation and
              stock market returns;
              Driessen, Maenhout & Vilkov (2012, RFS) correlation risk premium.

External / FRED (optional; loaded from data/interim/macro_external.parquet):

  log_vix
    Log of the CBOE Volatility Index (VIX).  The VIX subsumes market
    participants' expectations about near-term equity variance and is the
    dominant single predictor of realized cross-sectional covariance.
    Research: Engle & Figlewski (2012, J. Derivatives);
              Bollerslev, Tauchen & Zhou (2009, RFS) variance risk premium.

  term_spread
    10-year Treasury yield minus 2-year Treasury yield (percentage points).
    An inverted yield curve forecasts recession, which sharply raises equity
    correlations and conditional covariance.
    Research: Estrella & Mishkin (1998, Rev. Econ. Stat.);
              Wright (2006, J. Business & Econ. Stat.);
              Fama (1990, J. Finance) production, investment and the term structure.

  credit_spread
    ICE BofA US Corporate Option-Adjusted Spread (FRED: BAMLC0A0CMEY,
    percentage points).  Widens during financial distress, predicting elevated
    cross-sectional covariance.
    Research: Gilchrist & Zakrajsek (2012, AER) excess bond premium;
              Collin-Dufresne, Goldstein & Martin (2001, J. Finance).

Usage
-----
K = number of features included (4 CRSP-only or 7 with external data).
Total conditioning dim = 55 + K.

The macro DataFrame produced here contains RAW (unscaled) feature values.
The conditioning_scaler (fitted on training data only) scales the full
(55 + K)-dim concatenated vector.  Do not pre-scale macro features separately.

Leakage guarantee
-----------------
All features at anchor date t use only information in [t-T+1, t] for lookback
window T.  External features use the value ON t (or the last available value
before t via forward-fill).  No future information is introduced.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature name registries
# ---------------------------------------------------------------------------

CRSP_MACRO_FEATURES: List[str] = [
    "log_mkt_var_21d",
    "log_mkt_var_126d",
    "mkt_ret_21d",
    "avg_pairwise_corr_126d",
]

EXTERNAL_MACRO_FEATURES: List[str] = [
    "log_vix",
    "term_spread",
    "credit_spread",
]

ALL_MACRO_FEATURES: List[str] = CRSP_MACRO_FEATURES + EXTERNAL_MACRO_FEATURES

# Path where the active feature list is persisted alongside the scalers
FEATURE_NAMES_FILENAME = "macro_feature_names.json"


# ---------------------------------------------------------------------------
# CRSP-derived feature computation
# ---------------------------------------------------------------------------

def compute_crsp_macro_features(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    lookback_short: int = 21,
    lookback_long: int = 126,
    min_stocks_corr: int = 20,
) -> pd.DataFrame:
    """
    Compute the 4 CRSP-derived macro conditioning features at every viable
    trading date in the dataset.

    Parameters
    ----------
    crsp_df : cleaned CRSP panel with columns [date, permno, ret_total]
    trading_dates : full sorted trading calendar
    lookback_short : short variance window (21 days)
    lookback_long : long variance/correlation window (126 days)
    min_stocks_corr : minimum eligible stocks required for correlation feature

    Returns
    -------
    pd.DataFrame indexed by date with columns:
      [log_mkt_var_21d, log_mkt_var_126d, mkt_ret_21d, avg_pairwise_corr_126d]
    """
    td_sorted = trading_dates.sort_values()
    td_arr = td_sorted.to_numpy()

    # ---- Build wide return panel -----------------------------------------
    # Pivot to (date × permno); NaN where stock is absent that day.
    logger.info("Building wide return panel for macro feature computation …")
    ret_panel = crsp_df.pivot_table(
        index="date", columns="permno", values="ret_total"
    ).sort_index()

    dates_in_panel = ret_panel.index

    # ---- EW daily returns (mean across non-NaN stocks each day) -----------
    ew_ret = ret_panel.mean(axis=1, skipna=True)  # (n_dates,)

    # ---- Rolling features 1-3 via pandas rolling --------------------------
    # Using min_periods to avoid NaN propagation from sparse early dates.
    _log_clip = lambda s, min_val=1e-10: np.log(s.clip(lower=min_val))

    log_mkt_var_21d  = _log_clip(ew_ret.rolling(lookback_short, min_periods=lookback_short).var())
    log_mkt_var_126d = _log_clip(ew_ret.rolling(lookback_long,  min_periods=lookback_long ).var())

    mkt_ret_21d = (
        (1 + ew_ret)
        .rolling(lookback_short, min_periods=lookback_short)
        .apply(lambda x: float(np.prod(x)) - 1.0, raw=True)
    )

    # ---- Feature 4: implied equicorrelation (loop per date) ---------------
    # rho = (N * Var(EW) / avg_Var_i - 1) / (N - 1)
    # where EW and avg_Var_i are both estimated on the SAME set of stocks
    # with complete lookback_long-day return histories.
    logger.info(
        "Computing avg_pairwise_corr_126d at %d dates (may take ~30–60s) …",
        len(dates_in_panel),
    )

    equicorr_series = pd.Series(np.nan, index=dates_in_panel, dtype=float)

    # Precompute position of each date in the sorted trading-date array
    date_pos = {pd.Timestamp(d): i for i, d in enumerate(td_arr)}

    for date in dates_in_panel:
        pos = date_pos.get(pd.Timestamp(date))
        if pos is None or pos < lookback_long:
            continue

        lookback_td = td_sorted[pos - lookback_long + 1 : pos + 1]

        # Filter panel to the lookback window and keep complete-return stocks
        sub = ret_panel.loc[ret_panel.index.isin(lookback_td)]
        if len(sub) < lookback_long:
            continue  # some dates missing from the panel

        complete_mask = sub.notna().all(axis=0)
        sub_c = sub.loc[:, complete_mask]  # (126, N_complete)
        N = sub_c.shape[1]
        if N < min_stocks_corr:
            continue

        R = sub_c.values.astype(np.float64)   # (lookback_long, N)
        ew = R.mean(axis=1)                    # (lookback_long,)
        var_ew = float(np.var(ew, ddof=1))

        stock_vars = np.var(R, axis=0, ddof=1)
        avg_var = float(stock_vars.mean())

        if avg_var < 1e-10:
            continue

        rho = (N * var_ew / avg_var - 1.0) / (N - 1.0)
        equicorr_series[date] = float(np.clip(rho, -1.0, 1.0))

    # ---- Assemble output DataFrame ----------------------------------------
    out = pd.DataFrame(
        {
            "log_mkt_var_21d":        log_mkt_var_21d,
            "log_mkt_var_126d":       log_mkt_var_126d,
            "mkt_ret_21d":            mkt_ret_21d,
            "avg_pairwise_corr_126d": equicorr_series,
        },
        index=dates_in_panel,
    )
    out.index = pd.to_datetime(out.index)
    out = out.dropna(how="all")

    n_valid = out.notna().all(axis=1).sum()
    logger.info(
        "CRSP macro features: %d dates with all 4 features complete "
        "(out of %d total dates).",
        n_valid, len(out),
    )
    return out


# ---------------------------------------------------------------------------
# External feature loading
# ---------------------------------------------------------------------------

def load_external_macro_features(path: str | Path) -> Optional[pd.DataFrame]:
    """
    Load pre-downloaded FRED macro features from a Parquet file.

    Expected columns: [log_vix, term_spread, credit_spread]
    The DataFrame should be indexed by date (or have a 'date' column).

    Returns None if the file does not exist.

    How to create this file
    -----------------------
    Download via pandas-datareader or direct FRED API:
      VIXCLS       → log(VIX)
      DGS10, DGS2  → term_spread = DGS10 - DGS2
      BAMLC0A0CMEY → credit_spread (IG OAS, ICE BofA US Corporate)

    See scripts/00_download_macro_data.py for a ready-to-run download script.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(
            "External macro data file not found: %s. "
            "Using only 4 CRSP-derived features. "
            "Run scripts/00_download_macro_data.py to add VIX, term spread, "
            "and credit spread conditioning.",
            path,
        )
        return None

    df = pd.read_parquet(path)

    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    missing = [c for c in EXTERNAL_MACRO_FEATURES if c not in df.columns]
    if missing:
        logger.warning(
            "External macro file missing columns: %s. "
            "Those features will be omitted.",
            missing,
        )
        df = df[[c for c in EXTERNAL_MACRO_FEATURES if c in df.columns]]

    logger.info(
        "Loaded external macro features from %s: %d rows, columns %s",
        path, len(df), list(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# Combined macro feature DataFrame
# ---------------------------------------------------------------------------

def build_macro_feature_df(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    external_path: Optional[str | Path] = None,
    lookback_short: int = 21,
    lookback_long: int = 126,
    min_stocks_corr: int = 20,
) -> pd.DataFrame:
    """
    Build the complete macro feature DataFrame for all dates in the CRSP panel.

    External features are added if *external_path* points to an existing file;
    otherwise only the 4 CRSP-derived features are included.

    Returns
    -------
    pd.DataFrame indexed by date.
    Columns: CRSP_MACRO_FEATURES (always) + subset of EXTERNAL_MACRO_FEATURES.
    All values are raw (not standardized).
    """
    crsp_macro = compute_crsp_macro_features(
        crsp_df=crsp_df,
        trading_dates=trading_dates,
        lookback_short=lookback_short,
        lookback_long=lookback_long,
        min_stocks_corr=min_stocks_corr,
    )

    if external_path is not None:
        ext = load_external_macro_features(external_path)
    else:
        ext = None

    if ext is not None and not ext.empty:
        # Forward-fill external data to align with CRSP trading dates
        combined_idx = crsp_macro.index.union(ext.index)
        ext_reindexed = ext.reindex(combined_idx).ffill()
        ext_on_crsp = ext_reindexed.reindex(crsp_macro.index)
        macro_df = pd.concat([crsp_macro, ext_on_crsp], axis=1)
        logger.info(
            "Combined macro feature columns: %s", list(macro_df.columns)
        )
    else:
        macro_df = crsp_macro
        logger.info(
            "Macro feature columns (CRSP-only): %s", list(macro_df.columns)
        )

    return macro_df


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_macro_vector(
    macro_df: pd.DataFrame,
    date: pd.Timestamp,
) -> np.ndarray:
    """
    Return the (K,) raw macro feature vector for a given date.

    If the exact date is not in macro_df.index, use the most recent prior
    observation (forward-fill).  If no prior observation exists, return zeros.

    Parameters
    ----------
    macro_df : pd.DataFrame indexed by trading date
    date : anchor date

    Returns
    -------
    np.ndarray of shape (K,), dtype float64
    """
    K = macro_df.shape[1]
    idx = macro_df.index.searchsorted(date, side="right") - 1
    if idx < 0:
        logger.debug("No macro data before %s; returning zeros.", date.date())
        return np.zeros(K, dtype=np.float64)
    row = macro_df.iloc[idx]
    vec = row.values.astype(np.float64)
    # Replace NaN with zero (rare edge at the start of the sample)
    vec = np.where(np.isfinite(vec), vec, 0.0)
    return vec


def save_macro_feature_names(feature_names: List[str], save_dir: str | Path) -> None:
    """Persist the list of active macro feature names alongside the scalers."""
    path = Path(save_dir) / FEATURE_NAMES_FILENAME
    with open(path, "w") as fh:
        json.dump(feature_names, fh)
    logger.info("Saved macro feature names → %s", path)


def load_macro_feature_names(save_dir: str | Path) -> List[str]:
    """Load the list of active macro feature names from the scaler directory."""
    path = Path(save_dir) / FEATURE_NAMES_FILENAME
    if not path.exists():
        logger.warning(
            "macro_feature_names.json not found in %s. "
            "Assuming no macro features (condition_dim = 55).",
            save_dir,
        )
        return []
    with open(path) as fh:
        names = json.load(fh)
    logger.info("Loaded macro feature names: %s", names)
    return names
