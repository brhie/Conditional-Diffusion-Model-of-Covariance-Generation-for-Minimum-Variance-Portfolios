"""
05b_validate_denoising.py
--------------------------
Phase 5B: Evaluate all effective denoising-model validation configurations.

Grid
----
Models : schedule ∈ {linear, quadratic} × T ∈ {25, 50}  → 4 models
rho    : {0.05, 0.10, 0.20}                               → s_start = max(1, round(rho*T))
M      : {1, 20}                                          → draws averaged in cov space
Boundary: rho=0  → pure raw sample covariance (shared with the forecasting validation)

Effective configurations: 4 × 3 × 2 + 1 = 25

Selection metric: annualized realized GMV portfolio volatility (2014–2020 only).
No test data touched.

Disk caching
------------
Reuses results/validation/cache/sample_covs.pkl (built by Phase 5).
Denoising-specific caches saved to results/validation/cache/:
    denoised_schedule-{sched}_T-{T}_rho-{rho_str}.pkl
        {(date, sleeve_id): [cov_matrix_1, ..., cov_matrix_M_max]}

Outputs
-------
results/validation/denoising_validation_grid_results.csv
results/validation/denoising_top5_configurations.csv
artifacts/selected_denoising_model/selected_denoising_model_config.yaml
artifacts/selected_denoising_model/selected_denoising_model.pt
"""

from __future__ import annotations

import logging
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from tqdm.auto import tqdm as _tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_one_rebalance
from src.config import get_config
from src.diffusion import DDPMScheduler
from src.generate import deterministic_scenario_seed, generate_denoised_covariances
from src.gmv import solve_long_only_gmv
from src.metrics import annualized_volatility
from src.train import load_denoising_model
from src.transforms import covariance_to_log_vech
from src.utils import get_device, get_logger, set_global_seed

logger = get_logger("05b_validate_denoising", logging.INFO)

# Denoising-specific hyperparameter grids
SCHEDULE_GRID = ["linear", "quadratic"]
T_GRID = [25, 50]
RHO_GRID = [0.05, 0.10, 0.20]
M_GRID = [1, 20]
M_MAX = 20   # max draws generated (= largest M value)


def _rho_str(rho: float) -> str:
    """Format rho for filenames: 0.05 → '0.05'"""
    return f"{rho:.2f}"


# ---------------------------------------------------------------------------
# Helper: run portfolio backtest for given weights
# ---------------------------------------------------------------------------

def run_validation_backtest(
    crsp_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    val_rebalance_dates: List[pd.Timestamp],
    sleeve_weights_by_date: Dict[pd.Timestamp, Dict[int, np.ndarray]],
    sleeve_permnos_by_date: Dict[pd.Timestamp, Dict[int, List[int]]],
    horizon_days: int = 21,
) -> List[float]:
    """Compute the full validation daily return stream for given weights."""
    all_returns: List[float] = []
    for date in val_rebalance_dates:
        if date not in sleeve_weights_by_date:
            continue
        wts = sleeve_weights_by_date[date]
        perms = sleeve_permnos_by_date.get(date, {})
        if not wts:
            continue
        daily_rets, _, _, _ = run_one_rebalance(
            sleeve_weights=wts,
            sleeve_permnos=perms,
            crsp_df=crsp_df,
            trading_dates=trading_dates,
            rebalance_date=date,
            horizon_days=horizon_days,
        )
        all_returns.extend(daily_rets)
    return all_returns


# ---------------------------------------------------------------------------
# Helper: pretty-print leaderboard
# ---------------------------------------------------------------------------

def _print_leaderboard(results: list[dict], top_n: int = 5) -> None:
    if not results:
        return
    df = pd.DataFrame(results).sort_values("validation_annualized_realized_volatility")
    top = df.head(top_n)
    lines = ["", f"  ── Current top-{top_n} ──────────────────────────────────"]
    lines.append("  {:>3}  {:>10}  {:>4}  {:>5}  {:>3}  {:>9}".format(
        "rk", "schedule", "T", "rho", "M", "val_vol"
    ))
    lines.append("  " + "-" * 46)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        sched = str(row["schedule_type"])[:10]
        T_ = int(row["diffusion_steps_T"]) if pd.notna(row.get("diffusion_steps_T")) else "—"
        rho_ = f"{row['rho']:.2f}" if pd.notna(row.get("rho")) else "—"
        M_ = int(row["draws_M"]) if pd.notna(row.get("draws_M")) else "—"
        vol = row["validation_annualized_realized_volatility"]
        lines.append("  {:>3}  {:>10}  {:>4}  {:>5}  {:>3}  {:>9.6f}".format(
            rank, sched, T_, rho_, M_, vol
        ))
    lines.append("")
    msg = "\n".join(lines)
    if _has_tqdm:
        _tqdm.write(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = get_config()

    interim_dir = Path("data/interim")
    model_dir = Path("artifacts/models")
    scaler_dir = Path("artifacts/scalers")
    val_results_dir = Path("results/validation")
    selected_dir = Path("artifacts/selected_denoising_model")
    cache_dir = val_results_dir / "cache"

    for d in [val_results_dir, selected_dir, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)

    set_global_seed(cfg.random_seed)
    device = get_device()

    logger.info("=" * 60)
    logger.info("STEP 5B – Validate denoising hyperparameters (2014–2020)")
    logger.info("Device: %s", device)
    logger.info("=" * 60)

    # ---- Load data -------------------------------------------------------
    crsp_df = pd.read_parquet(interim_dir / "cleaned_crsp_daily.parquet")
    crsp_df["date"] = pd.to_datetime(crsp_df["date"])
    cal_df = pd.read_parquet(interim_dir / "trading_calendar.parquet")
    trading_dates = pd.DatetimeIndex(pd.to_datetime(cal_df["date"]).sort_values())
    reb_df = pd.read_parquet(interim_dir / "rebalance_dates.parquet")
    reb_df["rebalance_date"] = pd.to_datetime(reb_df["rebalance_date"])

    val_dates_all = sorted(
        reb_df[reb_df["split"] == "validation"]["rebalance_date"].tolist()
    )

    # Load scalers (same ones used throughout – conditioning scaler handles log-vech)
    import pickle as _pickle
    with open(scaler_dir / "conditioning_scaler.pkl", "rb") as fh:
        cond_scaler = _pickle.load(fh)

    lkb = cfg.rolling_windows["lookback_days"]
    hor = cfg.rolling_windows["horizon_days"]

    # =====================================================================
    # STAGE 1 – Load cached sample covariances (built by Phase 5)
    # =====================================================================
    sample_cov_cache_path = cache_dir / "sample_covs.pkl"
    if not sample_cov_cache_path.exists():
        logger.error(
            "Sample covariance cache not found at %s. "
            "Please run Phase 5 (05_validate_hyperparameters.py) first to build it.",
            sample_cov_cache_path,
        )
        sys.exit(1)

    logger.info("Loading sample covariance cache from %s …", sample_cov_cache_path)
    with open(sample_cov_cache_path, "rb") as fh:
        _sc = _pickle.load(fh)
    sleeve_permnos_by_date    = _sc["sleeve_permnos_by_date"]
    sleeve_sample_cov_by_date = _sc["sleeve_sample_cov_by_date"]
    n_sleeves = sum(len(v) for v in sleeve_sample_cov_by_date.values())
    logger.info("  Loaded %d dates, %d sleeves.", len(sleeve_permnos_by_date), n_sleeves)

    validation_results: list[dict] = []

    # =====================================================================
    # STAGE 2 – rho=0 boundary (raw sample covariance GMV) – once
    # =====================================================================
    logger.info("\nEvaluating rho=0 (raw sample covariance GMV boundary) …")
    sample_wts_by_date: Dict[pd.Timestamp, Dict[int, np.ndarray]] = {}
    for date in val_dates_all:
        sample_wts_by_date[date] = {}
        for sid, S_hist in sleeve_sample_cov_by_date.get(date, {}).items():
            try:
                sample_wts_by_date[date][sid] = solve_long_only_gmv(S_hist)
            except Exception:
                pass

    sample_returns = run_validation_backtest(
        crsp_df, trading_dates, val_dates_all,
        sample_wts_by_date, sleeve_permnos_by_date, hor,
    )
    sample_vol = annualized_volatility(np.array(sample_returns)) if sample_returns else np.nan
    validation_results.append({
        "schedule_type":                              "not_applicable",
        "diffusion_steps_T":                          None,
        "rho":                                        0.0,
        "draws_M":                                    None,
        "validation_annualized_realized_volatility":  sample_vol,
        "is_sample_covariance_boundary":              True,
    })
    logger.info("  rho=0.00  M=—  →  val_vol=%.6f  (sample covariance boundary)", sample_vol)

    # =====================================================================
    # STAGE 3 – Denoising configurations
    # =====================================================================
    total_models = len(SCHEDULE_GRID) * len(T_GRID)
    total_combos = total_models * len(RHO_GRID) * len(M_GRID)
    combos_done = 0

    logger.info(
        "\nEvaluating %d models × %d rho × %d M = %d denoising configurations …",
        total_models, len(RHO_GRID), len(M_GRID), total_combos,
    )

    model_pairs = [(s, T) for s in SCHEDULE_GRID for T in T_GRID]
    model_iter = (
        _tqdm(model_pairs, desc="Models", unit="model", dynamic_ncols=True)
        if _has_tqdm else model_pairs
    )

    for schedule_type, T in model_iter:
        seed = cfg.random_seed
        ckpt_path = model_dir / f"denoise_schedule-{schedule_type}_T-{T}_seed-{seed}.pt"

        if not ckpt_path.exists():
            logger.warning("  Denoiser checkpoint not found: %s – SKIPPING.", ckpt_path)
            combos_done += len(RHO_GRID) * len(M_GRID)
            continue

        header = f"schedule={schedule_type}  T={T}"
        msg = f"\n── {header} ──────────────────────────────────────"
        if _has_tqdm:
            _tqdm.write(msg)
        else:
            print(msg)

        # Load model and build scheduler once per (schedule, T)
        logger.info("  Loading denoiser checkpoint: %s …", ckpt_path.name)
        model, _ = load_denoising_model(ckpt_path, device=str(device))
        model.eval()
        scheduler = DDPMScheduler(
            schedule_type=schedule_type, T=T,
            beta_min=cfg.training["beta_min"],
            beta_max=cfg.training["beta_max"],
            device=device,
        )

        # ------------------------------------------------------------------
        # STAGE 3a – Generate (or load cached) denoised draws for each rho
        # ------------------------------------------------------------------
        # Cache key includes rho because s_start = f(rho, T) changes the
        # forward-corruption level.
        rho_caches: Dict[float, Dict[Tuple, List[np.ndarray]]] = {}

        for rho in RHO_GRID:
            rho_key = _rho_str(rho)
            cache_path = (
                cache_dir
                / f"denoised_schedule-{schedule_type}_T-{T}_rho-{rho_key}.pkl"
            )

            if cache_path.exists():
                msg = f"  ✓ Loading cached denoised draws (rho={rho_key}) from {cache_path.name}"
                if _has_tqdm:
                    _tqdm.write(msg)
                else:
                    logger.info(msg)
                with open(cache_path, "rb") as fh:
                    rho_caches[rho] = _pickle.load(fh)
            else:
                denoise_cache: Dict[Tuple, List[np.ndarray]] = {}

                all_sleeve_dates = [
                    (date, sid)
                    for date in val_dates_all
                    for sid in sleeve_sample_cov_by_date.get(date, {})
                ]

                gen_iter = (
                    _tqdm(
                        all_sleeve_dates,
                        desc=f"  Denoising rho={rho_key}",
                        unit="sleeve",
                        dynamic_ncols=True,
                        leave=False,
                    )
                    if _has_tqdm else all_sleeve_dates
                )

                for date, sid in gen_iter:
                    S_hist = sleeve_sample_cov_by_date[date][sid]
                    try:
                        cov_vech = covariance_to_log_vech(S_hist, cfg.ridge_epsilon)
                    except Exception as exc:
                        logger.debug(
                            "log_vech failed for sleeve %d at %s: %s", sid, date.date(), exc
                        )
                        continue

                    gen_seed = deterministic_scenario_seed(
                        schedule_type, T, date, sid, base_seed=seed
                    )
                    draws = generate_denoised_covariances(
                        model=model,
                        scheduler=scheduler,
                        condition_vector_raw=cov_vech,
                        conditioning_scaler=cond_scaler,
                        rho=rho,
                        num_draws=M_MAX,
                        seed=gen_seed,
                        device=device,
                    )
                    denoise_cache[(date, sid)] = draws

                with open(cache_path, "wb") as fh:
                    _pickle.dump(denoise_cache, fh, protocol=_pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "  ✓ Saved denoised cache rho=%s (%d sleeve-dates) → %s",
                    rho_key, len(denoise_cache), cache_path,
                )
                rho_caches[rho] = denoise_cache

        # ------------------------------------------------------------------
        # STAGE 3b – Evaluate every (rho, M) combination
        # ------------------------------------------------------------------
        combo_pairs = [(rho, M) for rho in RHO_GRID for M in M_GRID]
        combo_iter = (
            _tqdm(combo_pairs, desc="  Configs", unit="cfg",
                  dynamic_ncols=True, leave=False)
            if _has_tqdm else combo_pairs
        )

        model_best_vol = float("inf")
        model_results = []

        for rho, M in combo_iter:
            if rho not in rho_caches:
                continue

            denoise_cache = rho_caches[rho]
            wts_by_date: Dict[pd.Timestamp, Dict[int, np.ndarray]] = {}

            for date in val_dates_all:
                wts_by_date[date] = {}
                for sid, S_hist in sleeve_sample_cov_by_date.get(date, {}).items():
                    key = (date, sid)
                    if key not in denoise_cache:
                        continue
                    draws_M = denoise_cache[key][:M]
                    # Average M denoised covariance matrices in covariance space
                    denoised_cov = np.mean(np.stack(draws_M, axis=0), axis=0)
                    denoised_cov = 0.5 * (denoised_cov + denoised_cov.T)
                    try:
                        wts_by_date[date][sid] = solve_long_only_gmv(denoised_cov)
                    except Exception:
                        pass

            daily_rets = run_validation_backtest(
                crsp_df, trading_dates, val_dates_all,
                wts_by_date, sleeve_permnos_by_date, hor,
            )
            vol = annualized_volatility(np.array(daily_rets)) if daily_rets else np.nan

            row = {
                "schedule_type":                              schedule_type,
                "diffusion_steps_T":                          T,
                "rho":                                        rho,
                "draws_M":                                    M,
                "validation_annualized_realized_volatility":  vol,
                "is_sample_covariance_boundary":              False,
            }
            validation_results.append(row)
            model_results.append(row)
            combos_done += 1
            model_best_vol = min(model_best_vol, vol)

            global_best = min(
                r["validation_annualized_realized_volatility"]
                for r in validation_results
                if np.isfinite(r["validation_annualized_realized_volatility"])
            )
            marker = " ◀ best so far" if vol <= global_best else ""
            line = (
                f"    schedule={schedule_type:<10}  T={T:>2}  "
                f"rho={rho:.2f}  M={M:>2}  →  val_vol={vol:.6f}  "
                f"({vol*100:.4f}%){marker}"
            )
            if _has_tqdm:
                _tqdm.write(line)
            else:
                print(line)

            if _has_tqdm and hasattr(combo_iter, "set_postfix"):
                combo_iter.set_postfix(
                    rho=f"{rho:.2f}", M=M, vol=f"{vol:.5f}",
                    best=f"{model_best_vol:.5f}",
                )

        if model_results:
            model_df = pd.DataFrame(model_results).sort_values(
                "validation_annualized_realized_volatility"
            )
            best_row = model_df.iloc[0]
            summary = (
                f"\n  ✓ {header}  done — "
                f"best this model: rho={best_row['rho']:.2f}  "
                f"M={int(best_row['draws_M'])}  "
                f"vol={best_row['validation_annualized_realized_volatility']:.6f}  "
                f"[{combos_done}/{total_combos} configs evaluated]"
            )
            if _has_tqdm:
                _tqdm.write(summary)
            else:
                print(summary)

        _print_leaderboard(validation_results, top_n=5)

    # =====================================================================
    # STAGE 4 – Rank and select
    # =====================================================================
    results_df = pd.DataFrame(validation_results)
    results_df = results_df.sort_values(
        "validation_annualized_realized_volatility"
    ).reset_index(drop=True)
    results_df["rank"] = results_df.index + 1
    results_df["selected_primary_model"] = False

    schedule_priority = {"linear": 0, "quadratic": 1, "not_applicable": 2}

    valid_vols = results_df["validation_annualized_realized_volatility"].dropna()
    if valid_vols.empty:
        logger.error("No valid validation results. Cannot select a configuration.")
        return

    best_vol = valid_vols.iloc[0]
    tied = results_df[
        (results_df["validation_annualized_realized_volatility"] - best_vol).abs() < 1e-8
    ].copy()

    def tie_sort_key(row):
        rho_ = row["rho"] if pd.notna(row.get("rho")) else 999.0
        M_ = row["draws_M"] if pd.notna(row.get("draws_M")) else 999
        T_ = row["diffusion_steps_T"] if pd.notna(row.get("diffusion_steps_T")) else 999
        sched_ = schedule_priority.get(str(row["schedule_type"]), 999)
        # Prefer: larger rho first (more denoising), then smaller M, smaller T, linear
        return (-rho_, M_, T_, sched_)

    tied_sorted = tied.apply(tie_sort_key, axis=1).sort_values()
    best_idx = tied_sorted.index[0]
    results_df.loc[best_idx, "selected_primary_model"] = True
    selected = results_df.loc[best_idx].to_dict()

    # ── Final leaderboard ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL DENOISING VALIDATION RANKING (top 10)")
    logger.info("=" * 60)
    top10 = results_df.head(10)
    for _, row in top10.iterrows():
        star = " ◀ SELECTED" if row["selected_primary_model"] else ""
        T_str = f"T={int(row['diffusion_steps_T'])}" if pd.notna(row.get("diffusion_steps_T")) else "T=—"
        rho_str_ = f"rho={row['rho']:.2f}" if pd.notna(row.get("rho")) else "rho=—"
        M_str = f"M={int(row['draws_M'])}" if pd.notna(row.get("draws_M")) else "M=—"
        logger.info(
            "  #%2d  %-12s  %5s  %8s  %-4s  vol=%.6f%s",
            int(row["rank"]),
            row["schedule_type"],
            T_str,
            rho_str_,
            M_str,
            row["validation_annualized_realized_volatility"],
            star,
        )

    logger.info("\n=== SELECTED DENOISING CONFIGURATION ===")
    for k, v in selected.items():
        logger.info("  %-48s  %s", k, v)

    # ---- Save results -----------------------------------------------------
    out_csv = val_results_dir / "denoising_validation_grid_results.csv"
    results_df.to_csv(out_csv, index=False)

    top5 = results_df.head(5)
    top5.to_csv(val_results_dir / "denoising_top5_configurations.csv", index=False)
    logger.info(
        "\nTop-5 denoising configurations:\n%s",
        top5[["schedule_type", "diffusion_steps_T", "rho",
              "draws_M", "validation_annualized_realized_volatility"]].to_string()
    )

    # ---- Save selected config YAML ----------------------------------------
    selected_config = {
        "selected_denoising_model": {
            "schedule_type": str(selected["schedule_type"]),
            "diffusion_steps_T": (
                int(selected["diffusion_steps_T"])
                if pd.notna(selected.get("diffusion_steps_T")) else None
            ),
            "rho": float(selected["rho"]) if pd.notna(selected.get("rho")) else 0.0,
            "draws_M": (
                int(selected["draws_M"])
                if pd.notna(selected.get("draws_M")) else None
            ),
            "validation_metric": "gross_annualized_realized_gmv_portfolio_volatility",
            "validation_annualized_realized_volatility": float(
                selected["validation_annualized_realized_volatility"]
            ),
            "validation_period":  ["2014-01-01", "2020-12-31"],
            "test_period_locked": ["2021-01-01", "2025-12-31"],
        }
    }

    cfg_out = selected_dir / "selected_denoising_model_config.yaml"
    with open(cfg_out, "w") as fh:
        yaml.dump(selected_config, fh, default_flow_style=False)

    # ---- Copy selected checkpoint -----------------------------------------
    sched_sel = selected["schedule_type"]
    T_sel = selected["diffusion_steps_T"]
    if (
        pd.notna(sched_sel)
        and sched_sel != "not_applicable"
        and pd.notna(T_sel)
    ):
        src_ckpt = model_dir / f"denoise_schedule-{sched_sel}_T-{int(T_sel)}_seed-{cfg.random_seed}.pt"
        if src_ckpt.exists():
            shutil.copy2(src_ckpt, selected_dir / "selected_denoising_model.pt")
            logger.info("Copied selected denoising checkpoint → %s", selected_dir)

    logger.info("\nStep 5B complete.")


if __name__ == "__main__":
    main()
