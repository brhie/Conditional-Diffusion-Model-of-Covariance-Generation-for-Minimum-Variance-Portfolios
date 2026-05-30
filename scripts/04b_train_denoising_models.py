"""
04b_train_denoising_models.py
------------------------------
Phase 4B: Train 4 unconditional DDPM denoising models.

Grid: schedule ∈ {linear, quadratic} × T ∈ {25, 50}

The denoiser learns the distribution of historical 126-day sample covariance
matrices in standardized log-vech space.  At inference it acts as an SDEdit-
style nonlinear regularizer: observe the current sample covariance, add mild
noise to step s_start = max(1, round(rho * T)), reverse-denoise back to zero.

Training data: condition_scaled vectors from data/processed/covariance_pairs_train.npz
(the same 97k standardized log-vech conditioning vectors used to train the
forecasting model).

Outputs (per model)
-------------------
artifacts/models/denoise_schedule-{schedule_type}_T-{T}_seed-42.pt
artifacts/training_logs/denoise_schedule-{schedule_type}_T-{T}_seed-42.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.datasets import load_dataset
from src.train import train_unconditional_ddpm
from src.utils import get_logger, set_global_seed

logger = get_logger("04b_train_denoising_models", logging.INFO)

# Denoising model grid (separate from the forecasting model grid)
SCHEDULE_GRID = ["linear", "quadratic"]
T_GRID = [25, 50]


def main() -> None:
    cfg = get_config()

    processed_dir = Path("data/processed")
    model_dir = Path("artifacts/models")
    log_dir = Path("artifacts/training_logs")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.random_seed
    set_global_seed(seed)

    logger.info("=" * 60)
    logger.info(
        "STEP 4B – Train %d unconditional DDPM denoising models "
        "(schedule ∈ {linear, quadratic}, T ∈ {25, 50})",
        len(SCHEDULE_GRID) * len(T_GRID),
    )
    logger.info("=" * 60)

    # ---- Load training data (use condition_scaled only) -------------------
    logger.info("Loading training dataset (condition_scaled vectors) …")
    train_ds = load_dataset(processed_dir / "covariance_pairs_train.npz")
    logger.info(
        "Training pairs: %d, condition_scaled shape %s",
        len(train_ds["condition_scaled"]),
        train_ds["condition_scaled"].shape,
    )

    total = len(SCHEDULE_GRID) * len(T_GRID)
    counter = 0

    for schedule_type in SCHEDULE_GRID:
        for T in T_GRID:
            counter += 1
            logger.info(
                "\n[%d/%d] Training denoiser: schedule=%s, T=%d",
                counter, total, schedule_type, T,
            )

            ckpt_name = f"denoise_schedule-{schedule_type}_T-{T}_seed-{seed}.pt"
            ckpt_path = model_dir / ckpt_name

            if ckpt_path.exists():
                logger.info("  Checkpoint already exists: %s – SKIPPING.", ckpt_path)
                continue

            set_global_seed(seed)

            model, history = train_unconditional_ddpm(
                train_dataset=train_ds,
                schedule_type=schedule_type,
                T=T,
                beta_min=cfg.training["beta_min"],
                beta_max=cfg.training["beta_max"],
                hidden_dim=cfg.model["hidden_dim"],
                num_hidden=cfg.model["num_hidden_layers"],
                time_embed_dim=cfg.model["time_embedding_dim"],
                dropout=cfg.model["dropout"],
                epochs=cfg.training["epochs"],
                batch_size=cfg.training["batch_size"],
                learning_rate=cfg.training["learning_rate"],
                weight_decay=cfg.training["weight_decay"],
                seed=seed,
                save_dir=None,
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": "unconditional_denoiser",
                    "schedule_type": schedule_type,
                    "T": T,
                    "seed": seed,
                    "hidden_dim": cfg.model["hidden_dim"],
                    "num_hidden": cfg.model["num_hidden_layers"],
                    "time_embed_dim": cfg.model["time_embedding_dim"],
                    "dropout": cfg.model["dropout"],
                    "epochs_trained": cfg.training["epochs"],
                },
                ckpt_path,
            )
            log_path = log_dir / f"denoise_schedule-{schedule_type}_T-{T}_seed-{seed}.csv"
            history.to_csv(log_path, index=False)
            logger.info("  Saved denoiser checkpoint: %s", ckpt_path)

    logger.info("\nStep 4B complete – trained %d denoising model(s).", counter)


if __name__ == "__main__":
    main()
