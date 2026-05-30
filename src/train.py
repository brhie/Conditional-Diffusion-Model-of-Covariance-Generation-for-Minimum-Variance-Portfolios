"""
train.py
--------
Train a single conditional DDPM model.

Spec §18 – fixed 200-epoch training, no early stopping on validation GMV vol.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import CovariancePairDataset, CovarianceVecDataset
from .diffusion import DDPMScheduler, ddpm_training_step, q_sample
from .model import build_denoising_model, build_model
from .utils import get_device

logger = logging.getLogger(__name__)


def train_one_conditional_ddpm(
    train_dataset: dict,
    schedule_type: str,
    T: int,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
    hidden_dim: int = 128,
    num_hidden: int = 3,
    time_embed_dim: int = 32,
    dropout: float = 0.0,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device: Optional[str] = None,
    save_dir: Optional[str | Path] = None,
    condition_dim: Optional[int] = None,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    """
    Train one conditional DDPM and return (model, training_history).

    Parameters
    ----------
    train_dataset : dict from load_dataset or build_covariance_dataset
        Must contain 'condition_scaled' and 'target_scaled' keys.
    schedule_type : 'linear' (active configuration); 'quadratic' and 'logarithmic' also supported
    T : int
        Number of diffusion steps. Active grid: {400, 800, 1200, 2000}.
    condition_dim : int or None
        Dimension of the conditioning vector (55 + K macro features).
        If None, inferred from train_dataset['condition_scaled'].shape[1].
    save_dir : if not None, save checkpoint and loss history there.

    Returns
    -------
    (model, history_df) where history_df has columns [epoch, train_loss, elapsed_s]
    """
    # ---- Setup --------------------------------------------------------
    if device is None:
        device = get_device()
    device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # CUDA-specific speed-ups
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info(
        "Training DDPM: schedule=%s, T=%d, device=%s",
        schedule_type, T, device,
    )

    # ---- Model --------------------------------------------------------
    # Infer condition_dim from the dataset if not explicitly provided.
    # condition_dim = 55 (covariance vech) + K (macro features).
    if condition_dim is None:
        condition_dim = int(train_dataset["condition_scaled"].shape[1])
    logger.info("condition_dim = %d (55 covariance vech + %d macro features)",
                condition_dim, condition_dim - 55)

    model = build_model(
        noised_dim=55,
        condition_dim=condition_dim,
        time_embed_dim=time_embed_dim,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        dropout=dropout,
    ).to(device)

    # ---- Scheduler ----------------------------------------------------
    scheduler = DDPMScheduler(
        schedule_type=schedule_type,
        T=T,
        beta_min=beta_min,
        beta_max=beta_max,
        device=device,
    )

    # ---- DataLoader ---------------------------------------------------
    pytorch_dataset = CovariancePairDataset(train_dataset)
    # pin_memory speeds up CPU→GPU transfers on CUDA only.
    # num_workers=0 avoids broken-pipe errors in Jupyter on macOS (spawn context).
    use_pin = device.type == "cuda"
    num_workers = 2 if device.type == "cuda" else 0
    loader = DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

    # ---- Optimizer ----------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ---- Progress bar (tqdm.auto works in both terminal and Jupyter) ------
    try:
        from tqdm.auto import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    bar_desc = f"DDPM {schedule_type} T={T}"

    # ---- Training loop ------------------------------------------------
    history = []
    t0 = time.time()
    best_loss = float("inf")

    epoch_iter = (
        _tqdm(range(1, epochs + 1), desc=bar_desc, unit="ep", dynamic_ncols=True)
        if _has_tqdm else range(1, epochs + 1)
    )

    for epoch in epoch_iter:
        model.train()
        epoch_losses = []

        for cond_batch, tgt_batch in loader:
            cond_batch = cond_batch.to(device, non_blocking=use_pin)
            tgt_batch  = tgt_batch.to(device,  non_blocking=use_pin)

            optimizer.zero_grad()
            loss = ddpm_training_step(model, tgt_batch, cond_batch, scheduler)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        elapsed   = time.time() - t0
        best_loss = min(best_loss, mean_loss)
        history.append(
            {"epoch": epoch, "train_loss": mean_loss, "elapsed_s": elapsed}
        )

        # ── Live tqdm postfix ────────────────────────────────────────────
        if _has_tqdm:
            eta_s = elapsed / epoch * (epochs - epoch)
            epoch_iter.set_postfix(
                loss=f"{mean_loss:.5f}",
                best=f"{best_loss:.5f}",
                eta=f"{int(eta_s//60)}m{int(eta_s%60):02d}s",
            )

        # ── Periodic logger output (every 20 epochs, for log files) ─────
        if epoch % 20 == 0 or epoch == 1:
            eta_s = elapsed / epoch * (epochs - epoch)
            logger.info(
                "  Epoch %3d/%d | loss %.6f | best %.6f | %.0fs | ETA %dm%02ds",
                epoch, epochs, mean_loss, best_loss,
                elapsed, int(eta_s // 60), int(eta_s % 60),
            )

    # ── Final summary ────────────────────────────────────────────────────
    total_time = time.time() - t0
    logger.info(
        "Done: schedule=%s T=%d | final_loss=%.6f | best_loss=%.6f | %.0fs total",
        schedule_type, T, history[-1]["train_loss"], best_loss, total_time,
    )

    history_df = pd.DataFrame(history)

    # ---- Save ---------------------------------------------------------
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"ddpm_schedule-{schedule_type}_T-{T}_seed-{seed}"
        ckpt_path = save_dir / f"{model_name}.pt"
        log_path = save_dir / f"{model_name}.csv"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "schedule_type": schedule_type,
                "T": T,
                "seed": seed,
                "hidden_dim": hidden_dim,
                "num_hidden": num_hidden,
                "time_embed_dim": time_embed_dim,
                "dropout": dropout,
                "epochs_trained": epochs,
                "condition_dim": condition_dim,
            },
            ckpt_path,
        )
        history_df.to_csv(log_path, index=False)
        logger.info(
            "Saved checkpoint: %s  |  history: %s", ckpt_path, log_path
        )

    return model, history_df


# ---------------------------------------------------------------------------
# Train unconditional DDPM denoiser (covariance regularizer)
# ---------------------------------------------------------------------------

def train_unconditional_ddpm(
    train_dataset: dict,
    schedule_type: str,
    T: int,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
    hidden_dim: int = 128,
    num_hidden: int = 3,
    time_embed_dim: int = 32,
    dropout: float = 0.0,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device: Optional[str] = None,
    save_dir: Optional[str | Path] = None,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    """
    Train an unconditional DDPM denoiser on historical 126-day covariance vectors.

    The model learns P(S_hist) in standardized log-vech space, enabling
    SDEdit-style partial denoising at inference: add mild noise to the observed
    covariance and reverse-denoise back to the learned manifold.

    Training data: condition_scaled vectors from train_dataset
    (i.e., the standardized log-vech of the 126-day historical sample covariance).

    Returns
    -------
    (model, history_df) where history_df has columns [epoch, train_loss, elapsed_s]
    """
    if device is None:
        device = get_device()
    device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info(
        "Training unconditional DDPM denoiser: schedule=%s, T=%d, device=%s",
        schedule_type, T, device,
    )

    model = build_denoising_model(
        noised_dim=55,
        time_embed_dim=time_embed_dim,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        dropout=dropout,
    ).to(device)

    scheduler = DDPMScheduler(
        schedule_type=schedule_type,
        T=T,
        beta_min=beta_min,
        beta_max=beta_max,
        device=device,
    )

    pytorch_dataset = CovarianceVecDataset(train_dataset)
    use_pin = device.type == "cuda"
    num_workers = 2 if device.type == "cuda" else 0
    loader = DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    try:
        from tqdm.auto import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    bar_desc = f"Denoiser {schedule_type} T={T}"

    history = []
    t0 = time.time()
    best_loss = float("inf")

    epoch_iter = (
        _tqdm(range(1, epochs + 1), desc=bar_desc, unit="ep", dynamic_ncols=True)
        if _has_tqdm else range(1, epochs + 1)
    )

    for epoch in epoch_iter:
        model.train()
        epoch_losses = []

        for vec_batch in loader:
            vec_batch = vec_batch.to(device, non_blocking=use_pin)

            batch_size_ = vec_batch.shape[0]
            s = torch.randint(1, scheduler.T + 1, (batch_size_,), device=device)
            y_s, eps = q_sample(vec_batch, s, scheduler)
            eps_hat = model(y_s, s)
            loss = ((eps - eps_hat) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        elapsed = time.time() - t0
        best_loss = min(best_loss, mean_loss)
        history.append({"epoch": epoch, "train_loss": mean_loss, "elapsed_s": elapsed})

        if _has_tqdm:
            eta_s = elapsed / epoch * (epochs - epoch)
            epoch_iter.set_postfix(
                loss=f"{mean_loss:.5f}",
                best=f"{best_loss:.5f}",
                eta=f"{int(eta_s//60)}m{int(eta_s%60):02d}s",
            )

        if epoch % 20 == 0 or epoch == 1:
            eta_s = elapsed / epoch * (epochs - epoch)
            logger.info(
                "  Epoch %3d/%d | loss %.6f | best %.6f | %.0fs | ETA %dm%02ds",
                epoch, epochs, mean_loss, best_loss,
                elapsed, int(eta_s // 60), int(eta_s % 60),
            )

    total_time = time.time() - t0
    logger.info(
        "Done: schedule=%s T=%d | final_loss=%.6f | best_loss=%.6f | %.0fs total",
        schedule_type, T, history[-1]["train_loss"], best_loss, total_time,
    )

    history_df = pd.DataFrame(history)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"denoise_schedule-{schedule_type}_T-{T}_seed-{seed}"
        ckpt_path = save_dir / f"{model_name}.pt"
        log_path = save_dir / f"{model_name}.csv"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": "unconditional_denoiser",
                "schedule_type": schedule_type,
                "T": T,
                "seed": seed,
                "hidden_dim": hidden_dim,
                "num_hidden": num_hidden,
                "time_embed_dim": time_embed_dim,
                "dropout": dropout,
                "epochs_trained": epochs,
            },
            ckpt_path,
        )
        history_df.to_csv(log_path, index=False)
        logger.info("Saved denoiser checkpoint: %s  |  history: %s", ckpt_path, log_path)

    return model, history_df


# ---------------------------------------------------------------------------
# Load a saved denoising model
# ---------------------------------------------------------------------------

def load_denoising_model(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Load a previously saved unconditional DDPM denoiser checkpoint.

    Returns
    -------
    (model, meta_dict) where meta_dict contains schedule_type, T, etc.
    """
    if device is None:
        device = get_device()
    device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_denoising_model(
        noised_dim=55,
        time_embed_dim=ckpt.get("time_embed_dim", 32),
        hidden_dim=ckpt.get("hidden_dim", 128),
        num_hidden=ckpt.get("num_hidden", 3),
        dropout=ckpt.get("dropout", 0.0),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    logger.info(
        "Loaded denoising model: schedule=%s, T=%d from %s",
        meta.get("schedule_type"), meta.get("T"), checkpoint_path,
    )
    return model, meta


# ---------------------------------------------------------------------------
# Load a saved model
# ---------------------------------------------------------------------------

def load_trained_model(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Load a previously saved DDPM checkpoint.

    Returns
    -------
    (model, meta_dict)  where meta_dict contains schedule_type, T, etc.
    """
    if device is None:
        device = get_device()
    device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model(
        noised_dim=55,
        condition_dim=ckpt.get("condition_dim", 55),  # 55 + K macro features
        time_embed_dim=ckpt.get("time_embed_dim", 32),
        hidden_dim=ckpt.get("hidden_dim", 128),
        num_hidden=ckpt.get("num_hidden", 3),
        dropout=ckpt.get("dropout", 0.0),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    logger.info(
        "Loaded model: schedule=%s, T=%d from %s",
        meta.get("schedule_type"), meta.get("T"), checkpoint_path,
    )
    return model, meta
