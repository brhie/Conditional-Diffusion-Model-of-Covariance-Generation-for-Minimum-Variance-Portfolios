"""
00_download_macro_data.py
--------------------------
Download external macro conditioning features from FRED and save to
data/interim/macro_external.parquet.

Features downloaded
-------------------
  log_vix       : log(CBOE VIX) — FRED series VIXCLS
  term_spread   : 10Y − 2Y Treasury yield (pct pts) — FRED DGS10, DGS2
  credit_spread : ICE BofA US Corporate IG OAS (pct pts) — FRED BAMLC0A0CMEY

Research basis
--------------
  VIX           : Engle & Figlewski (2012 J. Derivatives);
                  Bollerslev, Tauchen & Zhou (2009 RFS) variance risk premium
  Term spread   : Estrella & Mishkin (1998 Rev. Econ. Stat.);
                  Wright (2006 J. Business & Econ. Stat.)
  Credit spread : Gilchrist & Zakrajsek (2012 AER) excess bond premium;
                  Collin-Dufresne, Goldstein & Martin (2001 J. Finance)

Requirements
------------
  pip install pandas-datareader

Output
------
  data/interim/macro_external.parquet
  Columns: [log_vix, term_spread, credit_spread]
  Index  : daily date (business days)

Usage
-----
  python scripts/00_download_macro_data.py

This script is OPTIONAL.  If the output file is absent, script 03 will use
only the 4 CRSP-derived macro features (condition_dim = 55 + 4 = 59).
If the file is present, all 7 features are used (condition_dim = 55 + 7 = 62).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger

logger = get_logger("00_download_macro_data", logging.INFO)

# Study period: pull slightly before 2000 to allow for forward-fill at boundary
START = "1998-01-01"
END   = "2025-12-31"


def _try_fred(series_ids: list[str]) -> pd.DataFrame:
    """Download one or more FRED series using pandas-datareader."""
    try:
        import pandas_datareader.data as web
    except ImportError:
        logger.error(
            "pandas-datareader not installed. Run: pip install pandas-datareader"
        )
        raise

    frames = []
    for sid in series_ids:
        logger.info("  Downloading FRED/%s …", sid)
        try:
            s = web.DataReader(sid, "fred", START, END)
            frames.append(s.rename(columns={sid: sid}))
        except Exception as exc:
            logger.warning("  FRED/%s failed: %s", sid, exc)
    if not frames:
        raise RuntimeError("No FRED series could be downloaded.")
    return pd.concat(frames, axis=1)


def main() -> None:
    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "macro_external.parquet"

    logger.info("=" * 60)
    logger.info("STEP 0 – Download external macro features from FRED")
    logger.info("Date range: %s – %s", START, END)
    logger.info("=" * 60)

    # ---- VIX -----------------------------------------------------------
    logger.info("Downloading VIX (VIXCLS) …")
    vix_raw = _try_fred(["VIXCLS"])
    log_vix = np.log(vix_raw["VIXCLS"].clip(lower=0.01))
    log_vix.name = "log_vix"

    # ---- Treasury yields for term spread --------------------------------
    logger.info("Downloading 10Y and 2Y Treasury yields (DGS10, DGS2) …")
    yields = _try_fred(["DGS10", "DGS2"])
    term_spread = (yields["DGS10"] - yields["DGS2"]).rename("term_spread")

    # ---- Investment-Grade corporate OAS ----------------------------------
    logger.info("Downloading IG OAS (BAMLC0A0CMEY) …")
    oas_raw = _try_fred(["BAMLC0A0CMEY"])
    credit_spread = oas_raw["BAMLC0A0CMEY"].rename("credit_spread")

    # ---- Combine, forward-fill, and save ----------------------------------
    df = pd.concat([log_vix, term_spread, credit_spread], axis=1)
    df = df.sort_index()

    # Forward-fill up to 5 business days to cover holidays / weekends
    df = df.ffill(limit=5)

    n_before = df.notna().all(axis=1).sum()
    df = df.dropna(how="all")
    n_after  = df.notna().all(axis=1).sum()

    logger.info(
        "External macro features: %d dates total, %d with all 3 complete.",
        len(df), n_after,
    )

    df.index.name = "date"
    df.to_parquet(out_path)
    logger.info("Saved → %s", out_path)
    logger.info("Columns: %s", list(df.columns))
    logger.info("Date range: %s – %s", df.index.min().date(), df.index.max().date())

    # Quick summary statistics
    print("\nFeature summary:")
    print(df.describe().round(4))


if __name__ == "__main__":
    main()
