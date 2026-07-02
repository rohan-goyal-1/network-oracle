from __future__ import annotations

import os

import pandas as pd


def save_results(df: pd.DataFrame, out: str, name: str) -> None:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")
