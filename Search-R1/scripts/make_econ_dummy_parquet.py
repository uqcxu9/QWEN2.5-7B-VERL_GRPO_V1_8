#!/usr/bin/env python3
"""Minimal RLHF-style parquet rows for EconAgent + VERL (prompt + econ metadata)."""
import os

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    out_dir = os.path.join(_ROOT, "data", "econ_dummy")
    os.makedirs(out_dir, exist_ok=True)
    row = {
        "prompt": [{"role": "user", "content": "EconAgent GRPO placeholder (real prompts come from env)."}],
        "year_index": 2,
        "state_group_id": "sg0",
        "episode_id": 0,
        "extra_info": {"index": 0},
    }
    train_path = os.path.join(out_dir, "train.parquet")
    val_path = os.path.join(out_dir, "val.parquet")
    pd.DataFrame([row]).to_parquet(train_path)
    pd.DataFrame([row]).to_parquet(val_path)
    print(f"Wrote {train_path} and {val_path}")


if __name__ == "__main__":
    main()
