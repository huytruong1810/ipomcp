"""Aggregate the L1 POMCP-vs-exact rows by profile, budget and horizon."""

import json
import sys
from pathlib import Path

import pandas as pd

rows = pd.DataFrame(json.loads(Path(sys.argv[1]).read_text()))
summary = (
    rows.groupby(["profile", "budget", "horizon"])
    .agg(
        cases=("optimal", "size"),
        optimal_rate=("optimal", "mean"),
        mean_loss=("loss", "mean"),
        max_loss=("loss", "max"),
        mean_max_q_error=("max_abs_q_error", "mean"),
        max_q_error=("max_abs_q_error", "max"),
        mean_plan_s=("plan_seconds", "mean"),
        exact_solve_s=("exact_solve_seconds", "mean"),
    )
    .reset_index()
)
summary.to_csv(Path(sys.argv[1]).with_name("summary.csv"), index=False)
print(summary.round(4).to_string(index=False))
