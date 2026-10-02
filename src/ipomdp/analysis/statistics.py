"""Statistical analysis and significance testing utilities for benchmark trials."""

from __future__ import annotations

import json
import os
from itertools import combinations
from typing import Any, Dict

import pandas as pd
from scipy import stats


def compute_statistical_significance(
    combined_df: pd.DataFrame,
    out_dir: str,
    output_filename: str = "statistical_significance_tests.json",
) -> Dict[str, Any]:
    """Paired seed comparisons with Holm-adjusted two-sided t-test p-values.

    Conditions must share every trial at the final requested step. Pairing by row
    position or truncating arrays can compare different seeds and conceal failed
    trials. These exploratory tests do not establish convergence or equivalence.

    Args:
        combined_df: Master DataFrame containing at least ['trial', 'step', 'condition', 'cum_reward_i'].
        out_dir: Directory to save the exported JSON results.
        output_filename: Name of the output JSON file.

    Returns:
        Dictionary of pairwise test statistics and Holm-adjusted p-values.
    """
    final = combined_df[combined_df.step == combined_df.step.max()]
    table = final.pivot(index="trial", columns="condition", values="cum_reward_i")
    if table.isna().any().any() or len(table) < 2:
        raise ValueError("Paired comparisons require a complete common panel of >=2 trials")

    results: Dict[str, Any] = {}
    for c1, c2 in combinations(table.columns, 2):
        differences = table[c1] - table[c2]
        if differences.eq(0).all():
            t_stat: float | None = 0.0
            p_t: float = 1.0
            p_w: float = 1.0
        elif differences.std() == 0:
            t_stat = None
            p_t = 0.0
            p_w = float(stats.wilcoxon(differences).pvalue)
        else:
            test = stats.ttest_1samp(differences, 0)
            t_stat = float(test.statistic)
            p_t = float(test.pvalue)
            p_w = float(stats.wilcoxon(differences).pvalue)

        results[f"{c1} vs {c2}"] = dict(
            n_pairs=len(table),
            mean_diff=float(differences.mean()),
            c1_mean=float(table[c1].mean()),
            c2_mean=float(table[c2].mean()),
            paired_t_stat=t_stat,
            paired_t_pval=p_t,
            wilcoxon_pval=p_w,
        )

    running = 0.0
    ordered = sorted(results.values(), key=lambda row: row["paired_t_pval"])
    for rank, row in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * row["paired_t_pval"]))
        row["holm_pval"] = running
        row["significant_at_05"] = running < 0.05

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    with open(out_path, "w") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)

    return results
