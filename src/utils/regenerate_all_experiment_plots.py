# Absolute Path: <project_root>/utils/regenerate_all_experiment_plots.py

import os
import sys
import re
import argparse
import glob
import pandas as pd
from utils.plotting import plot_all_metrics
from utils.paper_plots import generate_paper_plots
from core.logger import get_logger

logger = get_logger("RegeneratePlots")


def _extract_labels_and_title(folder_name: str, df: pd.DataFrame) -> tuple[dict, str]:
    """Extracts informative agent labels and title prefix from directory name and dataframe."""
    clean_title = folder_name.replace("Cell_", "").replace("cond_", "").replace("_", " ").replace("pct", "%")
    
    # Check if level columns exist in df
    lvl_i = None
    lvl_j = None
    if "level_i" in df.columns and not df["level_i"].dropna().empty:
        try:
            lvl_i = int(df["level_i"].iloc[0])
        except Exception:
            pass
    if "level_j" in df.columns and not df["level_j"].dropna().empty:
        try:
            lvl_j = int(df["level_j"].iloc[0])
        except Exception:
            pass

    # If not in columns, parse from folder name
    if lvl_i is None or lvl_j is None:
        match = re.search(r'L(\d+)_vs_L(\d+)', folder_name)
        if match:
            lvl_i = int(match.group(1))
            lvl_j = int(match.group(2))

    label_i = f"Agent I (Level-{lvl_i})" if lvl_i is not None else "Agent I"
    label_j = f"Agent J (Level-{lvl_j})" if lvl_j is not None else "Agent J"
    
    return {"i": label_i, "j": label_j}, clean_title


def regenerate_plots_for_directory(target_dir: str):
    logger.info(f"Scanning directory for batch_results.csv files: {target_dir}")
    csv_files = glob.glob(os.path.join(target_dir, "**", "batch_results.csv"), recursive=True)
    logger.info(f"Found {len(csv_files)} batch_results.csv datasets to render.")

    success_count = 0
    for idx, csv_path in enumerate(sorted(csv_files), 1):
        cell_dir = os.path.dirname(csv_path)
        folder_name = os.path.basename(cell_dir)
        logger.info(f"[{idx}/{len(csv_files)}] Processing: {folder_name}...")

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning(f"  Empty dataframe in {csv_path}. Skipping.")
                continue

            agent_labels, title_prefix = _extract_labels_and_title(folder_name, df)

            # Generate interactive Plotly HTML widgets
            plot_all_metrics(
                df,
                agent_labels=agent_labels,
                title_prefix=title_prefix,
                save_dir=cell_dir
            )

            # Generate publication vector PDFs
            generate_paper_plots(
                csv_path=csv_path,
                output_dir=cell_dir,
                agent_labels=agent_labels,
                title_prefix=title_prefix
            )
            success_count += 1
        except Exception as e:
            logger.error(f"  Failed generating plots for {cell_dir}: {e}")

    logger.info(f"Plot regeneration complete. Successfully generated plots in {success_count}/{len(csv_files)} directories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate all interactive HTML and publication vector plots from batch CSVs")
    parser.add_argument("--dir", type=str, required=True, help="Target experiment results directory")
    args = parser.parse_args()

    regenerate_plots_for_directory(args.dir)
