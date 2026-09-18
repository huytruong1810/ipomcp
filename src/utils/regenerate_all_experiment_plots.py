import argparse
import glob
import os

import pandas as pd

from core.logger import get_logger
from utils.paper_plots import generate_paper_plots
from utils.plot_metadata import agent_labels as resolve_agent_labels
from utils.plotting import plot_all_metrics

logger = get_logger("RegeneratePlots")


def regenerate_plots_for_directory(target_dir: str):
    logger.info(f"Scanning directory for batch_results.csv files: {target_dir}")
    csv_files = glob.glob(os.path.join(target_dir, "**", "batch_results.csv"), recursive=True)
    logger.info(f"Found {len(csv_files)} batch_results.csv datasets to render.")

    success_count = 0
    failures = []
    for idx, csv_path in enumerate(sorted(csv_files), 1):
        cell_dir = os.path.dirname(csv_path)
        folder_name = os.path.basename(cell_dir)
        logger.info(f"[{idx}/{len(csv_files)}] Processing: {folder_name}...")

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning(f"  Empty dataframe in {csv_path}. Skipping.")
                continue

            agent_labels = resolve_agent_labels(df)
            title_prefix = folder_name.replace("_", " ")

            # Generate interactive Plotly HTML widgets
            plot_all_metrics(
                df, agent_labels=agent_labels, title_prefix=title_prefix, save_dir=cell_dir
            )

            # Generate publication vector PDFs
            generate_paper_plots(
                csv_path=csv_path,
                output_dir=cell_dir,
                agent_labels=agent_labels,
                title_prefix=title_prefix,
            )
            success_count += 1
        except Exception as e:
            logger.exception("Failed generating plots for %s", cell_dir)
            failures.append((cell_dir, str(e)))

    logger.info(
        f"Plot regeneration complete. Successfully generated plots in {success_count}/{len(csv_files)} directories."
    )

    if failures:
        raise RuntimeError(f"Plot generation failed: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate all interactive HTML and publication vector plots from batch CSVs"
    )
    parser.add_argument(
        "--dir", type=str, required=True, help="Target experiment results directory"
    )
    args = parser.parse_args()

    regenerate_plots_for_directory(args.dir)
