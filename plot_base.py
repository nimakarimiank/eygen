"""Utility helpers for visualizing training_runs.csv outputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

@dataclass(frozen=True)
class RunRecord:
    """Container describing a single row in training_runs.csv."""

    is_pruned: bool
    pruned_percentile: float
    test_accuracy: float
    no_nodes: float


def load_training_runs(csv_path: Path) -> List[RunRecord]:
    """Parse the CSV file into RunRecord entries."""
    records: List[RunRecord] = []
    with csv_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            records.append(
                RunRecord(
                    is_pruned=bool(int(row["isPruned"])),
                    pruned_percentile=float(row["pruned_percentile"]),
                    test_accuracy=float(row["test_accuracy"]),
                    no_nodes=float(row["no_nodes"]),
                )
            )
    return records


def pairwise(records: Sequence[RunRecord]) -> Iterable[Tuple[RunRecord, RunRecord]]:
    """Yield consecutive record pairs (baseline, pruned)."""
    if len(records) % 2 != 0:
        raise ValueError("Expected an even number of rows (baseline/pruned pairs).")
    for idx in range(0, len(records), 2):
        yield records[idx], records[idx + 1]


def plot_test_accuracy(records: Sequence[RunRecord], *, output_path: Path | None = None, show: bool = True) -> None:
    """Plot test accuracy against prune percentile with color-coded pairs."""
    if not records:
        raise ValueError("No records were found in the CSV file.")

    fig, ax = plt.subplots(figsize=(10, 5))

    baseline_x = [record.no_nodes for record in records if not record.is_pruned]
    baseline_y = [record.test_accuracy for record in records if not record.is_pruned]
    # pruned_x = [record.pruned_percentile for record in records if record.is_pruned]
    # pruned_y = [record.test_accuracy for record in records if record.is_pruned]

    # if not baseline_x or not pruned_x:
    #     raise ValueError("CSV must contain both baseline and pruned rows.")

    ax.plot(baseline_x, baseline_y, color="tab:blue", label="Baseline", marker="o")
    # ax.plot(pruned_x, pruned_y, color="tab:red", label="Pruned", marker="*")

    ax.set_title("Test Accuracy vs. Prune Percentile")
    ax.set_xlabel("Prune Percentile")
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training_runs.csv metrics")
    parser.add_argument("--csv", type=Path, default=Path("training_runs.csv"), help="Path to training_runs.csv")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the plot as an image")
    parser.add_argument("--no-show", action="store_true", help="Skip showing the interactive window (useful for CI)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_training_runs(args.csv)
    plot_test_accuracy(records, output_path=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
