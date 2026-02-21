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
    color_map = {False: "tab:blue", True: "tab:red"}
    label_used = {False: False, True: False}

    for idx, record in enumerate(records):
        color = color_map[record.is_pruned]
        label = None if label_used[record.is_pruned] else ("Pruned" if record.is_pruned else "Baseline")
        label_used[record.is_pruned] = True
        ax.scatter(record.pruned_percentile, record.test_accuracy, color=color, label=label, zorder=3)

    # Draw faint connectors so each baseline/pruned pair is easy to visually match.
    try:
        for baseline, pruned in pairwise(records):
            ax.plot(
                [baseline.pruned_percentile, pruned.pruned_percentile],
                [baseline.test_accuracy, pruned.test_accuracy],
                color="gray",
                linewidth=0.75,
                linestyle="--",
                alpha=0.4,
                zorder=2,
            )
    except ValueError:
        pass  # Ignore if the CSV currently has an odd row count.

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
