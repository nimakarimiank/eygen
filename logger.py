import csv
from pathlib import Path
from typing import Iterable, Union

CSV_HEADERS = [
    "isPruned",
    "no_layers",
    "no_nodes",
    "no_epochs",
    "no_batches",
    "pruned_percentile",
    "train_accuracy",
    "test_accuracy",
]


def format_nodes(layer_nodes: Iterable[int], separator: str = "-") -> str:
    """Return a compact string describing the layer widths."""
    return separator.join(str(node) for node in layer_nodes)


def log_run(
    *,
    csv_path: Union[str, Path] = "training_runs.csv",
    is_pruned: bool,
    no_layers: int,
    no_nodes: str,
    no_epochs: int,
    no_batches: int,
    pruned_percentile: float,
    train_accuracy: float,
    test_accuracy: float,
) -> None:
    """Append a training/evaluation summary row to the CSV log."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            dict(
                isPruned=int(is_pruned),
                no_layers=no_layers,
                no_nodes=no_nodes,
                no_epochs=no_epochs,
                no_batches=no_batches,
                pruned_percentile=pruned_percentile,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
            )
        )
