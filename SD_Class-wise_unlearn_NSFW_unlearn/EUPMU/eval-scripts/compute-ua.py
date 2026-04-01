import argparse
import csv
import os
import re
from typing import List, Optional


def infer_class_to_forget(path: str) -> Optional[int]:
    patterns = [
        r"class_(\d+)",
        r"class_to_forget[=_-]?(\d+)",
        r"compvis-cl-class_(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
    return None


def resolve_csv_path(folder_path: Optional[str], csv_path: Optional[str]) -> str:
    if csv_path is not None:
        return csv_path
    if folder_path is None:
        raise ValueError("Either --folder_path or --csv_path must be provided.")

    candidates: List[str] = [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.endswith("_classification.csv")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No *_classification.csv files found in folder: {folder_path}"
        )
    if len(candidates) > 1:
        raise ValueError(
            "Multiple classification CSV files found. Please pass --csv_path explicitly."
        )
    return candidates[0]


def compute_table1_ua(csv_path: str, class_to_forget: int) -> dict:
    total = 0
    correct = 0

    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["case_number"]) != class_to_forget:
                continue
            total += 1
            if row["category_top1"] == row["class"]:
                correct += 1

    if total == 0:
        raise ValueError(
            f"No rows for forgotten class {class_to_forget} found in {csv_path}."
        )

    paper_ua = 100.0 * correct / total
    unlearn_acc = 100.0 - paper_ua
    return {
        "paper_ua": paper_ua,
        "unlearn_acc": unlearn_acc,
        "correct": correct,
        "total": total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compute-ua",
        description="Compute the Table 1 style UA metric from a saved classification CSV.",
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default=None,
        help="Folder containing a single *_classification.csv file.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to a saved *_classification.csv file.",
    )
    parser.add_argument(
        "--class_to_forget",
        type=int,
        default=None,
        help="Forgotten class id. If omitted, inferred from the path when possible.",
    )
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.folder_path, args.csv_path)
    class_to_forget = args.class_to_forget
    if class_to_forget is None:
        class_to_forget = infer_class_to_forget(csv_path)
    if class_to_forget is None and args.folder_path is not None:
        class_to_forget = infer_class_to_forget(args.folder_path)
    if class_to_forget is None:
        raise ValueError(
            "Could not infer class_to_forget from the path. Please pass --class_to_forget explicitly."
        )

    metrics = compute_table1_ua(csv_path, class_to_forget)
    print("=======================================")
    print(f"class_to_forget: {class_to_forget}")
    print(f"classification_csv: {csv_path}")
    print(f"samples_for_class: {metrics['total']}")
    print(f"correct_top1_predictions: {metrics['correct']}")
    print(f"paper_ua: {metrics['paper_ua']:.2f}")
    print(f"unlearn_acc: {metrics['unlearn_acc']:.2f}")
    print("=======================================")
