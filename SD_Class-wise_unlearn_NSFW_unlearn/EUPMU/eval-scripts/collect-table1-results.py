import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_CLASS_IDS = list(range(10))
METHOD_LABELS = {
    "eu": "EUPMU",
    "omd_tch": "OMD-TCH",
    "omd_tch_pgd": "OMD-TCH-PGD",
}


def parse_class_ids(raw: str) -> List[int]:
    return [int(part) for part in raw.split() if part.strip()]


def class_name_map(prompts_path: Path) -> Dict[int, str]:
    names = {}
    with prompts_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_name = row["class"].replace(" ", "")
            names[int(row["case_number"])] = class_name
    return names


def latest_model_dir(method_root: Path) -> Optional[Path]:
    candidates = [path for path in method_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def classification_csv_path(model_dir: Path) -> Optional[Path]:
    matches = list(model_dir.glob("*_classification.csv"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def compute_paper_ua(csv_path: Path, class_to_forget: int) -> float:
    total = 0
    correct = 0
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["case_number"]) != class_to_forget:
                continue
            total += 1
            if row["category_top1"] == row["class"]:
                correct += 1
    if total == 0:
        raise ValueError(f"No rows for class {class_to_forget} in {csv_path}")
    return 100.0 * correct / total


def read_fid(fid_path: Path) -> float:
    return float(fid_path.read_text().strip())


def collect_method_result(
    evaluation_root: Path, method: str, class_to_forget: int
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    method_root = evaluation_root / f"class_{class_to_forget}" / method
    if not method_root.exists():
        return None, None, f"missing evaluation folder: {method_root}"

    model_dir = latest_model_dir(method_root)
    if model_dir is None:
        return None, None, f"missing model directory under: {method_root}"

    csv_path = classification_csv_path(model_dir)
    fid_path = model_dir / "fid_result.txt"

    ua = None
    fid = None
    warnings = []

    if csv_path is None:
        warnings.append(f"missing classification csv in: {model_dir}")
    else:
        ua = compute_paper_ua(csv_path, class_to_forget)

    if not fid_path.exists():
        warnings.append(f"missing fid_result.txt in: {model_dir}")
    else:
        fid = read_fid(fid_path)

    warning = "; ".join(warnings) if warnings else None
    return ua, fid, warning


def average(values: List[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def format_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def write_csv(rows: List[Dict[str, str]], output_path: Path, fieldnames: List[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], output_path: Path, fieldnames: List[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(fieldnames) + " |\n"
    divider = "| " + " | ".join(["---"] * len(fieldnames)) + " |\n"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fieldnames) + " |\n")
    output_path.write_text("".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="collect-table1-results",
        description="Collect paper-style Table 1 UA/FID metrics into CSV and Markdown.",
    )
    parser.add_argument(
        "--evaluation_root",
        type=Path,
        default=Path("evaluation_folder/paper_table1_eupmu_omd_tch"),
        help="Root folder containing class_{id}/{method}/{model_dir} evaluation outputs.",
    )
    parser.add_argument(
        "--prompts_path",
        type=Path,
        default=Path("prompts/imagenette.csv"),
        help="Prompts CSV used to map class ids to paper row names.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="eu omd_tch",
        help="Space-separated method ids to include.",
    )
    parser.add_argument(
        "--class_ids",
        type=str,
        default="0 1 2 3 4 5 6 7 8 9",
        help="Space-separated forgotten class ids to include.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("evaluation_folder/paper_table1_eupmu_omd_tch/table1_eupmu_omd_tch.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=Path("evaluation_folder/paper_table1_eupmu_omd_tch/table1_eupmu_omd_tch.md"),
        help="Output Markdown path.",
    )
    args = parser.parse_args()

    methods = [method for method in args.methods.split() if method.strip()]
    class_ids = parse_class_ids(args.class_ids) if args.class_ids else DEFAULT_CLASS_IDS
    name_map = class_name_map(args.prompts_path)

    fieldnames = ["Forget.Class"]
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        fieldnames.extend([f"{label} UA", f"{label} FID"])

    rows: List[Dict[str, str]] = []
    per_method_ua: Dict[str, List[Optional[float]]] = {method: [] for method in methods}
    per_method_fid: Dict[str, List[Optional[float]]] = {method: [] for method in methods}

    warnings: List[str] = []
    for class_id in class_ids:
        row = {"Forget.Class": name_map.get(class_id, str(class_id))}
        for method in methods:
            ua, fid, warning = collect_method_result(args.evaluation_root, method, class_id)
            if warning is not None:
                warnings.append(f"class={class_id} method={method}: {warning}")
            per_method_ua[method].append(ua)
            per_method_fid[method].append(fid)
            label = METHOD_LABELS.get(method, method)
            row[f"{label} UA"] = format_value(ua)
            row[f"{label} FID"] = format_value(fid)
        rows.append(row)

    average_row = {"Forget.Class": "Average"}
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        average_row[f"{label} UA"] = format_value(average(per_method_ua[method]))
        average_row[f"{label} FID"] = format_value(average(per_method_fid[method]))
    rows.append(average_row)

    write_csv(rows, args.output_csv, fieldnames)
    write_markdown(rows, args.output_md, fieldnames)

    print("=======================================")
    print(f"Saved CSV table to: {args.output_csv}")
    print(f"Saved Markdown table to: {args.output_md}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print("=======================================")
