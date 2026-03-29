#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DISPLAY_NAMES = {
    "retrain": "Retrain",
    "FT": "FT",
    "GA": "GA",
    "wfisher": "IU",
    "famo": "FAMO",
    "igs": "UNGrad",
    "FT_prune": "l1-sparse",
    "RL": "Linearization",
    "eu": "EUPMU",
    "eu_fast": "EUPMU-fast",
    "gdr_gma": "GDR-GMA",
    "chebyshev": "Chebyshev",
    "omd_tch": "OMD-TCH",
    "omd_tch_eg": "AFLeg",
    "omd_tch_pgd": "OMD-TCH-PGD",
    "ada_omd_tch_eg": "AdaAFLeg",
    "RL_proximal": "SalUn-soft",
}

DESCRIPTION_NAMES = {
    "retrain": "Retrain: Full retraining after removing the forget set.",
    "RL": "Linearization: Random-label unlearning baseline on the forget set.",
    "FT": "FT: Plain fine-tuning baseline without special unlearning machinery.",
    "GA": "GA: Gradient-ascent baseline that pushes against the forget objective.",
    "wfisher": "IU: Weighted-Fisher influence-based unlearning baseline.",
    "famo": "FAMO: Efficient MGDA-style multi-objective weighting baseline.",
    "igs": "UNGrad: Explicit unilateral gradient-surgery baseline.",
    "FT_prune": "l1-sparse: Sparse fine-tuning / pruning-based unlearning baseline.",
    "gdr_gma": "GDR-GMA: Gradient surgery with direction rectification and magnitude adjustment.",
    "eu": "EUPMU: Efficient implicit unilateral gradient surgery.",
    "eu_fast": "EUPMU-fast: Faster approximation to EUPMU without extra retain recomputation.",
    "chebyshev": "Chebyshev: Augmented Tchebycheff scalarization baseline for retain/forget MOO.",
    "omd_tch": "OMD-TCH: Online mirror-descent Tchebycheff method with adaptive task reweighting.",
    "omd_tch_eg": "AFLeg: Exponentiated-gradient OMD-TCH variant.",
    "omd_tch_pgd": "OMD-TCH-PGD: Projected-gradient OMD-TCH variant.",
    "ada_omd_tch_eg": "AdaAFLeg: Adaptive exponentiated-gradient OMD-TCH variant with marked-model aggregation.",
    "RL_proximal": "SalUn-soft: Soft-thresholding SalUn-style proximal baseline.",
}

CANONICAL_METHOD_IDS = {
    "afleg": "omd_tch",
    "omd_tch_eg": "omd_tch",
    "afl": "omd_tch_pgd",
    "ada_afleg": "ada_omd_tch_eg",
}


def canonical_method_id(method_id: str) -> str:
    if "/" not in method_id:
        return CANONICAL_METHOD_IDS.get(method_id, method_id)
    base, rest = method_id.split("/", 1)
    base = CANONICAL_METHOD_IDS.get(base, base)
    return f"{base}/{rest}"





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan classification experiment outputs and emit a LaTeX table in the style of Table 4."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/resnet18/cifar10/forget_10.0%"),
        help="Root directory that contains method subdirectories such as retrain/ and RL/.",
    )
    parser.add_argument(
        "--accuracy-key",
        default="accuracy",
        choices=["accuracy", "avg_accuracy", "adaptive_accuracy"],
        help=(
            "Which accuracy block to use from evaluation_result.json. "
            "Use avg_accuracy for OMD-TCH averaged-iterate results or adaptive_accuracy for AdaOMD-TCH marked-model results."
        ),
    )
    parser.add_argument(
        "--mia-key",
        default="confidence",
        choices=["correctness", "confidence", "entropy", "m_entropy", "prob"],
        help="Which SVC_MIA_forget_efficacy submetric to use for the MIA column.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional whitelist of method ids to include, e.g. retrain eu chebyshev omd_tch. "
            "Method ids are inferred from directory names."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the LaTeX table. If omitted, print to stdout.",
    )
    parser.add_argument(
        "--caption",
        default="Table 4 style results on CIFAR-10 class-wise forgetting (10\\%).",
        help="Base LaTeX caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:table4_like",
        help="LaTeX label.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimals to print.",
    )
    parser.add_argument(
        "--no-descriptions-in-caption",
        action="store_true",
        help="Do not append one-line method definitions to the caption.",
    )
    return parser.parse_args()


def format_value(value: Optional[float], decimals: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def infer_method_id(json_path: Path, root: Path) -> str:
    rel = json_path.relative_to(root)
    parts = rel.parts
    if len(parts) < 2:
        return parts[0] if parts else json_path.stem

    top = parts[0]
    if top == "RL":
        if len(parts) >= 4 and parts[1] != "None" and parts[2] != "None":
            return f"{parts[1]}/{parts[2]}"
        if len(parts) >= 3 and parts[1] != "None":
            return parts[1]
    return top




def format_display_name(method_id: str) -> str:
    if "/" not in method_id:
        return DISPLAY_NAMES.get(method_id, method_id)
    base, tag = method_id.split("/", 1)
    base_name = DISPLAY_NAMES.get(base, base)
    if "eta_" in tag:
        eta = tag.split("eta_", 1)[1].replace("p", ".")
        return f"{base_name} ({eta})"
    return f"{base_name} [{tag}]"


def load_row(json_path: Path, root: Path, accuracy_key: str, mia_key: str) -> Optional[Dict[str, object]]:
    data = json.loads(json_path.read_text())
    accuracy = data.get(accuracy_key)
    if not isinstance(accuracy, dict):
        return None

    mia = data.get("SVC_MIA_forget_efficacy", {})
    if not isinstance(mia, dict):
        mia = {}

    ua = accuracy.get("forget")
    ra = accuracy.get("retain")
    ta = accuracy.get("test")
    mia_value = mia.get(mia_key)

    numeric_values = [ua, ra, ta, mia_value]
    avg_score = None
    if all(isinstance(v, (int, float)) for v in numeric_values):
        avg_score = sum(float(v) for v in numeric_values) / 4.0

    method_id = canonical_method_id(infer_method_id(json_path, root))
    return {
        "method_id": method_id,
        "display_name": format_display_name(method_id),
        "json_path": json_path,
        "ua": ua,
        "ra": ra,
        "ta": ta,
        "mia": mia_value,
        "avg_score": avg_score,
    }


def collect_rows(root: Path, accuracy_key: str, mia_key: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for json_path in sorted(root.glob("**/evaluation_result.json")):
        row = load_row(json_path, root, accuracy_key, mia_key)
        if row is not None:
            rows.append(row)
    return rows


def dedupe_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    deduped: Dict[str, Dict[str, object]] = {}
    for row in rows:
        method_id = canonical_method_id(str(row["method_id"]))
        row["method_id"] = method_id
        row["display_name"] = format_display_name(method_id)
        current = deduped.get(method_id)
        if current is None:
            deduped[method_id] = row
            continue
        current_depth = len(Path(str(current["json_path"])).parts)
        row_depth = len(Path(str(row["json_path"])).parts)
        if row_depth >= current_depth:
            deduped[method_id] = row
    return list(deduped.values())


def method_sort_key(row: Dict[str, object]) -> Tuple[int, str]:
    order = {
        "retrain": 0,
        "RL": 1,
        "FT": 2,
        "GA": 3,
        "wfisher": 4,
        "FT_prune": 5,
        "gdr_gma": 6,
        "eu": 7,
        "eu_fast": 8,
        "chebyshev": 9,
        "omd_tch": 10,
        "omd_tch_eg": 11,
        "omd_tch_pgd": 12,
        "ada_omd_tch_eg": 13,
    }
    method_id = str(row["method_id"])
    base_method = method_id.split('/')[0]
    return (order.get(base_method, 999), str(row["display_name"]))


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def build_caption(base_caption: str, rows: List[Dict[str, object]], include_descriptions: bool) -> str:
    if not include_descriptions:
        return base_caption
    description_parts = []
    seen = set()
    for row in sorted(rows, key=method_sort_key):
        method_id = str(row["method_id"])
        if method_id in seen:
            continue
        seen.add(method_id)
        if method_id in DESCRIPTION_NAMES:
            description_parts.append(DESCRIPTION_NAMES[method_id])
    if not description_parts:
        return base_caption
    return base_caption + " " + " ".join(description_parts)


def build_table(
    rows: List[Dict[str, object]],
    caption: str,
    label: str,
    decimals: int,
    mia_key: str,
    accuracy_key: str,
    include_descriptions: bool,
) -> str:
    full_caption = latex_escape(build_caption(caption, rows, include_descriptions))
    header = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        f"\\caption{{{full_caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        f"Method & UA & RA & TA & MIA ({mia_key}) & Avg. score " + r"\\",
        r"\midrule",
    ]

    body = []
    for row in sorted(rows, key=method_sort_key):
        line = (
            f"{row['display_name']} & "
            f"{format_value(row.get('ua'), decimals)} & "
            f"{format_value(row.get('ra'), decimals)} & "
            f"{format_value(row.get('ta'), decimals)} & "
            f"{format_value(row.get('mia'), decimals)} & "
            f"{format_value(row.get('avg_score'), decimals)} " + r"\\"
        )
        body.append(line)

    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        f"% accuracy_key={accuracy_key}, mia_key={mia_key}",
        r"\end{table}",
    ]
    return "\n".join(header + body + footer) + "\n"


def main() -> None:
    args = parse_args()
    rows = dedupe_rows(collect_rows(args.root, args.accuracy_key, args.mia_key))

    if args.methods is not None:
        allowed = {canonical_method_id(method_id) for method_id in args.methods}
        rows = [row for row in rows if canonical_method_id(str(row["method_id"])) in allowed]
    else:
        rows = [row for row in rows if "/" not in str(row["method_id"])]

    if not rows:
        raise SystemExit(f"No usable evaluation_result.json files found under {args.root}")

    table = build_table(
        rows,
        args.caption,
        args.label,
        args.decimals,
        args.mia_key,
        args.accuracy_key,
        include_descriptions=not args.no_descriptions_in_caption,
    )

    if args.output is not None:
        args.output.write_text(table)
        print(f"Saved LaTeX table to {args.output}")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
