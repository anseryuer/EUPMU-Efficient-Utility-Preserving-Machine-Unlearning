import os, re, ast, csv, json, math, random, datetime, subprocess
from typing import Dict, List, Optional, Any, Tuple

# ===============================
# 0) USER CONFIG (FILL THESE)
# ===============================

MASKS = {
    "cifar10":  "pretrained_models/resnet18/cifar10/model_SA_best.pth.tar",
    "cifar100": "pretrained_models/resnet18/cifar100/model_SA_best.pth.tar",
    "TinyImagenet": "pretrained_models/resnet18/TinyImagenet/model_SA_best.pth.tar",
}
GPU_ID = "0"
DATA_DIR_TINY = "../data/tiny-imagenet-200"

# Optional: turn on Gemini suggestions once you’ve added baselines and an API key
USE_GEMINI = True

# Optional baselines per (dataset, algo, forget_n); if omitted, loss is None
# Example:
# BASELINES = {
# BASELINES: Dict[tuple, Dict[str, float]] = {}


# ===============================
# 1) FIXED EPOCHS + SEARCH SPACE
# ===============================

# epochs locked to README values
FIXED_EPOCHS = {
    "SalUn": 10,
    "SalUn_Soft": 10,
    "RL": 10,
}

# Each algo config:
# - runner: which main_*.py to call
# - static: fixed flags (plus locked --unlearn_epochs injected later)
# - space: ONLY tunable params (all methods include --unlearn_lr)
PARAM_SEARCH = {
    "SalUn": {
        "runner": "main_random.py",
        "static": {"--unlearn": "RL"},
        "space": {
            "--unlearn_lr": {"type": "float_log", "min": 1e-5, "max": 1e-1},
        },
        "needs_saliency_path": True,
    },
    "SalUn_Soft": {
        "runner": "main_random.py",
        "static": {"--unlearn": "RL_proximal", "--mask_ratio": 0.5},
        "space": {
            "--unlearn_lr": {"type": "float_log", "min": 1e-5, "max": 1e-1},
        },
        "needs_saliency_path": True,
    },
    "RL" : {
        "runner": "main_forget.py",
        "static": {"--unlearn": "RL"},
        "space": {
            "--unlearn_lr": {"type": "float_log", "min": 1e-5, "max": 1e-1},
        },
        "needs_saliency_path": False,
    },
}

# DATASETS = ["TinyImagenet"]
DATASETS = ["cifar10", "cifar100", "TinyImagenet"]
OPPORTUNITIES = 5

FORGET_SIZES = {
    "cifar10": [4500, 13500, 22500],
    "cifar100": [4500, 13500, 22500],
    "TinyImagenet": [10000, 30000, 50000],
}

SAL_MAP_PERCENT = {
    "cifar10": {4500: "10.0", 13500: "30.0", 22500: "50.0"},
    "cifar100": {4500: "10.0", 13500: "30.0", 22500: "50.0"},
    "TinyImagenet": {10000: "10.0", 30000: "30.0", 50000: "50.0"},
}

ARCH = "resnet18"

BASELINE_METRICS = {
    "cifar10": (99.99555555555555, 94.55),
    "cifar100": (99.97111111111111, 75.62),
    "TinyImagenet": (95.29295293075, 66.35327055492886),
}

BASELINES = {
    (dataset, algo, forget_n): {"train_acc": metrics[0], "test_acc": metrics[1]}
    for dataset, metrics in BASELINE_METRICS.items()
    for algo in PARAM_SEARCH.keys()
    for forget_n in FORGET_SIZES[dataset]
}


# ===============================
# 2) OPTIONAL: GEMINI CLIENT
# ===============================
def maybe_get_gemini():
    if not USE_GEMINI:
        return None
    try:
        from google import genai
        return genai.Client()
    except Exception:
        return None

# ===============================
# 3) HELPERS
# ===============================
def log_uniform(lo: float, hi: float) -> float:
    import math, random
    return math.exp(random.uniform(math.log(lo), math.log(hi)))

def sample_param(spec: Dict[str, Any]) -> Any:
    t = spec["type"]
    if t == "float_log":
        return float(f"{log_uniform(spec['min'], spec['max']):.6g}")
    if t == "float":
        import random
        return float(f"{random.uniform(spec['min'], spec['max']):.6g}")
    if t == "int":
        import random
        return int(random.randint(spec["min"], spec["max"]))
    if t == "choice":
        import random
        return random.choice(spec["choices"])
    raise ValueError(f"Unknown param type: {t}")

def parse_metrics(output: str) -> Optional[Dict[str, float]]:
    m = re.search(r"accuracy\s*:\s*({.*})", output)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group(1))
        return {"train_acc": float(d["retain"]),
                "forget_acc": float(d["forget"]),
                "test_acc":   float(d["test"])}
    except Exception:
        return None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_saliency_map_path(dataset: str, forget_n: int) -> str:
    try:
        percent = SAL_MAP_PERCENT[dataset][forget_n]
    except KeyError as exc:
        raise KeyError(f"No saliency map configured for {dataset} forget_n={forget_n}") from exc

    path = os.path.join(
        "saliency_maps",
        ARCH,
        dataset,
        f"forget_{percent}%",
        "with_0.5.pt",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected saliency map not found at {path}")

    return path


def build_base_cmd(
    runner: str,
    dataset: str,
    forget_n: int,
    algo: str,
    saliency_path: Optional[str] = None,
) -> List[str]:
    cmd = [
        "python", "-u", runner,
        "--arch", ARCH,
        "--dataset", dataset,
        "--mask", MASKS[dataset],
        "--save_dir", "output",
        "--gpu", GPU_ID,
        "--num_indexes_to_replace", str(forget_n),  # random-data forgetting
    ]

    # Tiny ImageNet requires an explicit --data path
    if dataset == "TinyImagenet":
        cmd += ["--data", DATA_DIR_TINY]

    # fixed algo switches
    spec = PARAM_SEARCH[algo]
    for k, v in spec["static"].items():
        cmd += [k] if v == "" else [k, str(v)]

    if saliency_path:
        cmd += ["--path", saliency_path]

    # lock unlearn_epochs per your table
    cmd += ["--unlearn_epochs", str(FIXED_EPOCHS[algo])]
    return cmd

def run_once(cmd: List[str], dynamic: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    full = list(cmd)
    for k, v in dynamic.items():
        full += [k, str(v)]
    print("=" * 90)
    print("EXECUTING:", " ".join(full))
    print("=" * 90)
    try:
        r = subprocess.run(full, capture_output=True, text=True, check=True)
        return r.stdout, None
    except subprocess.CalledProcessError as e:
        msg = f"RC={e.returncode}\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
        print("--- ERROR ---\n" + msg)
        return "", msg

def compute_loss(baseline: Optional[Dict[str, float]], metrics: Optional[Dict[str, float]]) -> Optional[float]:
    if not baseline or not metrics:
        return None
    return (
        abs(baseline["train_acc"] - metrics["train_acc"]) +
        abs(baseline["test_acc"]  - metrics["test_acc"])  +
        abs((100 - metrics["forget_acc"]) - baseline["test_acc"])
    )

def csv_writer(path: str, header: List[str], row: Dict[str, Any]):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new: w.writeheader()
        w.writerow(row)

def jsonl_writer(path: str, row: Dict[str, Any]):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")

# --- Robust JSON extraction (unchanged logic, just centralized) ---
def _extract_json_like(text: str):
    if not text:
        return None
    txt = text.strip().replace("```json", "").replace("```", "").strip()
    # try direct loads
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # try from first '{'
    decoder = json.JSONDecoder()
    for i, ch in enumerate(txt):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(txt[i:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None

def suggest_with_gemini(
    client,
    history: List[Dict[str, Any]],
    tunable_keys: List[str],
    baseline: Dict[str, float],
    task_name: str = "AutoTune",
    invert_forget_term: bool = True,  # True = (100 - forget); False = (no 100-)
    max_attempts: int = 3,
):
    """
    Uses your original prompt style, with a switch to include or exclude the '100 - ...' term.
    - invert_forget_term=True  -> class-wise loss (old prompt formula)
    - invert_forget_term=False -> random-data loss (current task)
    Retries and robustly parses JSON.
    """
    if client is None or baseline is None:
        return None
    print("[Gemini] Generating suggestion based on history of", len(history), "records.")
    # Build the history string exactly like you had it
    history_str = "Experiment History (lower unlearning difference is better, increase learning rate if higher forget is needed, where the increase LR causes the model to be modified more. You can always try to crank it up to 1e-2 to see what happens, if the forget overshoot then you tune it back) (You have 5 opportunities for each command to test out the result so just try different ranges):\n"
    for record in history:
        params = {k: record[k] for k in tunable_keys if k in record}
        metrics = {k: v for k, v in record.items() if "acc" in k}
        # keep your favorite line/wording:
        loss_val = record.get("loss")
        if isinstance(loss_val, (int, float)):
            history_str += (
                f"- Parameters: {params}, Metrics: {metrics}, "
                f"Calculated unlearning difference: {loss_val:.4f}\n"
            )
        else:
            history_str += (
                f"- Parameters: {params}, Metrics: {metrics}, "
                f"Calculated unlearning difference: N/A\n"
            )

    # Pick the correct loss text
    if invert_forget_term:
        # CLASS-WISE (your old prompt)
        loss_text = (
            "unlearning difference = abs(original_train_acc - finetuned_train_acc) + "
            "abs(original_test_acc - finetuned_test_acc) + "
            "abs((100 - finetuned_forget_acc) - original_test_acc)"
        )
        notice_text = (
            "Notice that the forget accuracy is inverted in the last term: we want it to be as close "
            "to 100 - original_test_acc as possible."
        )
    else:
        # RANDOM-DATA (current task)
        loss_text = (
            "unlearning difference = abs(original_train_acc - finetuned_train_acc) + "
            "abs(original_test_acc - finetuned_test_acc) + "
            "abs(finetuned_forget_acc - original_test_acc)"
        )
        notice_text = (
            "Notice that for random-data forgetting we expect the unlearning accuracy to match "
            "the original test accuracy (both sample the same distribution), so no 100 - term."
        )

    prompt = f"""
You are an expert Machine Learning Engineer specializing in hyperparameter optimization.
I am performing a machine unlearning task called '{task_name}'.

My goal is to find the optimal hyperparameters to MINIMIZE a specific unlearning difference value.
The unlearning difference is calculated as:
{loss_text}

{notice_text}

The original model's performance was:
- original_train_acc: {baseline['train_acc']}
- original_test_acc: {baseline['test_acc']}

The parameters you can adjust are: {list(tunable_keys)}

Here is the history of the experiments I have run so far:
{history_str}

Analyze the history. See which parameter changes led to a lower loss.
Based on your analysis, suggest the BEST new set of hyperparameters for the next experiment.
Think step-by-step: if increasing a parameter made the loss worse, try decreasing it, and vice-versa. Explore the parameter space intelligently.

Provide your answer ONLY in a valid JSON format like this, with no other text or explanation:
{{
    "parameter1_name": value1,
    "parameter2_name": value2
}}
""".strip()

    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            parsed = _extract_json_like(getattr(resp, "text", "") or "")
            if parsed:
                # keep only the expected tunable keys
                return {k: parsed[k] for k in tunable_keys if k in parsed}
            preview = " ".join(((resp.text or "").strip().splitlines()[:3])) if getattr(resp, "text", None) else "<empty>"
            print(f"[Gemini] Unparseable response (attempt {attempt}): {preview}")
        except Exception as e:
            print(f"[Gemini] API error (attempt {attempt}): {e}")
        import time; time.sleep(0.5 * attempt)

    print("[Gemini] Falling back to random sampling after retries.")
    return None
def parse_mia_confidence(output: str) -> Optional[float]:
    """
    Extracts the 'confidence' field from:
    SVC_MIA_forget_efficacy : {'correctness': 0.71, 'confidence': 13.64, ...}
    Returns a float or None if not found/parsable.
    """
    m = re.search(r"SVC_MIA_forget_efficacy\s*:\s*({.*})", output)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group(1))
        # Some training logs might output ints—force float
        return float(d.get("confidence")) if "confidence" in d else None
    except Exception:
        return None

# ===============================
# 4) ORCHESTRATOR
# ===============================
def main():
    stamp = now_ts()
    out_dir = f"autotune_runs_{stamp}"
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "results.csv")
    jsonl_path = os.path.join(out_dir, "results.jsonl")

    csv_header = [
        "ts","dataset","algo","forget_n","iteration","runner",
        "train_acc","test_acc","forget_acc","loss","ok"
    ]

    client = maybe_get_gemini()

    for dataset in DATASETS:
        for forget_n in FORGET_SIZES[dataset]:
            for algo, spec in PARAM_SEARCH.items():
                saliency_path = get_saliency_map_path(dataset, forget_n) if spec.get("needs_saliency_path") else None
                base_cmd = build_base_cmd(
                    spec["runner"],
                    dataset,
                    forget_n,
                    algo,
                    saliency_path=saliency_path,
                )
                tunable_keys = list(spec["space"].keys())
                history: List[Dict[str, Any]] = []

                for it in range(1, (OPPORTUNITIES+1)):  # opportunities
                    baseline = BASELINES.get((dataset, algo, forget_n))
                    suggestion = suggest_with_gemini(client, history, tunable_keys, baseline) if (client and baseline and history) else None
                    if suggestion is None:
                        # random sample from the algo's space
                        suggestion = {k: sample_param(v) for k, v in spec["space"].items()}

                    stdout, err = run_once(base_cmd, suggestion)
                    metrics = parse_metrics(stdout) if stdout else None
                    mia_conf = parse_mia_confidence(stdout) if stdout else None

                    loss_val = compute_loss(baseline, metrics)

                    row: Dict[str, Any] = {
                        "ts": now_ts(),
                        "dataset": dataset,
                        "algo": algo,
                        "forget_n": forget_n,
                        "iteration": it,
                        "runner": spec["runner"],
                        "ok": err is None,
                        "loss": None if loss_val is None else float(f"{loss_val:.6g}"),
                        # new metric:
                        "mia_confidence": mia_conf,
                    }
                    if saliency_path:
                        row["saliency_path"] = saliency_path
                    row.update(suggestion)
                    if metrics:
                        row.update(metrics)
                    else:
                        row.update({"train_acc": None, "test_acc": None, "forget_acc": None})

                    # Expand CSV header with any new keys
                    for k in row.keys():
                        if k not in csv_header:
                            csv_header.append(k)

                    jsonl_writer(jsonl_path, row)
                    csv_writer(csv_path, csv_header, row)

                    # compact history for suggestion model
                    hist_item = {k: row.get(k) for k in (tunable_keys + ["train_acc","test_acc","forget_acc","loss"])}
                    history.append(hist_item)

    print("\n" + "="*60)
    print("DONE. Results saved to:")
    print(" -", csv_path)
    print(" -", jsonl_path)
    print("="*60)

if __name__ == "__main__":
    main()
