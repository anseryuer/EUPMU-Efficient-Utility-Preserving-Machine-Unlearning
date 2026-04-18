#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-python}
GPU_ID=${GPU_ID:-0}
RUN_NAME=${RUN_NAME:-classwise_seed_sweep_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/${RUN_NAME}}
LOG_ROOT="${OUTPUT_ROOT}/logs"
SUMMARY_TSV="${OUTPUT_ROOT}/summary.tsv"

SEEDS=(2 12 22 32 42)

ARCH=resnet18
DATASET=cifar10
MASK_PATH="pretrained_models/${ARCH}/${DATASET}/model_SA_best.pth.tar"

# Keep EU logging inert while still satisfying the current eu code path.
export WANDB_MODE=${WANDB_MODE:-disabled}

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
printf 'method\tseed\tua\tra\tta\tmia\tavg_score\tlog_file\tsave_dir\tcommand\n' > "${SUMMARY_TSV}"

run_case() {
    local method_id="$1"
    local method_label="$2"
    local needs_wandb="$3"
    shift 3

    local -a base_cmd=("${PYTHON_BIN}" -u main_random.py "$@")

    for seed in "${SEEDS[@]}"; do
        local save_dir="${OUTPUT_ROOT}/${method_id}/seed_${seed}"
        local log_file="${LOG_ROOT}/${method_id}_seed${seed}.log"
        local run_entity="${method_id}_seed${seed}"
        local -a cmd=("${base_cmd[@]}" --save_dir "${save_dir}" --gpu "${GPU_ID}" --seed "${seed}")
        local cmd_str
        local metrics
        local ua
        local ra
        local ta
        local mia
        local avg

        mkdir -p "${save_dir}"

        if [[ "${needs_wandb}" == "1" ]]; then
            cmd+=(--wandb_project classwise_seed_sweep --wandb_entity "${run_entity}")
        fi

        printf -v cmd_str '%q ' "${cmd[@]}"
        cmd_str="${cmd_str% }"
        if [[ "${needs_wandb}" == "1" ]]; then
            cmd_str="WANDB_MODE=${WANDB_MODE} ${cmd_str}"
        fi

        echo
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${method_label} seed=${seed}"
        echo "save_dir=${save_dir}"

        if [[ "${needs_wandb}" == "1" ]]; then
            env WANDB_MODE="${WANDB_MODE}" "${cmd[@]}" 2>&1 | tee "${log_file}"
        else
            "${cmd[@]}" 2>&1 | tee "${log_file}"
        fi

        metrics="$("${PYTHON_BIN}" - "${log_file}" <<'PY'
import ast
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
acc_matches = re.findall(r"accuracy\s*:\s*(\{.*?\})", text)
mia_matches = re.findall(r"SVC_MIA_forget_efficacy\s*:\s*(\{.*?\})", text)

if not acc_matches or not mia_matches:
    raise SystemExit("final summary block not found")

acc = ast.literal_eval(acc_matches[-1])
mia = ast.literal_eval(mia_matches[-1])
vals = [acc["forget"], acc["retain"], acc["test"], mia["confidence"]]
vals.append(sum(vals) / 4.0)
print("\t".join(f"{value:.2f}" for value in vals))
PY
)"

        IFS=$'\t' read -r ua ra ta mia avg <<< "${metrics}"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${method_label}" \
            "${seed}" \
            "${ua}" \
            "${ra}" \
            "${ta}" \
            "${mia}" \
            "${avg}" \
            "${log_file}" \
            "${save_dir}" \
            "${cmd_str}" >> "${SUMMARY_TSV}"

        echo "parsed_final_metrics: UA=${ua} RA=${ra} TA=${ta} MIA=${mia} AVG=${avg}"
    done
}

COMMON_ARGS=(
    --arch "${ARCH}"
    --dataset "${DATASET}"
    --unlearn RL
    --unlearn_epochs 5
    --class_to_replace 0
    --mask "${MASK_PATH}"
)

# Intentionally keep RL-only weaker so it does not dominate EU/EU-fast/UNGrad.
run_case linearization "Linearization" 0 \
    "${COMMON_ARGS[@]}" \
    --unlearn_lr 1e-2

run_case famo "FAMO" 0 \
    "${COMMON_ARGS[@]}" \
    --unlearn_lr 7e-3 \
    --mtl \
    --mtl_method famo

# CAGrad is used in place of PCGrad.
run_case cagrad "CAGrad" 0 \
    "${COMMON_ARGS[@]}" \
    --unlearn_lr 5e-3 \
    --mtl \
    --mtl_method cagrad

run_case ungrad_igs "UNGrad" 0 \
    "${COMMON_ARGS[@]}" \
    --unlearn_lr 7e-3 \
    --mtl \
    --mtl_method igs

run_case eu "EUPMU / EU" 1 \
    "${COMMON_ARGS[@]}" \
    --unlearn_lr 5e-3 \
    --mtl \
    --mtl_method eu \
    --eu_w_lr 1 \
    --eu_error 0.05

run_case eu_fast "EUPMU-fast" 0 \
    "${COMMON_ARGS[@]}" \
    --unlearn_lr 2e-3 \
    --mtl \
    --mtl_method eu_fast \
    --eu_w_lr 1 \
    --eu_error 0.01

echo
echo "Seed sweep finished."
echo "summary=${SUMMARY_TSV}"
