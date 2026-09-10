#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 {webqsp|cwq} /absolute/output/root [/absolute/server-config.json]" >&2
  exit 2
fi
if [[ -z "${CHATKBQA_ROOT:-}" ]]; then
  echo "CHATKBQA_ROOT must point to a pinned ChatKBQA checkout" >&2
  exit 2
fi
if [[ -z "${RETRIEVAL_REPLICATE_1:-}" && -z "${RETRIEVAL_REPLICATE_2:-}" && "${ALLOW_UNCONFIRMED_BASELINE:-0}" != "1" ]]; then
  echo "Set RETRIEVAL_REPLICATE_1 (an independent inference PTH), or explicitly set ALLOW_UNCONFIRMED_BASELINE=1" >&2
  exit 2
fi
if [[ -z "${RETRIEVER_CHECKPOINT:-}" && "${ALLOW_UNCONFIRMED_BASELINE:-0}" != "1" ]]; then
  echo "Set RETRIEVER_CHECKPOINT to the cpt.pth used by retrieve/inference.py, or explicitly set ALLOW_UNCONFIRMED_BASELINE=1" >&2
  exit 2
fi

dataset="$1"
experiment_root="$2/$dataset"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
config="${3:-${QFORACLE_CONFIG:-$script_dir/configs/$dataset.json}}"
if [[ ! -f "$config" ]]; then
  echo "Config does not exist: $config" >&2
  exit 2
fi
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"

python -m experiments.query_factor_oracle.preflight \
  --config "$config" \
  --dataset "$dataset" \
  --output "$experiment_root/preflight.json"
report_budget="$(python -m experiments.query_factor_oracle.print_config \
  --config "$config" --field protocol.prompt_top_k)"

frozen_retrieval="$experiment_root/phase0/retrieval/baseline_retrieval.pth"
phase0_lock="$experiment_root/phase0/baseline.lock.json"
retrieval_checks=()
checkpoint_check=()
if [[ -n "${RETRIEVER_CHECKPOINT:-}" ]]; then
  checkpoint_check=(--retriever-checkpoint "$RETRIEVER_CHECKPOINT")
fi
confirmation_check=(--require-confirmed-baseline)
if [[ "${ALLOW_UNCONFIRMED_BASELINE:-0}" == "1" ]]; then
  confirmation_check=()
fi
if [[ -n "${RETRIEVAL_REPLICATE_1:-}" ]]; then
  retrieval_checks+=(--retrieval-replicate "rerun_1=$RETRIEVAL_REPLICATE_1")
fi
if [[ -n "${RETRIEVAL_REPLICATE_2:-}" ]]; then
  retrieval_checks+=(--retrieval-replicate "rerun_2=$RETRIEVAL_REPLICATE_2")
fi

if [[ -f "$phase0_lock" ]]; then
  if [[ ! -f "$frozen_retrieval" ]]; then
    echo "Existing Phase-0 lock has no frozen retrieval copy: $frozen_retrieval" >&2
    exit 2
  fi
  # Never replace an existing lock just to start/resume the workflow.  Audit
  # current inputs in an isolated directory and compare them with the original
  # lock; the canonical Phase-0 directory remains untouched on drift/failure.
  python -m experiments.query_factor_oracle.phase0_freeze \
    --config "$config" \
    --output-dir "$experiment_root/phase0_restart_audit" \
    "${retrieval_checks[@]}" \
    "${checkpoint_check[@]}" \
    --audit-datasets \
    --no-copy-retrieval \
    --verify-lock "$phase0_lock"
else
  python -m experiments.query_factor_oracle.phase0_freeze \
    --config "$config" \
    --output-dir "$experiment_root/phase0" \
    "${retrieval_checks[@]}" \
    "${checkpoint_check[@]}" \
    --audit-datasets
fi

for replicate in 0 1; do
  run_dir="$experiment_root/phase0/reasoning/replicate_$replicate"
  python -m experiments.query_factor_oracle.run_reasoning \
    --config "$config" \
    --retrieval "$frozen_retrieval" \
    --variant baseline \
    --replicate "$replicate" \
    --output-dir "$run_dir"
  python -m experiments.query_factor_oracle.evaluate_reasoning \
    --config "$config" \
    --predictions "$run_dir/predictions.jsonl" \
    --retrieval "$frozen_retrieval" \
    --output-dir "$run_dir/evaluation_phase0"
done

python -m experiments.query_factor_oracle.phase0_freeze \
  --config "$config" \
  --output-dir "$experiment_root/phase0" \
  --prediction "replicate_0=$experiment_root/phase0/reasoning/replicate_0/predictions.jsonl" \
  --prediction "replicate_1=$experiment_root/phase0/reasoning/replicate_1/predictions.jsonl" \
  --qa-summary "replicate_0=$experiment_root/phase0/reasoning/replicate_0/evaluation_phase0/qa_summary.json" \
  --qa-summary "replicate_1=$experiment_root/phase0/reasoning/replicate_1/evaluation_phase0/qa_summary.json" \
  "${retrieval_checks[@]}" \
  "${checkpoint_check[@]}" \
  --audit-datasets \
  "${confirmation_check[@]}" \
  --verify-lock "$phase0_lock"

python -m experiments.query_factor_oracle.phase1_build_factors \
  --config "$config" \
  --baseline "$frozen_retrieval" \
  --output-dir "$experiment_root/phase1"

# Re-evaluate the already generated baseline predictions with the same factor
# artifact used by every oracle arm, so family/status slices are paired.
for replicate in 0 1; do
  run_dir="$experiment_root/phase0/reasoning/replicate_$replicate"
  python -m experiments.query_factor_oracle.evaluate_reasoning \
    --config "$config" \
    --predictions "$run_dir/predictions.jsonl" \
    --retrieval "$frozen_retrieval" \
    --factors "$experiment_root/phase1/query_factors.jsonl.gz" \
    --output-dir "$run_dir/evaluation"
done

python -m experiments.query_factor_oracle.phase2_rerank \
  --config "$config" \
  --baseline "$frozen_retrieval" \
  --factors "$experiment_root/phase1/query_factors.jsonl.gz" \
  --output-dir "$experiment_root/phase2"

qa_arguments=(
  --qa-summary "baseline=$experiment_root/phase0/reasoning/replicate_0/evaluation/qa_summary.json"
)
for variant in family_structure relation_set relation_slot_branch direction all_factors; do
  run_dir="$experiment_root/phase2/$variant/reasoning/replicate_0"
  variant_retrieval="$experiment_root/phase2/$variant/retrieval_result.pth"
  python -m experiments.query_factor_oracle.run_reasoning \
    --config "$config" \
    --retrieval "$variant_retrieval" \
    --variant "$variant" \
    --replicate 0 \
    --output-dir "$run_dir"
  python -m experiments.query_factor_oracle.evaluate_reasoning \
    --config "$config" \
    --predictions "$run_dir/predictions.jsonl" \
    --retrieval "$variant_retrieval" \
    --factors "$experiment_root/phase1/query_factors.jsonl.gz" \
    --output-dir "$run_dir/evaluation"
  qa_arguments+=(--qa-summary "$variant=$run_dir/evaluation/qa_summary.json")
done

python -m experiments.query_factor_oracle.build_report \
  --retrieval-summary "$experiment_root/phase2/retrieval_summary.json" \
  --phase0-validation "$experiment_root/phase0/validation.json" \
  "${qa_arguments[@]}" \
  --budget "$report_budget" \
  --output-dir "$experiment_root/report"
