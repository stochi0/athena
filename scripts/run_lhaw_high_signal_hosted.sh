#!/usr/bin/env bash
# Run the highest-signal LHAW eval TOMLs on Prime hosted infrastructure.
#
# Hosted runs execute on Prime-managed workers and store results in Prime Evals
# automatically (you do not run `prime eval push` for normal hosted-only flows).
#
# Prerequisites:
#   1. Publish this environment once: `uv sync && prime env push`
#   2. Export PRIME_EVAL_ENV_ID to your Environments Hub slug (e.g. primeintellect/lhaw_rlm)
#   3. Authenticate: `prime config set-api-key "$PRIME_API_KEY"`
#
# Optional env:
#   HF_TOKEN                — forwarded to hosted workers via --custom-secrets (local .env alone is not enough)
#   LHAW_EVAL_SUBSET=all|ambiguity|domains|dimensions
#   LHAW_PUSH_ENV=1           — run `prime env push` before evals (latest wheel on Hub)
#   LHAW_HOSTED_TIMEOUT_MINUTES   (default 180)
#   LHAW_HOSTED_POLL_INTERVAL     (default 30)
#
# Usage:
#   Put PRIME_API_KEY (and optionally PRIME_EVAL_ENV_ID) in repo .env, or export them.
#   ./scripts/run_lhaw_high_signal_hosted.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if ! command -v prime >/dev/null 2>&1; then
  echo "error: prime CLI not found. Install with: uv tool install prime" >&2
  exit 1
fi

PRIME_EVAL_ENV_ID="${PRIME_EVAL_ENV_ID:-}"
if [[ -z "$PRIME_EVAL_ENV_ID" && -f "$ROOT/.prime/.env-metadata.json" ]]; then
  PRIME_EVAL_ENV_ID="$(
    PRIME_METADATA_JSON="$ROOT/.prime/.env-metadata.json" python3 -c \
      'import json, os; d = json.load(open(os.environ["PRIME_METADATA_JSON"])); print(d["owner"] + "/" + d["name"])' 2>/dev/null || true
  )"
fi
if [[ -z "$PRIME_EVAL_ENV_ID" ]]; then
  echo "error: set PRIME_EVAL_ENV_ID to your Environments Hub slug (e.g. primeintellect/lhaw_rlm)" >&2
  echo "       add it to .env, export it, or run \`prime env push\` from this repo (writes .prime/.env-metadata.json)" >&2
  exit 1
fi

SUBSET="${LHAW_EVAL_SUBSET:-all}"
TIMEOUT_MINUTES="${LHAW_HOSTED_TIMEOUT_MINUTES:-180}"
POLL_INTERVAL="${LHAW_HOSTED_POLL_INTERVAL:-30}"

# Hosted workers do not see this machine's .env; pass HF hub token when set.
HOSTED_SECRET_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  _hosted_secrets_json="$(
    HF_TOKEN="$HF_TOKEN" python3 -c 'import json, os; print(json.dumps({"HF_TOKEN": os.environ["HF_TOKEN"]}))'
  )"
  HOSTED_SECRET_ARGS=(--custom-secrets "$_hosted_secrets_json")
fi

declare -a CONFIGS
case "$SUBSET" in
  all)
    CONFIGS=(
      configs/eval/slice_outcome_critical.toml
      configs/eval/slice_benign.toml
      configs/eval/slice_divergent.toml
      configs/eval/slice_swe_bench.toml
      configs/eval/slice_mcp_atlas.toml
      configs/eval/slice_agent_company.toml
      configs/eval/dim_goal.toml
      configs/eval/dim_constraint.toml
      configs/eval/dim_input.toml
      configs/eval/dim_context.toml
      configs/eval/dim_goal_and_constraint.toml
    )
    ;;
  ambiguity)
    CONFIGS=(
      configs/eval/slice_outcome_critical.toml
      configs/eval/slice_benign.toml
      configs/eval/slice_divergent.toml
    )
    ;;
  domains)
    CONFIGS=(
      configs/eval/slice_swe_bench.toml
      configs/eval/slice_mcp_atlas.toml
      configs/eval/slice_agent_company.toml
    )
    ;;
  dimensions)
    CONFIGS=(
      configs/eval/dim_goal.toml
      configs/eval/dim_constraint.toml
      configs/eval/dim_input.toml
      configs/eval/dim_context.toml
      configs/eval/dim_goal_and_constraint.toml
    )
    ;;
  *)
    echo "error: unknown LHAW_EVAL_SUBSET=$SUBSET (all|ambiguity|domains|dimensions)" >&2
    exit 1
    ;;
esac

if [[ "${LHAW_PUSH_ENV:-0}" == "1" ]]; then
  echo "==> prime env push"
  prime env push
fi

declare -a TMP_TOMLS=()

cleanup() {
  rm -f "${TMP_TOMLS[@]}"
}
trap cleanup EXIT

prepare_hosted_toml() {
  local src="$1"
  local out
  out="$(mktemp)"
  # Hub slug + drop local-only env_dir_path (hosted loads the published env)
  sed -e "s|^env_id = .*|env_id = \"${PRIME_EVAL_ENV_ID}\"|" \
    -e '/^env_dir_path = /d' \
    "$src" > "$out"
  echo "$out"
}

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "error: missing $cfg" >&2
    exit 1
  fi
  prepared="$(prepare_hosted_toml "$cfg")"
  TMP_TOMLS+=("$prepared")
  name="lhaw-$(basename "$cfg" .toml)"
  echo "==> hosted eval: $cfg (name: $name)"
  prime eval run "$prepared" \
    --env-path "$ROOT" \
    --hosted \
    --follow \
    --poll-interval "$POLL_INTERVAL" \
    --timeout-minutes "$TIMEOUT_MINUTES" \
    --allow-sandbox-access \
    --allow-instances-access \
    --eval-name "$name" \
    "${HOSTED_SECRET_ARGS[@]}"
done

echo "==> Recent evaluations (hosted runs are already on Prime Evals)"
prime eval list --plain 2>/dev/null | head -25 || true
