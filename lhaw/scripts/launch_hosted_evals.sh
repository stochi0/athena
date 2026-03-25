#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CUSTOM_SECRETS=""
if [[ -f .env ]]; then
  CUSTOM_SECRETS="$(ENV_FILE="$PWD/.env" python3 <<'PY'
import json, os
from pathlib import Path
out = {}
for raw in Path(os.environ["ENV_FILE"]).read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    k, v = k.strip(), v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        v = v[1:-1]
    out[k] = v
print(json.dumps(out))
PY
)"
fi
args=(--hosted)
[[ -n "$CUSTOM_SECRETS" ]] && args+=(--custom-secrets "$CUSTOM_SECRETS")
for cfg in \
  configs/eval/slice_outcome_critical.toml \
  configs/eval/slice_benign.toml \
  configs/eval/slice_divergent.toml \
  configs/eval/slice_swe_bench.toml \
  configs/eval/slice_mcp_atlas.toml \
  configs/eval/slice_agent_company.toml \
  configs/eval/dim_goal.toml \
  configs/eval/dim_constraint.toml \
  configs/eval/dim_input.toml \
  configs/eval/dim_context.toml \
  configs/eval/dim_goal_and_constraint.toml; do
  prime eval run stochi0/lhaw_rlm "$cfg" "${args[@]}"
done
