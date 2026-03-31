# AdvancedIF Rubric Discovery Environment

Verifiers-native environment for learning to generate rubric lists from full
conversation trajectories in `facebook/AdvancedIF`.

## What This Environment Does

- Input: full trajectory (`conversation_history`) rendered as role-tagged text.
- Target: rubric list from `prompt_metadata.rubrics`.
- Model output: JSON object with `{"rubrics": [...]}`.
- Reward: LLM judge alignment with the gold rubric list (coverage, faithfulness,
  non-redundancy).

This setup stays within verifiers-native environment/rubric/parser patterns.

## Dataset Analysis (full split: `facebook/AdvancedIF`, `train`)

- Total rows: `1645`
- Benchmarks:
  - `carried_context_multi_turn_eval_v5`: `736`
  - `system_steerability_v2`: `507`
  - `complex_if_single_turn_v5`: `402`
- Assistant-turn distribution:
  - `0`: `405`, `1`: `125`, `2`: `162`, `3`: `177`, `4`: `431`
  - `5`: `157`, `6`: `86`, `7`: `52`, `8`: `22`, `9`: `28`
- Rubric-count distribution (count -> rows):
  - `1->23`, `2->60`, `3->91`, `4->128`, `5->152`, `6->209`, `7->229`,
    `8->191`, `9->191`, `10->86`, `11->66`, `12->63`, `13->44`, `14->53`,
    `15->14`, `16->16`, `17->12`, `18->6`, `19->4`, `20->7`

## Install

```bash
cd advanced_if
uv sync
```

## Run Eval

```bash
uv run prime eval run advanced-if-rubric-discovery \
  -m gpt-4.1-mini \
  -a '{"max_examples": 20, "judge_model": "gpt-4.1-mini"}'
```

## Config

`load_environment(config)` accepts:

- `dataset_name` (default `"facebook/AdvancedIF"`)
- `dataset_split` (default `"train"`)
- `max_examples` / `seed`
- `judge_model` / `judge_sampling_args`
- `judge_api_key_var` (default `PRIME_API_KEY`)
- `judge_base_url` (default `https://api.pinference.ai/api/v1`)
- `max_turns` (default `1`)
- `include_dataset_analysis_in_state` (default `true`)

## Notes

- Reward uses judge JSON fields: `coverage`, `faithful`, `non_redundant`.
- Judge calls are resolved through verifiers client config (`ClientConfig` + `resolve_client`)
  and default to Prime Inference via `PRIME_API_KEY`.
