# HuggingFace dataset schema (ScaleAI/lhaw)

Reference for columns produced by the upstream [ScaleAI/lhaw](https://huggingface.co/datasets/ScaleAI/lhaw) dataset. The environment maps these into verifiers rollout examples in `core/dataset.py`.


| Column                  | Type         | Description                                                            |
| ----------------------- | ------------ | ---------------------------------------------------------------------- |
| `variant_id`            | `VARCHAR`    | Identifier for the variant.                                            |
| `original_task`         | `VARCHAR`    | The original task name.                                                |
| `dataset`               | `VARCHAR`    | Source benchmark name (e.g. MCP-Atlas, SWE-Bench Pro).                 |
| `original_prompt`       | `VARCHAR`    | Fully specified prompt text.                                           |
| `underspecified_prompt` | `VARCHAR`    | Prompt shown to the model (intentionally incomplete).                  |
| `information_dimension` | `VARCHAR[]`  | Which information aspects were removed or thinned.                     |
| `ambiguity_class`       | `VARCHAR`    | `outcome-critical`, `divergent`, or `benign`.                          |
| `removed_segments`      | struct array | Segments removed from the prompt (dimension, id, subdimension, value). |
| `expected_questions`    | struct array | Reference clarifying questions per segment.                            |
| `terminal_states`       | `VARCHAR`    | Description of terminal states for the variant.                        |


## Field notes

- `**underspecified_prompt**`: Model-facing task text (ambiguous by design).
- `**expected_questions**`: Ground-truth style signal for what good clarification might ask.
- `**ambiguity_class**`: Stratifies how harmful or open-ended the missing detail is.
- `**removed_segments**`: Oracle content the simulated user can answer from.
- `**information_dimension**`: Filter or analyze along goal / constraint / input / context style axes.

## Optional extensions for `native_reward`

The released HF dataset does not include native benchmark rewards. If you are extending the schema for native benchmark reward training, the root environment will also consume these optional fields when present:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `native_result` | struct | Native per-trajectory result with fields like `success`, `score`, and `total`. |
| `native_result_path` | `VARCHAR` | Path to a JSON file containing the native result payload. |
| `native_trials` | struct array | Trial-level outcomes for computing `pass@3`, `Ckpt%`, `Ask%`, and `Avg/Traj`. |
| `native_baseline_trials` | struct array | Baseline trial outcomes without `ask_user`, used for `Gain/Q`. |
| `native_summary` | struct | Precomputed aggregate metrics such as `pass_at_3`, `checkpoint_percent`, `ask_percent`, `avg_questions_per_trajectory`, and `gain_per_question`. |

