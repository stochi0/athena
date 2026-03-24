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

