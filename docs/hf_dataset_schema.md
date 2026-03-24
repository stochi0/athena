
# HuggingFace Dataset Schema

| Column                   | Type                                                                                                      | Description                                                                                                            |
|--------------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `variant_id`             | `VARCHAR`                                                                                                | Identifier for the variant.                                                                                            |
| `original_task`          | `VARCHAR`                                                                                                | The original task name.                                                                                                |
| `dataset`                | `VARCHAR`                                                                                                | Name of the dataset the variant belongs to.                                                                            |
| `original_prompt`        | `VARCHAR`                                                                                                | The original prompt text.                                                                                              |
| `underspecified_prompt`  | `VARCHAR`                                                                                                | Prompt that is intentionally underspecified.                                                                            |
| `information_dimension`  | `VARCHAR[]`                                                                                              | Array of information‑dimension strings.                                                                                |
| `ambiguity_class`        | `VARCHAR`                                                                                                | Class label describing the type of ambiguity.                                                                          |
| `removed_segments`       | `STRUCT(dimension VARCHAR, id VARCHAR, subdimension VARCHAR, "value" VARCHAR)[]`                         | Array of structs describing segments that were removed, with dimension, id, subdimension, and value.                   |
| `expected_questions`     | `STRUCT(questions VARCHAR[], segment_id VARCHAR)[]`                                                      | Array of structs containing expected questions (questions array) and the associated segment_id.                        |
| `terminal_states`        | `VARCHAR`                                                                                                | Description of terminal states.                                                                                        |

---

### Field Explanations

- **`underspecified_prompt`**: The model's input (the ambiguous task)
- **`expected_questions`**: Ground truth for what clarifying questions should be asked
- **`ambiguity_class`** (`outcome-critical` / `divergent` / `benign`): Difficulty stratification
- **`removed_segments`**: The oracle information the simulated user holds
- **`information_dimension`**: Lets you reward dimension-aware clarification

---
