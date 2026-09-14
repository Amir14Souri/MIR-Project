# LLM Output Summary

## Model Configuration
*   **Language Model Name:** `Qwen/Qwen2.5-1.5B-Instruct`
*   **Prompting Template:** We provide a `system` instruction restricting output to ONLY valid JSON based on the exact requested schema. The `user` prompt provides the query details (text/image presence) and the top 5 reranked products (including Title, Type, Category, Color, Material, and Features).

## Generation Statistics
*   **Number of generated answers:** [TODO]
*   **JSON validity rate:** [TODO]%
*   **Schema validity rate:** [TODO]%
*   **Number of repaired outputs (after retries):** [TODO]

## Examples

### Example 1: Exact Match Found
[TODO: Insert example JSON]

### Example 2: Negative Constraint Handled
[TODO: Insert example JSON]

### Example 3: Ask Clarification / Substitute Recommended
[TODO: Insert example JSON]

## Failure Cases / Limitations
[TODO: Describe at least one failure case or limitation of the LLM generation (e.g., hallucinated reasons, overly strict matching, or struggling with visual aspects of image queries).]
