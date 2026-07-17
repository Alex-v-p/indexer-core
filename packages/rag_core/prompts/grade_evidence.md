You grade retrieved evidence for a question before an answer is generated.

Return exactly one JSON object and no prose or Markdown fences:
{
  "grades": [
    {
      "rank": 1,
      "relevance_score": 0.0,
      "rationale": "one short sentence"
    }
  ],
  "sufficiency": "missing | weak | sufficient",
  "coverage_score": 0.0,
  "rationale": "one short sentence"
}

Rules:
- Return exactly one grade for every evidence rank shown below. Do not add or omit ranks.
- relevance_score measures how directly a chunk helps answer the question, from 0.0 to 1.0.
- A chunk is considered relevant when its relevance_score is at least {{ relevance_threshold }}.
- missing: no chunk contains relevant answer evidence.
- weak: at least one chunk is relevant, but important facts, sides of a comparison, steps, or context are absent or uncertain.
- sufficient: the relevant chunks collectively support a grounded answer to the whole question.
- coverage_score measures how completely the evidence set covers the question, from 0.0 to 1.0.
- Judge only the supplied evidence. Do not rely on outside knowledge or retrieval scores.

Question:
{{ question }}

Evidence:
{{ evidence }}
