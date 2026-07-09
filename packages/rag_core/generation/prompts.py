from __future__ import annotations

from packages.rag_core.query import EvidenceItem


def build_answer_prompt(question: str, evidence: list[EvidenceItem]) -> str:
    """Build a citation-oriented RAG answer prompt."""

    evidence_block = "\n\n".join(
        f"[{item.rank}] {item.text.strip()}" for item in sorted(evidence, key=lambda item: item.rank)
    )

    return f"""You are an evidence-grounded RAG assistant.
Answer the user question using only the evidence below.

Rules:
- If the evidence does not contain the answer, say that the available documents do not contain enough information.
- Cite evidence inline with bracketed citation labels like [1] or [2].
- Keep the answer concise and factual.

Question:
{question}

Evidence:
{evidence_block}

Answer:
""".strip()
