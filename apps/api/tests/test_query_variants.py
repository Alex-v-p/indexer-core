from __future__ import annotations

import pytest

from packages.rag_core.retrieval import (
    LLMQueryVariantGenerator,
    QueryVariantGenerationError,
    build_query_variant_prompt,
    parse_query_variants,
)


class StaticLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_parse_query_variants_accepts_json_and_removes_duplicates() -> None:
    variants = parse_query_variants(
        '''```json
        {"queries": [
            "How does reciprocal rank fusion combine results?",
            "What is RRF in hybrid retrieval?",
            "what is rrf in hybrid retrieval?",
            "How does hybrid search merge vector and keyword rankings?"
        ]}
        ```''',
        original_question="How does hybrid retrieval use RRF?",
        count=3,
    )

    assert variants == [
        "How does reciprocal rank fusion combine results?",
        "What is RRF in hybrid retrieval?",
        "How does hybrid search merge vector and keyword rankings?",
    ]


def test_parse_query_variants_has_a_line_based_fallback() -> None:
    variants = parse_query_variants(
        "Here are the alternatives:\n1. multi-query retrieval query expansion\n- retrieving with paraphrased questions",
        original_question="How does multi-query retrieval work?",
        count=2,
    )

    assert variants == [
        "multi-query retrieval query expansion",
        "retrieving with paraphrased questions",
    ]


async def test_llm_query_variant_generator_uses_prompt_and_validates_output() -> None:
    llm = StaticLLM('{"queries": ["query expansion in RAG", "RAG retrieval with paraphrases"]}')
    generator = LLMQueryVariantGenerator(llm_provider=llm)

    variants = await generator.generate("How does multi-query RAG work?", count=2)

    assert variants == ["query expansion in RAG", "RAG retrieval with paraphrases"]
    assert "Create exactly 2" in llm.prompts[0]
    assert "How does multi-query RAG work?" in llm.prompts[0]


async def test_llm_query_variant_generator_does_not_truncate_the_original_question() -> None:
    question = "How does this architecture behave? " + ("context " * 80)
    llm = StaticLLM('{"queries": ["architecture behavior with extended context"]}')
    generator = LLMQueryVariantGenerator(llm_provider=llm, max_variant_chars=40)

    await generator.generate(question, count=1)

    assert "context context context" in llm.prompts[0]
    assert len(llm.prompts[0]) > len(question)


async def test_llm_query_variant_generator_rejects_empty_model_output() -> None:
    generator = LLMQueryVariantGenerator(llm_provider=StaticLLM('{"queries": []}'))

    with pytest.raises(QueryVariantGenerationError, match="no usable alternatives"):
        await generator.generate("What is indexed?", count=2)


def test_query_variant_prompt_requires_a_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        build_query_variant_prompt("  ", count=2)
