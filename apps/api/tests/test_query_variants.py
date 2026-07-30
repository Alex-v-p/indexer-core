from __future__ import annotations

import pytest

from packages.rag_core.retrieval import (
    LLMQueryVariantGenerator,
    QueryVariantGenerationError,
    build_query_variant_prompt,
    parse_query_variants,
    query_variant_response_schema,
)


class StaticLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.schemas: list[dict[str, object]] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses[0]

    async def generate_structured(self, prompt: str, *, response_schema) -> str:
        self.prompts.append(prompt)
        self.schemas.append(response_schema)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def test_parse_query_variants_accepts_exact_json() -> None:
    variants = parse_query_variants(
        '''```json
        {"queries": [
            "How does reciprocal rank fusion combine results?",
            "What is RRF in hybrid retrieval?",
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


def test_parse_query_variants_rejects_normalized_duplicates() -> None:
    with pytest.raises(QueryVariantGenerationError, match="distinct"):
        parse_query_variants(
            '{"queries":["What is RRF?", " what   is rrf? "]}',
            original_question="How does hybrid retrieval work?",
            count=2,
        )


def test_parse_query_variants_no_longer_accepts_line_oriented_output() -> None:
    with pytest.raises(QueryVariantGenerationError, match="invalid JSON"):
        parse_query_variants(
            "1. multi-query retrieval query expansion\n- retrieving with paraphrased questions",
            original_question="How does multi-query retrieval work?",
            count=2,
        )


def test_query_variant_schema_is_exact_and_bounded() -> None:
    schema = query_variant_response_schema(count=3, max_variant_chars=88)
    queries = schema["properties"]["queries"]

    assert schema["additionalProperties"] is False
    assert queries["minItems"] == 1
    assert queries["maxItems"] == 3
    assert queries["uniqueItems"] is True
    assert queries["items"]["maxLength"] == 88


async def test_llm_query_variant_generator_uses_prompt_and_validates_output() -> None:
    llm = StaticLLM('{"queries": ["query expansion in RAG", "RAG retrieval with paraphrases"]}')
    generator = LLMQueryVariantGenerator(llm_provider=llm)

    variants = await generator.generate("How does multi-query RAG work?", count=2)

    assert variants == ["query expansion in RAG", "RAG retrieval with paraphrases"]
    assert "Create exactly 2" in llm.prompts[0]
    assert "How does multi-query RAG work?" in llm.prompts[0]
    assert len(llm.schemas) == 1
    assert llm.schemas[0] == query_variant_response_schema(count=2)


async def test_llm_query_variant_generator_repairs_once_with_request_local_metadata() -> None:
    llm = StaticLLM(
        "not-json",
        '{"queries": ["query expansion in RAG", "RAG retrieval with paraphrases"]}',
    )
    generator = LLMQueryVariantGenerator(llm_provider=llm)

    result = await generator.generate_with_metadata("How does multi-query RAG work?", count=2)

    assert result.variants == ("query expansion in RAG", "RAG retrieval with paraphrases")
    assert result.structured_output.outcome == "repair_valid"
    assert result.structured_output.failure_code == "invalid_json"
    assert len(llm.schemas) == 2


async def test_llm_query_variant_generator_does_not_truncate_the_original_question() -> None:
    question = "How does this architecture behave? " + ("context " * 80)
    llm = StaticLLM('{"queries": ["architecture behavior with extended context"]}')
    generator = LLMQueryVariantGenerator(llm_provider=llm, max_variant_chars=40)

    await generator.generate(question, count=1)

    assert "context context context" in llm.prompts[0]
    assert len(llm.prompts[0]) > len(question)


async def test_llm_query_variant_generator_rejects_empty_model_output() -> None:
    generator = LLMQueryVariantGenerator(llm_provider=StaticLLM('{"queries": []}'))

    with pytest.raises(QueryVariantGenerationError, match="Structured query-variant generation failed") as captured:
        await generator.generate("What is indexed?", count=2)

    assert captured.value.structured_output is not None
    assert captured.value.structured_output.failure_code == "repair_invalid"


def test_query_variant_prompt_requires_a_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        build_query_variant_prompt("  ", count=2)
