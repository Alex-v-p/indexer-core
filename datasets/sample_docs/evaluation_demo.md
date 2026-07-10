# Indexer Core evaluation demo

## Query execution

Indexer Core passes a shared `QueryState` through graph nodes. The baseline graph executes retrieval first and answer generation second, so every evaluated question exercises both evidence retrieval and the final answer path.

## Evidence and citations

Retrieved chunks are normalized into evidence items with a rank, score, source identifiers, text, and metadata. Answer generation creates citations that point back to the ranked evidence used as context.

## Evaluation metrics

The evaluation harness measures recall at k, mean reciprocal rank, and citation hit rate. Answer faithfulness is represented by an explicit placeholder until a groundedness evaluator is implemented.
