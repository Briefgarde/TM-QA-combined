import json
from datetime import datetime
from pathlib import Path
import os

def save_results_reranker(
    scored: list[list[dict]],
    metadata: list[dict],
    metrics: dict,
    model_path_or_name: str,
    threshold: float,
    k_values: list[int],
    output_dir: str = "."
) -> None:
    """
    Saves ranked output and evaluation metrics to JSON files.

    Args:
        scored             : Output of rerank() calls — list of ranked candidate lists.
        metadata           : List of per-query metadata dicts from rerank().
        metrics            : Output of evaluate_reranker_dataset().
        model_path_or_name : Model identifier, used in filename.
        threshold          : Threshold used for binary label computation.
        k_values           : k values used in evaluation.
        output_dir         : Directory to write output files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_path_or_name.replace("/", "_")

    # Fuse scored and metadata by query index
    fused = [
        {
            "reranker_time_ms": meta["reranker_time_ms"],
            "reranker_input_tokens": meta["reranker_input_tokens"],
            "candidates": candidates
        }
        for candidates, meta in zip(scored, metadata)
    ]

    results_payload = {
        "metadata": {
            "model": model_path_or_name,
            "threshold": threshold,
            "k_values": k_values,
            "timestamp": timestamp,
            "n_queries": len(scored)
        },
        "metrics": metrics,
        "ranked_output": fused
    }

    output_path = Path(output_dir) / f"{model_slug}_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"Results saved to {output_path}")


def save_result_generation(
    per_query_results: list[dict],
    averaged: dict,
    metadata: dict,
    model_path_or_name: str,
    output_dir: str
) -> None:
    """
    Saves generation evaluation results to a single JSON file.

    Args:
        per_query_results  : Output of evaluate_generation_dataset(), one dict per query.
        averaged           : Macro-averaged metrics across queries.
        metadata           : Pipeline metadata dict containing generation and
                             reranking configuration.
        model_path_or_name : GenLM identifier, used in filename.
        output_dir         : Directory to write output file. Created if it does not exist.
    """
    

    payload = {
        "metadata": metadata,
        "averaged_metrics": averaged,
        "per_query_results": per_query_results
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_path_or_name.replace("/", "_")
    mode = metadata.get("generationMetadata", {}).get("modeContextAssembling", "unknown")
    k = metadata.get("generationMetadata", {}).get("topk", {})
    split = metadata.get("split", {})
    filename = f"{model_slug}_top{k}_{mode}_{split}_{timestamp}.json"

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Generation results saved to {output_path}")