import json
from datetime import datetime
from pathlib import Path

# def save_results(ranked_output, metrics, model_path_or_name, threshold, k_values, output_dir="."):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     model_slug = model_path_or_name.replace("/", "_")
    
#     metadata = {
#         "model": model_path_or_name,
#         "threshold": threshold,
#         "k_values": k_values,
#         "timestamp": timestamp,
#         "n_queries": len(ranked_output)
#     }

#     results_payload = {
#         "metadata": metadata,
#         "metrics": metrics
#     }

#     metrics_path = Path(output_dir) / f"{model_slug}_{timestamp}_metrics.json"
#     ranked_path = Path(output_dir) / f"{model_slug}_{timestamp}_ranked.json"

#     metrics_path.parent.mkdir(parents=True, exist_ok=True)

#     with open(metrics_path, "w", encoding="utf-8") as f:
#         json.dump(results_payload, f, indent=2)
#     with open(ranked_path, "w", encoding="utf-8") as f:
#         json.dump(ranked_output, f, indent=2)

#     print(f"Metrics saved to {metrics_path}")
#     print(f"Ranked output saved to {ranked_path}")




# alt save_results with fusing int a single file.
def save_results(
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