import json
from datetime import datetime
from pathlib import Path

def save_results(ranked_output, metrics, model_path_or_name, threshold, k_values, output_dir="."):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_path_or_name.replace("/", "_")
    
    metadata = {
        "model": model_path_or_name,
        "threshold": threshold,
        "k_values": k_values,
        "timestamp": timestamp,
        "n_queries": len(ranked_output)
    }

    results_payload = {
        "metadata": metadata,
        "metrics": metrics
    }

    metrics_path = Path(output_dir) / f"{model_slug}_{timestamp}_metrics.json"
    ranked_path = Path(output_dir) / f"{model_slug}_{timestamp}_ranked.json"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    with open(ranked_path, "w", encoding="utf-8") as f:
        json.dump(ranked_output, f, indent=2)

    print(f"Metrics saved to {metrics_path}")
    print(f"Ranked output saved to {ranked_path}")