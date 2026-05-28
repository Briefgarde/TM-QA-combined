import torch
from sklearn.metrics import ndcg_score
import numpy as np
import time


# alt rerank with time and token perf
def rerank(
    query: str,
    candidates: list[dict],
    model,
    tokenizer,
    batch_size: int = 16,
    device: str = None
) -> tuple[list[dict], dict]:
    """
    Scores and sorts candidates by relevance to the query using the cross-encoder.

    Args:
        query      : The query string.
        candidates : List of dicts with at least 'sentence' and 'relevance_score' keys.
        model      : Loaded cross-encoder model.
        tokenizer  : Associated tokenizer.
        batch_size : Number of (query, sentence) pairs per forward pass.
        device     : Target device. If None, inferred from model parameters.

    Returns:
        ranked   : Candidates sorted by predicted relevance score descending,
                   each dict extended with a 'predicted_score' key.
        metadata : Dict with reranker_time_ms and reranker_input_tokens.
    """
    if device is None:
        device = next(model.parameters()).device

    sentences = [c['sentence'] for c in candidates]
    all_scores = []
    total_tokens = 0

    model.eval()
    start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            encoded = tokenizer(
                [[query, s] for s in batch_sentences],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            total_tokens += encoded['input_ids'].numel()
            logits = model(**encoded).logits.squeeze(-1)
            all_scores.extend(logits.cpu().tolist())

    elapsed_ms = (time.perf_counter() - start) * 1000

    ranked = [
        {**c, 'predicted_score': score}
        for c, score in zip(candidates, all_scores)
    ]
    ranked.sort(key=lambda x: x['predicted_score'], reverse=True)

    metadata = {
        "reranker_time_ms": elapsed_ms,
        "reranker_input_tokens": total_tokens
    }

    return ranked, metadata


def evaluate_reranker(
    ranked_candidates: list[dict],
    threshold: float = 0.5,
    k_values: list[int] = [5, 10, 20]
) -> dict:
    """
    Evaluates a single query's ranked candidate list against binary relevance labels.

    Args:
        ranked_candidates : Output of rerank() — list of dicts with 'relevance_score'
                            and 'predicted_score', sorted by predicted_score descending.
        threshold         : Cutoff for binarizing relevance_score into a label.
        k_values          : List of cutoff values for @k metrics.

    Returns:
        Dict of metric names to float values.
    """
    labels = [int(c['relevance_score'] >= threshold) for c in ranked_candidates]
    scores = [c['predicted_score'] for c in ranked_candidates]

    n_relevant = sum(labels)
    results = {}

    # MAP
    num_hits = 0
    sum_precision = 0.0
    for i, label in enumerate(labels):
        if label == 1:
            num_hits += 1
            sum_precision += num_hits / (i + 1)
    results['MAP'] = (sum_precision / n_relevant) if n_relevant > 0 else 0.0

    # MRR
    results['MRR'] = 0.0
    for i, label in enumerate(labels):
        if label == 1:
            results['MRR'] = 1.0 / (i + 1)
            break

    # NDCG@k, P@k, R@k, F1@k
    for k in k_values:
        top_k_labels = labels[:k]
        n_relevant_at_k = sum(top_k_labels)

        # NDCG@k — sklearn expects 2D arrays
        if n_relevant > 0:
            ndcg = ndcg_score(
                y_true=np.array([labels]),
                y_score=np.array([scores]),
                k=k
            )
        else:
            ndcg = 0.0
        results[f'NDCG@{k}'] = ndcg

        # P@k
        precision = n_relevant_at_k / k
        results[f'P@{k}'] = precision

        # R@k
        recall = (n_relevant_at_k / n_relevant) if n_relevant > 0 else 0.0
        results[f'R@{k}'] = recall

        # F1@k
        if precision + recall > 0:
            results[f'F1@{k}'] = 2 * precision * recall / (precision + recall)
        else:
            results[f'F1@{k}'] = 0.0

    return results


def evaluate_reranker_dataset(
    all_ranked_candidates: list[list[dict]],
    all_metadata: list[dict],
    threshold: float = 0.5,
    k_values: list[int] = [5, 10, 20]
) -> dict:
    """
    Evaluates the reranker over the full dataset by averaging per-query metrics.

    Args:
        all_ranked_candidates : List of rerank() outputs, one per query.
        all_metadata          : List of metadata, one per query.
        threshold             : Cutoff for binarizing relevance_score.
        k_values              : List of cutoff values for @k metrics.

    Returns:
        Dict of metric names to macro-averaged float values across queries.
    """

    all_results = [
        evaluate_reranker(ranked, threshold, k_values)
        for ranked in all_ranked_candidates
    ]

    averaged = {}
    for metric in all_results[0].keys():
        averaged[metric] = float(np.mean([r[metric] for r in all_results]))

    averaged['avg_reranker_time_ms'] = float(np.mean([m['reranker_time_ms'] for m in all_metadata]))
    averaged['avg_reranker_input_tokens'] = float(np.mean([m['reranker_input_tokens'] for m in all_metadata]))

    return averaged