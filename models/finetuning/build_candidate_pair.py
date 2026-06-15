import sys
sys.path.append('../../')

import random
from inference.rerank import rerank


def build_training_pairs(
    pool_entry: dict,
    model,
    tokenizer,
    threshold: float = 0.5,
    upper_margin: float = 0.15,
    lower_margin: float = None,
    max_positives_per_query: int = 10,
    n_random_negatives: int = 3,
    n_hard_negatives: int = 2,
    batch_size: int = 16,
    device: str = None
) -> list[dict]:
    """
    Builds (query, positive, negative) training triplets for a single pool entry.

    Positives are sentences with relevance_score >= threshold + upper_margin.
    Negatives are sentences with relevance_score <= threshold - lower_margin.
    Sentences in between are excluded (dead zone).

    If the number of positives exceeds max_positives_per_query, a random
    subset is sampled to prevent queries with disproportionately many
    positives from dominating the training set.

    For each positive, pairs are constructed with:
        - n_hard_negatives: negatives scored highest by the pretrained model.
        - n_random_negatives: negatives sampled randomly from the remaining pool.

    Args:
        pool_entry             : Single candidate pool entry with 'query' and 'candidates'.
        model                  : Pretrained reranker model for hard negative identification.
        tokenizer              : Associated tokenizer.
        threshold              : Relevance score boundary.
        upper_margin           : Positive boundary is threshold + upper_margin.
        lower_margin           : Negative boundary is threshold - lower_margin.
                                 If None, defaults to upper_margin (symmetric dead zone).
        max_positives_per_query: Maximum number of positives to use per query.
                                 If exceeded, a random sample is taken.
        n_random_negatives     : Number of random negatives per positive.
        n_hard_negatives       : Number of hard negatives per positive.
        batch_size             : Batch size for the pretrained model inference pass.
        device                 : Device for inference. If None, inferred from model.

    Returns:
        List of dicts, each with keys:
            'query'    : Query string.
            'positive' : Positive sentence string.
            'negative' : Negative sentence string.
    """
    if lower_margin is None:
        lower_margin = upper_margin

    positive_threshold = threshold + upper_margin
    negative_threshold = threshold - lower_margin

    candidates = pool_entry['candidates']
    query = pool_entry['query']

    positives = [
        c for c in candidates
        if c['relevance_score'] >= positive_threshold
    ]
    negatives = [
        c for c in candidates
        if c['relevance_score'] <= negative_threshold
    ]

    # No pairs possible if either side is empty
    if not positives or not negatives:
        return []

    # Cap the number of positives to avoid queries with many positives
    # dominating the training set
    if len(positives) > max_positives_per_query:
        positives = random.sample(positives, max_positives_per_query)

    # Run pretrained model on negatives to identify hard negatives
    scored_negatives, _ = rerank(
        query=query,
        candidates=negatives,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device
    )
    # scored_negatives is sorted by predicted_score descending
    # top-n are the hard negatives, the rest are candidates for random sampling
    hard_negatives = scored_negatives[:n_hard_negatives]
    soft_negatives = scored_negatives[n_hard_negatives:]

    pairs = []
    for positive in positives:
        # Pair with hard negatives
        for hard_neg in hard_negatives:
            pairs.append({
                'query': query,
                'positive': positive['sentence'],
                'negative': hard_neg['sentence']
            })

        # Pair with random negatives sampled from soft pool
        n_to_sample = min(n_random_negatives, len(soft_negatives))
        if n_to_sample > 0:
            sampled = random.sample(soft_negatives, n_to_sample)
            for neg in sampled:
                pairs.append({
                    'query': query,
                    'positive': positive['sentence'],
                    'negative': neg['sentence']
                })

    return pairs