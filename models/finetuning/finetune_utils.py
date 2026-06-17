import sys
sys.path.append('../../')

import random

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset

from inference.rerank import rerank, evaluate_reranker_dataset


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


class RerankerDataset(Dataset):
    """
    Wraps (query, positive, negative) training triplets.

    Tokenization is deferred to the collate function to allow dynamic
    padding per batch rather than padding every item to a fixed length.
    """
    def __init__(self, pairs: list[dict]):
        """
        Args:
            pairs : List of dicts with 'query', 'positive', 'negative' keys,
                    as produced by build_training_pairs.
        """
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def make_collate_fn(tokenizer, max_length: int = 512):
    """
    Returns a collate function that tokenizes (query, sentence) pairs
    for both positive and negative sentences in a batch.

    Args:
        tokenizer  : Reranker tokenizer.
        max_length : Maximum sequence length for truncation.

    Returns:
        A collate function suitable for torch.utils.data.DataLoader.
    """
    def collate_fn(batch: list[dict]):
        queries = [item['query'] for item in batch]
        positives = [item['positive'] for item in batch]
        negatives = [item['negative'] for item in batch]

        pos_encoded = tokenizer(
            queries, positives,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        neg_encoded = tokenizer(
            queries, negatives,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        return pos_encoded, neg_encoded

    return collate_fn


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler=None,
    margin: float = 1.0,
    max_grad_norm: float = 1.0,
    device: str = None
) -> float:
    """
    Runs one training epoch over the dataloader using margin ranking loss.

    Args:
        model         : Reranker model (will be set to train mode).
        dataloader    : DataLoader yielding (pos_encoded, neg_encoded) batches
                        from make_collate_fn.
        optimizer     : Optimizer (e.g. AdamW).
        scheduler     : Optional learning rate scheduler, stepped per batch.
        margin        : Margin for margin ranking loss.
        max_grad_norm : Maximum gradient norm for clipping. Set to None to disable.
        device        : Device to run on. If None, inferred from model.

    Returns:
        Average loss over the epoch.
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    total_loss = 0.0

    for pos_encoded, neg_encoded in tqdm(dataloader, desc="Training"):
        pos_encoded = pos_encoded.to(device)
        neg_encoded = neg_encoded.to(device)

        pos_scores = model(**pos_encoded).logits.squeeze(-1)
        neg_scores = model(**neg_encoded).logits.squeeze(-1)

        target = torch.ones_like(pos_scores)
        loss = F.margin_ranking_loss(pos_scores, neg_scores, target, margin=margin)

        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)




def validate(
    model,
    tokenizer,
    val_pool: list[dict],
    threshold: float = 0.5,
    k_values: list[int] = [5, 10, 15, 20],
    batch_size: int = 16,
    device: str = None
) -> dict:
    """
    Evaluates the reranker on the validation pool using the existing
    rerank and evaluate_reranker_dataset functions.

    Args:
        model      : Reranker model (will be set to eval mode).
        tokenizer  : Associated tokenizer.
        val_pool   : Validation candidate pool, list of entries with
                     'query' and 'candidates'.
        threshold  : Relevance score threshold for binary labels.
        k_values   : k values for @k metrics.
        batch_size : Batch size for rerank inference.
        device     : Device for inference. If None, inferred from model.

    Returns:
        Dict of macro-averaged metrics, same structure as
        evaluate_reranker_dataset's output (includes NDCG@k, MAP, MRR,
        P@k, R@k, F1@k, avg_reranker_time_ms, avg_reranker_input_tokens).
    """
    model.eval()

    scored_list = []
    metadata_list = []

    for entry in tqdm(val_pool, desc="Validating"):
        ranked, meta = rerank(
            query=entry['query'],
            candidates=entry['candidates'],
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device
        )
        scored_list.append(ranked)
        metadata_list.append(meta)

    metrics = evaluate_reranker_dataset(
        scored_list, metadata_list,
        threshold=threshold,
        k_values=k_values
    )

    return metrics



def finetune_reranker(
    model,
    tokenizer,
    dataset,
    val_pool: list[dict],
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    warmup_ratio: float = 0.1,
    margin: float = 1.0,
    max_grad_norm: float = 1.0,
    threshold: float = 0.5,
    k_values: list[int] = [5, 10, 15, 20],
    val_batch_size: int = 16,
    early_stopping_patience: int = 2,
    device: str = None
) -> dict:
    """
    Finetunes the reranker with margin ranking loss, validating after each
    epoch and saving the best checkpoint by validation NDCG@10.

    Args:
        model                  : Reranker model.
        tokenizer              : Associated tokenizer.
        dataset                : RerankerDataset wrapping training pairs.
        val_pool               : Validation candidate pool.
        output_dir             : Directory to save the best checkpoint and history.
        epochs                 : Number of training epochs.
        batch_size             : Training batch size.
        learning_rate          : Peak learning rate for AdamW.
        warmup_ratio           : Fraction of total steps used for linear warmup.
        margin                 : Margin for margin ranking loss.
        max_grad_norm          : Gradient clipping norm.
        threshold              : Relevance score threshold for validation labels.
        k_values               : k values for validation @k metrics.
        val_batch_size         : Batch size for validation rerank inference.
        early_stopping_patience: Number of epochs without NDCG@10 improvement
                                 before stopping early.
        device                 : Device to run on. If None, inferred from model.

    Returns:
        Dict with 'history' (list of per-epoch metrics) and 'best_epoch'.
    """
    if device is None:
        device = next(model.parameters()).device

    os.makedirs(output_dir, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer)
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    history = []
    best_ndcg = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")

        train_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            margin=margin,
            max_grad_norm=max_grad_norm,
            device=device
        )

        val_metrics = validate(
            model=model,
            tokenizer=tokenizer,
            val_pool=val_pool,
            threshold=threshold,
            k_values=k_values,
            batch_size=val_batch_size,
            device=device
        )

        ndcg_key = f"NDCG@{k_values[1] if len(k_values) > 1 else k_values[0]}" # this is to grab @10, 
        current_ndcg = val_metrics.get(ndcg_key, val_metrics.get(f"NDCG@{k_values[0]}"))

        print(f"Train loss: {train_loss:.4f} | Val {ndcg_key}: {current_ndcg:.4f}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": val_metrics
        }
        history.append(epoch_record)

        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_epoch = epoch
            epochs_without_improvement = 0

            best_path = os.path.join(output_dir, "best_model")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"New best model saved at epoch {epoch} ({ndcg_key}={current_ndcg:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement ({epochs_without_improvement}/{early_stopping_patience})")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_ndcg": best_ndcg
    }