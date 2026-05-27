from datasets import Dataset
import json
import re
from sklearn.model_selection import train_test_split
import string

def load_bioASQ(
    abstracts_path: str,
    bioasq_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple[Dataset, Dataset, Dataset, dict]:
    """
    Loads and preprocesses the BioASQ dataset and the tokenized abstracts.

    Args:
        abstracts_path : Path to the abstract_list_tokenized.json file.
        bioasq_path    : Path to the BioASQ training JSON file.
        train_ratio    : Fraction of data for training.
        val_ratio      : Fraction of data for validation.
        test_ratio     : Fraction of data for testing.
        seed           : Random seed for reproducibility.

    Returns:
        train_dataset  : HF Dataset for training.
        val_dataset    : HF Dataset for validation.
        test_dataset   : HF Dataset for testing.
        abstracts      : Raw dict mapping PubMed ID (str) to list of sentences.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio, val_ratio and test_ratio must sum to 1."

    with open(abstracts_path, 'r', encoding='utf-8') as f:
        abstracts = json.load(f)

    with open(bioasq_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    questions = raw['questions']

    entries = []
    for q in questions:
        pubmed_ids = [
            match[0]
            for link in q.get('documents', [])
            for match in [re.findall(r"pubmed\/(\d+)", link)]
            if match
        ]

        # Filter to IDs that are actually present in the tokenized abstracts
        available_ids = [pid for pid in pubmed_ids if pid in abstracts]

        if not available_ids:
            continue

        snippets = [
            {
                "document_id": re.findall(r"pubmed\/(\d+)", s["document"])[0],
                "start": s["offsetInBeginSection"],
                "end": s["offsetInEndSection"],
                "text": s["text"]
            }
            for s in q.get("snippets", [])
            if re.findall(r"pubmed\/(\d+)", s["document"])
        ]

        entries.append({
            "id": q["id"],
            "query": q["body"],
            "pubmed_ids": available_ids,
            "snippets": snippets
        })

    # First split off the test set, then split remainder into train/val
    test_size = test_ratio
    val_size = val_ratio / (train_ratio + val_ratio)

    train_val, test = train_test_split(entries, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed)

    train_dataset = Dataset.from_list(train)
    val_dataset = Dataset.from_list(val)
    test_dataset = Dataset.from_list(test)

    print(f"Dataset loaded: {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test entries.")
    print(f"Entries dropped (no matching abstracts): {len(questions) - len(entries)}")

    return train_dataset, val_dataset, test_dataset, abstracts




def normalize(text: str) -> list[str]:
    """
    Lowercases, strips punctuation, collapses whitespace, and splits into words.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()


def lcs_length(words_a: list[str], words_b: list[str]) -> int:
    """
    Computes the length of the Longest Common Substring between two word lists.
    """
    n, m = len(words_a), len(words_b)
    best = 0
    # dp[i][j] = length of longest common substring ending at words_a[i-1], words_b[j-1]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if words_a[i - 1] == words_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])
            else:
                dp[i][j] = 0
    return best


def compute_label(sentence_words: list[str], snippets: list[dict]) -> float:
    """
    Returns a float relevance score in [0, 1] representing the best overlap
    between the sentence and any snippet in the query.

    Score is computed as max(lcs / len(snippet_words), lcs / len(sentence_words))
    across all snippets, taking the maximum across snippets.

    Args:
        sentence_words : Normalized word list of the candidate sentence.
        snippets       : List of snippet dicts with a 'text' key (pre-filtered
                         to available abstracts upstream).

    Returns:
        Float in [0, 1]. Returns 0.0 if no snippets are available.
    """
    best = 0.0
    for snippet in snippets:
        snippet_words = normalize(snippet['text'])
        if not snippet_words:
            continue
        lcs = lcs_length(sentence_words, snippet_words)
        score = max(lcs / len(snippet_words), lcs / len(sentence_words))
        best = max(best, score)
    return best


def build_candidate_pool_bioASQ(
    entry: dict,
    abstracts: dict,
) -> dict:
    """
    Builds the candidate pool for a single query entry.

    Args:
        entry     : A single dataset entry with keys 'id', 'query', 'pubmed_ids', 'snippets'.
        abstracts : Dict mapping PubMed ID (str) to list of sentence strings.
        threshold : Minimum fraction of snippet covered by LCS to assign label 1.

    Returns:
        A dict with:
            'id'         : Query ID.
            'query'      : Query string.
            'candidates' : List of {'sentence': str, 'label': int}.
    """
    # Filter snippets to those whose source abstract is available
    valid_snippets = [
        s for s in entry['snippets']
        if s['document_id'] in abstracts
    ]

    candidates = []
    for pid in entry['pubmed_ids']:
        if pid not in abstracts:
            continue
        for sentence in abstracts[pid]:
            sentence_words = normalize(sentence)
            if not sentence_words:
                continue
            label = compute_label(sentence_words, valid_snippets)
            candidates.append({
                'sentence': sentence,
                'relevance_score': label
            })

    return {
        'id': entry['id'],
        'query': entry['query'],
        'candidates': candidates
    }