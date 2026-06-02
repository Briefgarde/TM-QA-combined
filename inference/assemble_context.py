from tqdm import tqdm

def assemble_context(
    pool_entry: dict,
    scored: list[dict],
    abstracts: dict,
    tokenizer,
    mode: str = "reranked",
    top_k: int = 10,
    max_tokens: int = 1024,
    prompt_template: str = "Context: {context}\nQuestion: {query}\nAnswer:"
) -> dict:
    """
    Assembles the prompt context for a single query.

    Args:
        pool_entry      : Single entry from the candidate pool, carrying id, query,
                          answer and pubmed_ids.
        scored          : Reranked candidate list from rerank(). Required for
                          mode='reranked', ignored for mode='full'.
        abstracts       : Full abstracts dict mapping PubMed ID to sentence list.
                          Required for mode='full', ignored for mode='reranked'.
        tokenizer       : GenLM tokenizer, used for token counting.
        mode            : 'reranked' uses top_k sentences from scored in reranked order.
                          'full' uses all sentences from abstracts in original order.
        top_k           : Number of top sentences to use in reranked mode.
        max_tokens      : Maximum number of tokens allowed for the full prompt.
                          Sentences are truncated to fit within this budget.
        prompt_template : Format string with {context} and {query} placeholders.

    Returns:
        Dict with keys:
            'id'            : Query ID.
            'query'         : Query string.
            'answer'        : Gold answer string(s).
            'prompt'        : Fully assembled prompt string.
            'context_tokens': Token count of the assembled prompt.
            'n_sentences'   : Number of sentences included in the context.
            'mode'          : Mode used for assembly.
    """
    query_id = pool_entry['id']
    query = pool_entry['query']
    answer = pool_entry['answer']

    if mode == "reranked":
        sentence_pool = [c['sentence'] for c in scored[:top_k]]
    elif mode == "full":
        sentence_pool = [
            sentence
            for pid in pool_entry['pubmed_ids']
            if pid in abstracts
            for sentence in abstracts[pid]
        ]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'reranked' or 'full'.")

    # Greedily add sentences until the prompt would exceed max_tokens
    included_sentences = []
    for sentence in sentence_pool:
        candidate_context = " ".join(included_sentences + [sentence])
        candidate_prompt = prompt_template.format(
            context=candidate_context,
            query=query
        )
        token_count = len(tokenizer.encode(candidate_prompt, add_special_tokens=False))
        if token_count > max_tokens:
            break
        included_sentences.append(sentence)

    final_context = " ".join(included_sentences)
    final_prompt = prompt_template.format(context=final_context, query=query)
    final_token_count = len(tokenizer.encode(final_prompt, add_special_tokens=False))

    return {
        'id': query_id,
        'query': query,
        'answer': answer,
        'prompt': final_prompt,
        'context_tokens': final_token_count,
        'n_sentences': len(included_sentences),
        'mode': mode
    }


def assemble_context_dataset(
    pool: list[dict],
    scored_list: list[list[dict]],
    abstracts: dict,
    tokenizer,
    mode: str = "reranked",
    top_k: int = 10,
    max_tokens: int = 1024,
    prompt_template: str = "Context: {context}\nQuestion: {query}\nAnswer:"
) -> list[dict]:
    """
    Assembles prompt contexts for the full dataset.

    Args:
        pool        : Full candidate pool (train_pool, test_pool, or val_pool).
        scored_list : List of rerank() outputs aligned with pool by index.
                      Pass None for mode='full'.
        abstracts   : Full abstracts dict.
        tokenizer   : GenLM tokenizer.
        mode        : 'reranked' or 'full'.
        top_k       : Number of top sentences for reranked mode.
        max_tokens  : Token budget for the full prompt.
        prompt_template : Format string with {context} and {query} placeholders.

    Returns:
        List of context dicts, one per query, in the same order as pool.
    """
    if mode == "reranked" and scored_list is None:
        raise ValueError("scored_list must be provided for mode='reranked'.")

    if mode == "full":
        scored_list = [None] * len(pool)

    contexts = []
    for pool_entry, scored in tqdm(
        zip(pool, scored_list),
        total=len(pool),
        desc=f"Assembling context ({mode})"
    ):
        contexts.append(assemble_context(
            pool_entry=pool_entry,
            scored=scored,
            abstracts=abstracts,
            tokenizer=tokenizer,
            mode=mode,
            top_k=top_k,
            max_tokens=max_tokens,
            prompt_template=prompt_template
        ))

    return contexts