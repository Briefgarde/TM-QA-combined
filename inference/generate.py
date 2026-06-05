import torch
from tqdm import tqdm
import time

import string
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
import numpy as np

# this is needed for register_bertscore_model
import sys
import bert_score.utils as bert_score_utils
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

class NewlineStoppingCriteria(StoppingCriteria):
    """Stops generation when a newline token is produced."""
    def __init__(self, newline_token_ids: list[int]):
        self.newline_token_ids = set(newline_token_ids)

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() in self.newline_token_ids


def generate(
    context_entry: dict,
    model,
    tokenizer,
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.2,
    use_newline_stop: bool = True,
    stopping_criteria:StoppingCriteriaList=None,
) -> dict:
    """
    Generates an answer for a single context entry using greedy decoding.

    Args:
        context_entry      : Single output from assemble_context(), carrying
                             id, query, answer, prompt.
        model              : Loaded causal LM.
        tokenizer          : Associated tokenizer.
        max_new_tokens     : Maximum number of tokens to generate.
        repetition_penalty : Penalty applied to already-seen tokens to reduce
                             repetition loops.
        use_newline_stop   : If True, stops generation at the first newline token.

    Returns:
        Dict with keys:
            'id'              : Query ID.
            'query'           : Query string.
            'answer'          : Gold answer.
            'generated'       : Generated answer string.
            'prompt_tokens'   : Number of tokens in the prompt.
            'generated_tokens': Number of tokens generated.
    """
    prompt = context_entry['prompt']

    if use_newline_stop and stopping_criteria is None:
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        newline_ids += tokenizer.encode("\n\n", add_special_tokens=False)
        stopping_criteria = StoppingCriteriaList([
            NewlineStoppingCriteria(newline_ids)
        ])

    #timing including tokenizing the prompt and generation
    start = time.perf_counter()
    
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=tokenizer.model_max_length
    ).to(model.device)
    
    
    if use_newline_stop:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )
    else: 
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    prompt_length = inputs['input_ids'].shape[1]
    generated_ids = output_ids[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        'id': context_entry['id'],
        'query': context_entry['query'],
        'answer': context_entry['answer'],
        'generated': generated_text,
        'prompt_tokens': prompt_length,
        'generated_tokens': len(generated_ids),
        'generation_time_ms': elapsed_ms,
    }


def generate_dataset(
    contexts: list[dict],
    model,
    tokenizer,
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.2,
    use_newline_stop: bool = True
) -> list[dict]:
    """
    Generates answers for the full context dataset.

    Args:
        contexts           : Output of assemble_context_dataset().
        model              : Loaded causal LM.
        tokenizer          : Associated tokenizer.
        max_new_tokens     : Maximum number of tokens to generate.
        repetition_penalty : Penalty applied to already-seen tokens.
        use_newline_stop   : If True, stops generation at the first newline token.

    Returns:
        List of generation result dicts, one per query, in the same order as contexts.
    """
    stopping_criteria = None
    if use_newline_stop:
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        newline_ids += tokenizer.encode("\n\n", add_special_tokens=False)
        stopping_criteria = StoppingCriteriaList([
            NewlineStoppingCriteria(newline_ids)
        ])
    results = []
    for entry in tqdm(contexts, desc="Generating answers"):
        results.append(generate(
            context_entry=entry,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            use_newline_stop=use_newline_stop,
            stopping_criteria=stopping_criteria
        ))
    return results



def normalize_text(text: str) -> list[str]:
    """
    Lowercases, strips punctuation, and splits into words.
    Used for EM and token-level F1.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()


def compute_em(generated: str, gold: str) -> float:
    return float(normalize_text(generated) == normalize_text(gold))


def compute_token_f1(generated: str, gold: str) -> float:
    generated_tokens = normalize_text(generated)
    gold_tokens = normalize_text(gold)

    if not generated_tokens or not gold_tokens:
        return 0.0

    generated_counts = {}
    for t in generated_tokens:
        generated_counts[t] = generated_counts.get(t, 0) + 1

    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    overlap = sum(min(generated_counts.get(t, 0), gold_counts[t]) for t in gold_counts)

    if overlap == 0:
        return 0.0

    precision = overlap / len(generated_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_generation(
    generation_entry: dict,
    rouge_scorer_obj,
    bertscore_results: dict,
    entry_index: int
) -> dict:
    """
    Evaluates a single generation result.

    Args:
        generation_entry  : Single output from generate(), carrying id, query,
                            answer, generated, prompt_tokens, generated_tokens.
        rouge_scorer_obj  : Instantiated rouge_score.rouge_scorer.RougeScorer.
        bertscore_results : Precomputed BertScore results dict for the full dataset,
                            keyed by index. BertScore is computed in batch at dataset
                            level for efficiency.
        entry_index       : Index of this entry in the dataset, used to fetch
                            the correct BertScore from bertscore_results.

    Returns:
        Dict of metric names to float values, plus context metadata.
    """
    generated = generation_entry['generated']
    gold = generation_entry['answer']

    # BioASQ ideal_answer is a list, take the first element
    # normally this shouldn't be needed, I've flattened it earlier. 
    if isinstance(gold, list):
        gold = gold[0]

    rouge_scores = rouge_scorer_obj.score(gold, generated)

    results = {
        'id': generation_entry['id'],
        'query': generation_entry['query'],
        'generated' : generated,
        'answer' : gold,
        'EM': compute_em(generated, gold),
        'F1': compute_token_f1(generated, gold),
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure,
        'BertScore_roberta_P': bertscore_results['roberta']['P'][entry_index],
        'BertScore_roberta_R': bertscore_results['roberta']['R'][entry_index],
        'BertScore_roberta_F1': bertscore_results['roberta']['F1'][entry_index],
        'BertScore_biomedical_P': bertscore_results['biomedical']['P'][entry_index],
        'BertScore_biomedical_R': bertscore_results['biomedical']['R'][entry_index],
        'BertScore_biomedical_F1': bertscore_results['biomedical']['F1'][entry_index],
        'prompt_tokens': generation_entry['prompt_tokens'],
        'generated_tokens': generation_entry['generated_tokens'],
        'generation_time_ms': generation_entry['generation_time_ms']
    }

    return results

#this is a monkey patch for something that kept throwing an overflow error
def register_bertscore_model(model_name: str, num_layers: int = 12, max_length: int = 512) -> None:
    """
    Registers an unsupported model in bert_score's internal lookup table
    and clamps its tokenizer's model_max_length if it carries a sentinel value.
    """
    # Register in the lookup table so bert_score does not raise KeyError
    if model_name not in bert_score_utils.model2layers:
        bert_score_utils.model2layers[model_name] = num_layers

    # Patch the tokenizer class to clamp model_max_length at load time
    original_from_pretrained = AutoTokenizer.from_pretrained.__func__

    def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        tokenizer = original_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
        if (pretrained_model_name_or_path == model_name
                and tokenizer.model_max_length > sys.maxsize):
            tokenizer.model_max_length = max_length
        return tokenizer

    AutoTokenizer.from_pretrained = classmethod(patched_from_pretrained)






def evaluate_generation_dataset(
    generation_results: list[dict],
    bertscore_model_roberta: str = "roberta-large",
    bertscore_model_biomedical: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    bertscore_biomedical_num_layers: int = 12,
    device: str = "cpu"
) -> tuple[list[dict], dict]:
    """
    Evaluates generation results over the full dataset.

    Args:
        generation_results        : Output of generate_dataset().
        bertscore_model_roberta   : HuggingFace model name for roberta BertScore.
        bertscore_model_biomedical: HuggingFace model name for biomedical BertScore.
        bertscore_biomedical_num_layers: Number of layers for the biomedical model.
        device                    : Device for BertScore computation.

    Returns:
        per_query_results : List of per-query metric dicts.
        averaged          : Dict of macro-averaged metrics across queries.
    """
    generated_texts = [e['generated'] for e in generation_results]
    gold_texts = [
        e['answer'][0] if isinstance(e['answer'], list) else e['answer']
        for e in generation_results
    ]

    # Register biomedical model in bert_score's lookup table and patch
    # its tokenizer to avoid OverflowError from sentinel model_max_length
    register_bertscore_model(bertscore_model_biomedical, num_layers=bertscore_biomedical_num_layers)

    # BERTscore can be batched for more efficiency
    print(f"Computing BertScore with {bertscore_model_roberta}...")
    P_rob, R_rob, F1_rob = bert_score_fn(
        generated_texts, gold_texts,
        model_type=bertscore_model_roberta,
        device=device,
        verbose=False
    )

    print(f"Computing BertScore with {bertscore_model_biomedical}...")
    P_bio, R_bio, F1_bio = bert_score_fn(
        generated_texts, gold_texts,
        model_type=bertscore_model_biomedical,
        num_layers=bertscore_biomedical_num_layers,
        device=device,
        verbose=False
    )

    bertscore_results = {
        'roberta': {
            'P': P_rob.tolist(),
            'R': R_rob.tolist(),
            'F1': F1_rob.tolist()
        },
        'biomedical': {
            'P': P_bio.tolist(),
            'R': R_bio.tolist(),
            'F1': F1_bio.tolist()
        }
    }

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False
    )

    per_query_results = [
        evaluate_generation(entry, scorer, bertscore_results, i)
        for i, entry in enumerate(tqdm(generation_results, desc="Evaluating generation"))
    ]

    numeric_keys = [
        k for k in per_query_results[0].keys()
        if k not in ('id', 'query', 'generated', 'answer')
    ]
    averaged = {
        metric: float(np.mean([r[metric] for r in per_query_results]))
        for metric in numeric_keys
    }

    return per_query_results, averaged