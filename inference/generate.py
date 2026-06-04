import torch
from tqdm import tqdm


def generate(
    context_entry: dict,
    model,
    tokenizer,
    max_new_tokens: int = 256,
    repetition_penalty:float=1.2
) -> dict:
    """
    Generates an answer for a single context entry using greedy decoding.

    Args:
        context_entry  : Single output from assemble_context(), carrying
                         id, query, answer, prompt.
        model          : Loaded causal LM.
        tokenizer      : Associated tokenizer.
        repetition_penalty : a float that indicate how strongly the model should be penalized for repeating sentences. 
        max_new_tokens : Maximum number of tokens to generate.

    Returns:
        Dict with keys:
            'id'             : Query ID.
            'query'          : Query string.
            'answer'         : Gold answer.
            'generated'      : Generated answer string.
            'prompt_tokens'  : Number of tokens in the prompt.
            'generated_tokens: Number of tokens generated.
    """
    prompt = context_entry['prompt']

    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=tokenizer.model_max_length
    ).to(model.device)

    prompt_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = output_ids[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        'id': context_entry['id'],
        'query': context_entry['query'],
        'answer': context_entry['answer'],
        'generated': generated_text,
        'prompt_tokens': prompt_length,
        'generated_tokens': len(generated_ids)
    }


def generate_dataset(
    contexts: list[dict],
    model,
    tokenizer,
    max_new_tokens: int = 256,
    repetition_penalty:float=1.2
) -> list[dict]:
    """
    Generates answers for the full context dataset.

    Args:
        contexts       : Output of assemble_context_dataset().
        model          : Loaded causal LM.
        tokenizer      : Associated tokenizer.
        max_new_tokens : Maximum number of tokens to generate.

    Returns:
        List of generation result dicts, one per query, in the same order as contexts.
    """
    results = []
    for entry in tqdm(contexts, desc="Generating answers"):
        results.append(generate(
            context_entry=entry,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty
        ))
    return results