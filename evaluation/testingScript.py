import sys
sys.path.append('../')
# so that it sits at the root of the program. 

from data.load_dataset import load_bioASQ, build_candidate_pool_bioASQ
from models.load_reranker import load_reranker, load_genLM
from saveResult import save_results_reranker, save_result_generation
from inference.rerank import rerank, evaluate_reranker_dataset
from inference.generate import generate_dataset, generate_dataset_batch, evaluate_generation_dataset
from inference.assemble_context import assemble_context_dataset
import json
from tqdm import tqdm
import os
from pprint import pp
import torch


def env_bool(name: str, default: bool) -> bool:
    """
    Reads an environment variable as a boolean.
    Env variables are always strings, and bool("false") / bool("0") is True in Python
    (any non-empty string is truthy), so the string has to be compared explicitly.
    Accepts true/1/yes and false/0/no (case-insensitive); unset or empty -> default.
    """
    value = os.environ.get(name)
    if not value:
        return default
    value = value.strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise ValueError(f"Env variable {name} must be true/false/1/0/yes/no, got '{value}'")


DEBUG_N = int(os.environ["DEBUG_N"]) if os.environ.get("DEBUG_N") else None
USE_PRECOMPUTE_FILE = env_bool("USE_PRECOMPUTE_FILE", True)
POOL_PATH = str(os.environ["POOL_PATH"]) if os.environ.get("POOL_PATH") else None
DATASET = str(os.environ["DATASET"]) if os.environ.get("DATASET") else "unknown" # e.g. BioASQ-training14b, PubMedQA-oracle, PubMedQA-rag
SPLIT = str(os.environ["SPLIT"]) if os.environ.get("SPLIT") else "test"

print(f"{DEBUG_N=}")
print(f"{USE_PRECOMPUTE_FILE=}")
print(f"{POOL_PATH=}")
print(f"{DATASET=}")
print(f"{SPLIT=}")



#----------------#
# Reranking part

print("starting reranking part")

MODEL_NAME_RERANKER = str(os.environ["MODEL_NAME_RERANKER"]) if os.environ.get("MODEL_NAME_RERANKER") else None
TEST_RERANKER = env_bool("TEST_RERANKER", False)
OUTPUT_DIR_RERANKER_RESULT = str(os.environ["OUTPUT_DIR_RERANKER_RESULT"]) if os.environ.get("OUTPUT_DIR_RERANKER_RESULT") else None

print(TEST_RERANKER)
print(OUTPUT_DIR_RERANKER_RESULT)
print(MODEL_NAME_RERANKER)

modelName = MODEL_NAME_RERANKER

model, tokenizer = load_reranker(model_path_or_name=modelName)

test_pool_path = POOL_PATH

if USE_PRECOMPUTE_FILE: # this default to true if not provided
    print("Pre-computed pools found. Loading from JSON files...")
    
    with open(test_pool_path, "r", encoding="utf-8") as test_p_json:
        test_pool = json.load(test_p_json)

pool_slice = test_pool if DEBUG_N is None else test_pool[:DEBUG_N]

queries = [pool_slice[i]['query'] for i in range(len(pool_slice))]
candidates_list = [pool_slice[i]['candidates'] for i in range(len(pool_slice))]

scoreTest = []
metadataTest = []
for q, cand in tqdm(zip(queries, candidates_list), total=len(queries), desc="Reranking"):
    ranked, meta = rerank(query=q, candidates=cand, model=model, tokenizer=tokenizer)
    scoreTest.append(ranked)
    metadataTest.append(meta)

k_values=[5,10,15,20] # those are for NDCG@k, P@k, R@k et F1@k, not for the number of sentence included
output_dir=OUTPUT_DIR_RERANKER_RESULT

# this default to FALSE if not provided
# since some ds do not have a relevance score, we can't always test on it
# only bioASQ has a relevance score, PubMedQA does NOT
if TEST_RERANKER: 
    thresholds_to_test = [0.3, 0.5, 0.7] 
    for threshold in thresholds_to_test: 
        metrics = evaluate_reranker_dataset(
            scoreTest, metadataTest,
            threshold=threshold,
            k_values=k_values
        )
        save_results_reranker(
            scored=scoreTest,
            metadata=metadataTest,
            metrics=metrics,
            model_path_or_name=modelName,
            threshold=threshold,
            k_values=k_values,
            output_dir=output_dir
        )

del model, tokenizer # this is only for the reranker
torch.cuda.empty_cache()

#--------------#
# gen part
print("starting gen part")

MODEL_NAME_GEN = str(os.environ["MODEL_NAME_GEN"]) if os.environ.get("MODEL_NAME_GEN") else "stanford-crfm/BioMedLM"
MODE_CONTEXT_ASSEMBLING = str(os.environ["MODE_CONTEXT_ASSEMBLING"]) if os.environ.get("MODE_CONTEXT_ASSEMBLING") else "reranked"
MAX_NEW_TOKEN = int(os.environ["MAX_NEW_TOKEN"]) if os.environ.get("MAX_NEW_TOKEN") else 100
MAX_CONTEXT_TOKEN = int(os.environ["MAX_CONTEXT_TOKEN"]) if os.environ.get("MAX_CONTEXT_TOKEN") else 1024
PROMPT_TEMPLATE = str(os.environ["PROMPT_TEMPLATE"]) if os.environ.get("PROMPT_TEMPLATE") else "Context: {context}\nQuestion: {query}\nAnswer:"
REPETITION_PENALTY = float(os.environ["REPETITION_PENALTY"]) if os.environ.get("REPETITION_PENALTY") else 1.2
USE_NEWLINE_STOP = env_bool("USE_NEWLINE_STOP", True)
BATCH_SIZE_GEN = int(os.environ["BATCH_SIZE_GEN"]) if os.environ.get("BATCH_SIZE_GEN") else 4
BERTSCORE_MODEL_ROBERTA = str(os.environ["BERTSCORE_MODEL_ROBERTA"]) if os.environ.get("BERTSCORE_MODEL_ROBERTA") else "roberta-large"
BERTSCORE_MODEL_BIOMEDICAL = str(os.environ["BERTSCORE_MODEL_BIOMEDICAL"]) if os.environ.get("BERTSCORE_MODEL_BIOMEDICAL") else "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
OUTPUT_DIR_GEN = str(os.environ["OUTPUT_DIR_GEN"]) if os.environ.get("OUTPUT_DIR_GEN") else "result/TestOraclePubMedQA"

print(MODEL_NAME_GEN)
print(MODE_CONTEXT_ASSEMBLING)
print(MAX_NEW_TOKEN)
print(MAX_CONTEXT_TOKEN)
print(PROMPT_TEMPLATE)
print(REPETITION_PENALTY)
print(USE_NEWLINE_STOP)
print(BATCH_SIZE_GEN)
print(BERTSCORE_MODEL_ROBERTA)
print(BERTSCORE_MODEL_BIOMEDICAL)
print(OUTPUT_DIR_GEN)

modelGenName = MODEL_NAME_GEN 
# default stanford-crfm/BioMedLM
modeContextAssembling = MODE_CONTEXT_ASSEMBLING  
# can be reranked, to use the top-k sentence from the abstract,
# or full, which uses all of the abstracts as context. 
# default reranked
maxNewToken = MAX_NEW_TOKEN # token reserved, out of 1024, for generation
maxContextToken = MAX_CONTEXT_TOKEN-maxNewToken # the leftover token can be used for context. 
# the default 1024 for BioMedLM, but can be adjusted if using other model. 
promptTemplate = PROMPT_TEMPLATE 
repetitionPenalty=REPETITION_PENALTY # this penalize the LM for getting stuck in a loop of the exact same sentence. 
# This is likely to happen in greedy decoding setup, since there's no variability. 
useNewlineStop = USE_NEWLINE_STOP # default to true
batchSizeGen = BATCH_SIZE_GEN
bertscoreModelRoberta = BERTSCORE_MODEL_ROBERTA
bertscoreModelBiomedical = BERTSCORE_MODEL_BIOMEDICAL
outputDirGen = OUTPUT_DIR_GEN

modelGen, tokenGen = load_genLM(model_path_or_name=modelGenName)


if modeContextAssembling == "reranked":
    top_k_values_to_test = [1,2,3,5] # this is for the number of sentence to include
    # TODO ^ this needs to be aligned with the paper
    for top_k in top_k_values_to_test:
            contexts = assemble_context_dataset(
                pool=pool_slice,
                scored_list=scoreTest,
                abstracts=None,
                tokenizer=tokenGen,
                mode=modeContextAssembling,
                top_k=top_k,
                max_context_tokens=maxContextToken,
                prompt_template=promptTemplate
            )


            generated_sets = generate_dataset_batch(
                contexts=contexts,
                model=modelGen,
                tokenizer=tokenGen,
                max_new_tokens=maxNewToken,
                repetition_penalty=repetitionPenalty,
                use_newline_stop=useNewlineStop,
                batch_size=batchSizeGen
            )

            per_query_results, averaged = evaluate_generation_dataset(
                generation_results=generated_sets,
                bertscore_model_roberta=bertscoreModelRoberta,
                bertscore_model_biomedical=bertscoreModelBiomedical
            )

            metadataPipeline = {
                "dataset": DATASET,
                "split": SPLIT,
                "pool_path": POOL_PATH,
                "debug_n": DEBUG_N,
                "n_queries": len(pool_slice),
                "rerankingMetadata": {
                    "modelReranker": modelName,
                },
                "generationMetadata": {
                    "modelGenName": modelGenName,
                    "modeContextAssembling": "reranked",
                    "topk": top_k,
                    "max_new_token": maxNewToken,
                    "maxContextToken": maxContextToken,
                    "prompt_template": promptTemplate,
                    "repetition_penalty": repetitionPenalty,
                    "bertscore_model_roberta": bertscoreModelRoberta,
                    "bertscore_model_biomedical": bertscoreModelBiomedical,
                    "use_newline_stop": useNewlineStop,
                    "batch_size": batchSizeGen
                }
            }

            save_result_generation(
                per_query_results=per_query_results,
                averaged=averaged,
                metadata=metadataPipeline,
                model_path_or_name=modelGenName,
                output_dir=outputDirGen
            )
elif modeContextAssembling == "full":
    top_k = 0
    contexts = assemble_context_dataset(
        pool=pool_slice,
        scored_list=scoreTest,
        abstracts=None,
        tokenizer=tokenGen,
        mode="full",
        top_k=top_k,
        max_context_tokens=maxContextToken,
        prompt_template=promptTemplate
    )
    generated_sets = generate_dataset_batch(
        contexts=contexts,
        model=modelGen,
        tokenizer=tokenGen,
        max_new_tokens=maxNewToken,
        repetition_penalty=repetitionPenalty,
        use_newline_stop=useNewlineStop,
        batch_size=batchSizeGen
    )

    per_query_results, averaged = evaluate_generation_dataset(
                generation_results=generated_sets,
                bertscore_model_roberta=bertscoreModelRoberta,
                bertscore_model_biomedical=bertscoreModelBiomedical
            )

    metadataPipeline = {
                "dataset": DATASET,
                "split": SPLIT,
                "pool_path": POOL_PATH,
                "debug_n": DEBUG_N,
                "n_queries": len(pool_slice),
                "rerankingMetadata": {
                    "modelReranker": None, # full mode does not use the reranker output
                },
                "generationMetadata": {
                    "modelGenName": modelGenName,
                    "modeContextAssembling": "full",
                    "topk": top_k,
                    "max_new_token": maxNewToken,
                    "maxContextToken": maxContextToken,
                    "prompt_template": promptTemplate,
                    "repetition_penalty": repetitionPenalty,
                    "bertscore_model_roberta": bertscoreModelRoberta,
                    "bertscore_model_biomedical": bertscoreModelBiomedical,
                    "use_newline_stop": useNewlineStop,
                    "batch_size": batchSizeGen
                }
            }

    save_result_generation(
                per_query_results=per_query_results,
                averaged=averaged,
                metadata=metadataPipeline,
                model_path_or_name=modelGenName,
                output_dir=outputDirGen
            )


else: 
    print(f"modeContextAssembling must be either reranked or full. Current : {modeContextAssembling}")