import sys
sys.path.append('../')
# so that it sits at the root of the program. 
from models.load_reranker import load_reranker, load_genLM
from saveResult import save_result_generation, save_results_reranker
from inference.rerank import rerank, evaluate_reranker_dataset
from inference.generate import generate_dataset_batch, evaluate_generation_dataset
from inference.assemble_context import assemble_context_dataset
import json
from tqdm import tqdm
import os
from pprint import pp
import torch

#---------
DEBUG_N = int(os.environ["DEBUG_N"]) if os.environ.get("DEBUG_N") else None
USE_PRECOMPUTE_FILE = bool(os.environ["USE_PRECOMPUTE_FILE"]) if os.environ.get("USE_PRECOMPUTE_FILE") else True
print(DEBUG_N)
print(USE_PRECOMPUTE_FILE)



# those might need to come from a .sh script later. 
modelName = "ncbi/MedCPT-Cross-Encoder" 
# modelName = "../models/finetuning/finetuningResult/run_lr1e-05_epoch5warm_0-1_margin1_maxPos10_maxRneg3_maxHneg2/best_model"

# threshold = 0.5 # % of overlap a sentence must have with a snippet to be considered a good sentence. 
# min_lcs_words = 5 
# k_values=[5,10,15,20]
# batch_sizeReranker = 16 # how many sentences the reranker treast at once for a given query. 
# this may affect speed during inference, so keep an eye on it. 
# output_dir="result/TestBaseLineReranker"

#----------------------------#
# reranking part

model, tokenizer = load_reranker(model_path_or_name=modelName)

test_pool_path = "pubmedqa_oracle_pool.json"

# Check if the test pool file already exists
# this quickly takes a lot of time to run, so I dump it for ease of use since I tend to work pretty iteratively. 
if USE_PRECOMPUTE_FILE:
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


# Since PubMedQA does not have a relevance score like we did for BioASQ, 
# the reranker can not be directly evaluated. 
# thresholds_to_test = [0.3, 0.5, 0.7] 
# for threshold in thresholds_to_test: 
#     metrics = evaluate_reranker_dataset(
#         scoreTest, metadataTest,
#         threshold=threshold,
#         k_values=k_values
#     )
#     save_results_reranker(
#         scored=scoreTest,
#         metadata=metadataTest,
#         metrics=metrics,
#         model_path_or_name=modelName,
#         threshold=threshold,
#         k_values=k_values,
#         output_dir=output_dir
#     )

del model, tokenizer # this is only for the reranker
torch.cuda.empty_cache()

#----------------------------#
# Generative part

modelGenName = "stanford-crfm/BioMedLM"
modeContextAssembling = "reranked" # can be reranked, to use the top-k sentence from the abstract,
# or full, which uses all of the abstracts as context. 
top_k = 5 # control how many sentences go into the context. 
max_new_token = 100 # token reserved, out of 1024, for generation
maxContextToken = 1024-max_new_token # the leftover token can be used for context. 
prompt_template = "Context: {context}\nQuestion: {query}\nAnswer:" # TODO or to test at least
repetition_penalty=1.2 # this penalize the LM for getting stuck in a loop of the exact same sentence. 
# This is likely to happen in greedy decoding setup, since there's no variability. 
use_newline_stop = True
batch_size_gen = 4 
bertscore_model_roberta = "roberta-large"
bertscore_model_biomedical = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
output_dir_gen = "result/TestOraclePubMedQA"

modelGen, tokenGen = load_genLM(model_path_or_name=modelGenName)


# ------------testing reranked mode---------------
top_k_values_to_test = [1,2,3,5]
for top_k in top_k_values_to_test:
        contexts = assemble_context_dataset(
            pool=pool_slice,
            scored_list=scoreTest,
            abstracts=None,
            tokenizer=tokenGen,
            mode="reranked",
            top_k=top_k,
            max_context_tokens=maxContextToken,
            prompt_template=prompt_template
        )


        generated_sets = generate_dataset_batch(
            contexts=contexts,
            model=modelGen,
            tokenizer=tokenGen,
            max_new_tokens=max_new_token,
            repetition_penalty=repetition_penalty,
            use_newline_stop=use_newline_stop,
            batch_size=batch_size_gen
        )

        per_query_results, averaged = evaluate_generation_dataset(
            generation_results=generated_sets,
            bertscore_model_roberta=bertscore_model_roberta,
            bertscore_model_biomedical=bertscore_model_biomedical
        )

        metadataPipeline = {
            "dataset": "BioASQ-training14b",
            "split": "test",
            "generationMetadata": {
                "modelGenName": modelGenName,
                "modeContextAssembling": "reranked",
                "topk": top_k,
                "max_new_token": max_new_token,
                "maxContextToken": maxContextToken,
                "prompt_template": prompt_template,
                "repetition_penalty": repetition_penalty,
                "bertscore_model_roberta": bertscore_model_roberta,
                "bertscore_model_biomedical": bertscore_model_biomedical,
                "use_newline_stop": use_newline_stop,
                "batch_size": batch_size_gen
            }
        }

        save_result_generation(
            per_query_results=per_query_results,
            averaged=averaged,
            metadata=metadataPipeline,
            model_path_or_name=modelGenName,
            output_dir=output_dir_gen
        )

# ------------testing full mode---------------
# top_k = 0
# contexts = assemble_context_dataset(
#             pool=pool_slice,
#             scored_list=scoreTest,
#             abstracts=abstracts,
#             tokenizer=tokenGen,
#             mode="full",
#             top_k=top_k,
#             max_context_tokens=maxContextToken,
#             prompt_template=prompt_template
#         )

# generated_sets = generate_dataset_batch(
#             contexts=contexts,
#             model=modelGen,
#             tokenizer=tokenGen,
#             max_new_tokens=max_new_token,
#             repetition_penalty=repetition_penalty,
#             use_newline_stop=use_newline_stop,
#             batch_size=batch_size_gen
#         )

# per_query_results, averaged = evaluate_generation_dataset(
#             generation_results=generated_sets,
#             bertscore_model_roberta=bertscore_model_roberta,
#             bertscore_model_biomedical=bertscore_model_biomedical
#         )

# metadataPipeline = {
#             "dataset": "BioASQ-training14b",
#             "split": "test",
#             "generationMetadata": {
#                 "modelGenName": modelGenName,
#                 "modeContextAssembling": "full",
#                 "topk": top_k,
#                 "max_new_token": max_new_token,
#                 "maxContextToken": maxContextToken,
#                 "prompt_template": prompt_template,
#                 "repetition_penalty": repetition_penalty,
#                 "bertscore_model_roberta": bertscore_model_roberta,
#                 "bertscore_model_biomedical": bertscore_model_biomedical,
#                 "use_newline_stop": use_newline_stop,
#                 "batch_size": batch_size_gen
#             }
#         }

# save_result_generation(
#             per_query_results=per_query_results,
#             averaged=averaged,
#             metadata=metadataPipeline,
#             model_path_or_name=modelGenName,
#             output_dir=output_dir_gen
#         )
