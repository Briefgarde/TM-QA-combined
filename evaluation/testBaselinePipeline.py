import sys
sys.path.append('../')
# so that it sits at the root of the program. 

from data.load_dataset import load_bioASQ, build_candidate_pool_bioASQ
from models.load_reranker import load_reranker, load_genLM
from saveResult import save_results_reranker, save_result_generation
from inference.rerank import rerank, evaluate_reranker_dataset
from inference.generate import generate_dataset, evaluate_generation_dataset
from inference.assemble_context import assemble_context_dataset
import json
from tqdm import tqdm
import os
from pprint import pp


# those might need to come from a .sh script later. 
modelName = "ncbi/MedCPT-Cross-Encoder" # while I haven't tested it yet, we should be able to test other reranker really easily. 
threshold = 0.5 # % of overlap a sentence must have with a snippet to be considered a good sentence. 
k_values=[5,10,15,20]
abstracts_path = "../data/BioASQ-training14b/abstract_list_tokenized.json"
bioasq_path='../data/BioASQ-training14b/training14b.json'
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1
batch_size = 16 # how many sentences the reranker treast at once for a given query. 
# this may affect speed during inference, so keep an eye on it. 
output_dir="result/TestBaseLineReranker"

#----------------------------#
# reranking part

model, tokenizer = load_reranker(model_path_or_name=modelName)

train_dataset, val_dataset, test_dataset, abstracts = load_bioASQ(
    abstracts_path=abstracts_path,
    bioasq_path=bioasq_path,
    train_ratio=train_ratio, 
    test_ratio=test_ratio,
    val_ratio=val_ratio
)

train_pool_path = "train_pool.json"
test_pool_path = "test_pool.json"
val_pool_path = "val_pool.json"

# Check if the test pool file already exists
# this quickly takes a lot of time to run, so I dump it for ease of use since I tend to work pretty iteratively. 
# if os.path.exists(test_pool_path):
if os.path.exists(test_pool_path):
    print("Pre-computed pools found. Loading from JSON files...")
    
    with open(train_pool_path, "r", encoding="utf-8") as train_p_json:
        train_pool = json.load(train_p_json)
        
    with open(test_pool_path, "r", encoding="utf-8") as test_p_json:
        test_pool = json.load(test_p_json)
        
    with open(val_pool_path, "r", encoding="utf-8") as val_p_json:
        val_pool = json.load(val_p_json)

else:
    print("Pre-computed pools not found. Building pools...")
    
    # Build the pools
    train_pool = [build_candidate_pool_bioASQ(entry, abstracts) for entry in tqdm(train_dataset, desc="Building train pool")]
    test_pool = [build_candidate_pool_bioASQ(entry, abstracts) for entry in tqdm(test_dataset, desc="Building test pool")]
    val_pool = [build_candidate_pool_bioASQ(entry, abstracts) for entry in tqdm(val_dataset, desc="Building val pool")]
    
    print("Saving built pools to JSON files...")
    
    # Dump the pools for later use
    with open(train_pool_path, "w", encoding="utf-8") as train_p_json:
        json.dump(train_pool, train_p_json, ensure_ascii=False, indent=4)
        
    with open(test_pool_path, "w", encoding="utf-8") as test_p_json:
        json.dump(test_pool, test_p_json, ensure_ascii=False, indent=4)
        
    with open(val_pool_path, "w", encoding="utf-8") as val_p_json:
        json.dump(val_pool, val_p_json, ensure_ascii=False, indent=4)


queries = [test_pool[i]['query'] for i in range(2)]
candidates_list = [test_pool[i]['candidates'] for i in range(2)]

# queries = [test_pool[i]['query'] for i in range(10)]
# candidates_list = [test_pool[i]['candidates'] for i in range(10)]

# queries = [test_pool[i]['query'] for i in range(len(test_pool))]
# candidates_list = [test_pool[i]['candidates'] for i in range(len(test_pool))]

scoreTest = []
metadataTest = []
for q, cand in tqdm(zip(queries, candidates_list), total=len(queries), desc="Reranking"):
    ranked, meta = rerank(query=q, candidates=cand, model=model, tokenizer=tokenizer)
    scoreTest.append(ranked)
    metadataTest.append(meta)

# potentially, this might get looped over threshold values. 
metrics = evaluate_reranker_dataset(scoreTest, metadataTest, threshold=threshold, k_values=k_values)
save_results_reranker(scored=scoreTest, metadata=metadataTest, metrics=metrics, model_path_or_name=modelName, threshold=threshold, k_values=k_values, output_dir=output_dir)


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

bertscore_model_roberta = "roberta-large"
bertscore_model_biomedical = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
output_dir_gen = "result/TestBaseLinePipeline"

modelGen, tokenGen = load_genLM(model_path_or_name=modelGenName)

contexts = assemble_context_dataset(pool=test_pool[:2], 
                                    scored_list=scoreTest,
                                    abstracts=abstracts,
                                    tokenizer=tokenGen,
                                    mode=modeContextAssembling,
                                    top_k=top_k,
                                    max_context_tokens=maxContextToken,
                                    prompt_template=prompt_template)

generated_sets = generate_dataset(
                                    contexts=contexts,
                                    model=modelGen,
                                    tokenizer=tokenGen,
                                    max_new_tokens=max_new_token,
                                    repetition_penalty=repetition_penalty
                                )
# compared to generated_sets, per_query_results has strictly more info since it contain the id, query, answer, generated, and the metric
per_query_results, averaged = evaluate_generation_dataset(
    generation_results=generated_sets,
    bertscore_model_roberta=bertscore_model_roberta,
    bertscore_model_biomedical=bertscore_model_biomedical
)

metadataPipeline = {
    "generationMetadata" : {
        "modelGenName" : modelGenName,
        "modeContextAssembling" : modeContextAssembling,
        "topk" : top_k,
        "max_new_token" : max_new_token,
        "maxContextToken" : maxContextToken,
        "prompt_template" : prompt_template,
        "repetition_penalty" : repetition_penalty,
        "bertscore_model_roberta" : bertscore_model_roberta,
        "bertscore_model_biomedical" : bertscore_model_biomedical
    },
    "rerankingMetadata" : {
        "modelReranker" : modelName,
        "threshold" : threshold,
        "k_values" : k_values,

    }
}
save_result_generation(
    per_query_results=per_query_results,
    averaged=averaged,
    metadata=metadataPipeline,
    model_path_or_name=modelGenName,
    output_path_or_dir=output_dir_gen)