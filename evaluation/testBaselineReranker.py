import sys
sys.path.append('../')
# so that it sits at the root of the program. 

from data.load_dataset import load_bioASQ, build_candidate_pool_bioASQ
from models.load_reranker import load_reranker
from saveResult import save_results
from inference.rerank import rerank, evaluate_reranker_dataset
import json
from tqdm import tqdm
import os

# those might need to come from a .sh script later. 
modelName = "ncbi/MedCPT-Cross-Encoder"
threshold = 0.5 # % of overlap a sentence must have with a snippet to be considered a good sentence. 
k_values=[5,10,15,20]
abstracts_path = "../data/BioASQ-training14b/abstract_list_tokenized.json"
bioasq_path='../data/BioASQ-training14b/training14b.json'
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1
batch_size = 16 # how many sentences the reranker treast at once for a given query. 
output_dir="result/TestBaseLineReranker"

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

# queries = [test_pool[i]['query'] for i in range(10)]
# candidates_list = [test_pool[i]['candidates'] for i in range(10)]

queries = [test_pool[i]['query'] for i in range(len(test_pool))]
candidates_list = [test_pool[i]['candidates'] for i in range(len(test_pool))]

scoreTest = []
metadataTest = []
for q, cand in tqdm(zip(queries, candidates_list), total=len(queries), desc="Reranking"):
    ranked, meta = rerank(query=q, candidates=cand, model=model, tokenizer=tokenizer)
    scoreTest.append(ranked)
    metadataTest.append(meta)

# potentially, this might get looped over threshold values. 
metrics = evaluate_reranker_dataset(scoreTest, metadataTest, threshold=threshold, k_values=k_values)
save_results(scored=scoreTest, metadata=metadataTest, metrics=metrics, model_path_or_name=modelName, threshold=threshold, k_values=k_values, output_dir=output_dir)

