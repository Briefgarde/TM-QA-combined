import sys
sys.path.append('../')

from data.load_dataset import load_bioASQ, build_candidate_pool_bioASQ
from models.load_reranker import load_reranker
from saveResult import save_results
from inference.rerank import rerank, evaluate_reranker_dataset
import json
from tqdm import tqdm

# this might need to come from a .sh script later. 
modelName = "ncbi/MedCPT-Cross-Encoder"
threshold = 0.5
k_values=[5,10,15,20]
abstracts_path = "../data/BioASQ-training14b/abstract_list_tokenized.json"
bioasq_path='../data/BioASQ-training14b/training14b.json'
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1
batch_size = 16
output_dir="result/TestBaseLineReranker"

model, tokenizer = load_reranker(model_path_or_name=modelName)

train_dataset, val_dataset, test_dataset, abstracts = load_bioASQ(
    abstracts_path=abstracts_path,
    bioasq_path=bioasq_path,
    train_ratio=train_ratio, 
    test_ratio=test_ratio,
    val_ratio=val_ratio
)

# prepare candidate 
# train_pool = [build_candidate_pool_bioASQ(entry, abstracts) for entry in tqdm(train_dataset, desc="Building train pool")]
test_pool = [build_candidate_pool_bioASQ(entry, abstracts) for entry in tqdm(test_dataset, desc="Building test pool")]

# if things were already run before, this can be used more quickly. 
# with open("train_pool.json", "r", encoding="utf-8") as train_p_json:
#     train_pool = json.load(train_p_json)
# with open("test_pool.json", "r", encoding="utf-8") as test_p_json:
#     test_pool = json.load(test_p_json)

# queries = [test_pool[i]['query'] for i in range(10)]
# candidates_list = [test_pool[i]['candidates'] for i in range(10)]

# scoreTest = []
# for q, cand in tqdm(zip(queries, candidates_list), total=len(queries), desc="Reranking"):
#     scoreTest.append(rerank(query=q, candidates=cand, model=model, tokenizer=tokenizer))

queries = [test_pool[i]['query'] for i in range(len(test_pool))]
candidates_list = [test_pool[i]['candidates'] for i in range(len(test_pool))]

scoreTest = []
for q, cand in tqdm(zip(queries, candidates_list), total=len(queries), desc="Reranking"):
    scoreTest.append(rerank(query=q, candidates=cand, model=model, tokenizer=tokenizer))

metrics = evaluate_reranker_dataset(scoreTest, threshold=threshold, k_values=k_values)

save_results(scoreTest, metrics=metrics, model_path_or_name=modelName, threshold=threshold, k_values=k_values, output_dir=output_dir)

