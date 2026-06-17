import sys
sys.path.append('../../')

from data.load_dataset import load_bioASQ, build_candidate_pool_bioASQ
from models.load_reranker import load_reranker
# from inference.rerank import rerank, evaluate_reranker, evaluate_reranker_dataset
from pprint import pp
import os
import json
from tqdm import tqdm
from models.finetuning.finetune_utils import build_training_pairs, RerankerDataset, finetune_reranker
import os



USE_PRECOMPUTE_FILE = bool(os.environ["USE_PRECOMPUTE_FILE"]) if os.environ.get("USE_PRECOMPUTE_FILE") else True
DEBUG_N = int(os.environ["DEBUG_N"]) if os.environ.get("DEBUG_N") else None

debug = f"""
USE_PRECOMPUTE_FILE = {USE_PRECOMPUTE_FILE}
DEBUG_N = {DEBUG_N}
"""
print(debug)

abstracts_path = "../../data/BioASQ-training14b/abstract_list_tokenized.json"
bioasq_path='../../data/BioASQ-training14b/training14b.json'
modelName = "ncbi/MedCPT-Cross-Encoder"
min_lcs_words = 5


train_dataset, val_dataset, test_dataset, abstracts = load_bioASQ(
    abstracts_path=abstracts_path,
    bioasq_path=bioasq_path,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)


train_pool_path = "../../evaluation/train_pool.json"
test_pool_path = "../../evaluation/test_pool.json"
val_pool_path = "../../evaluation/val_pool.json"

# Check if the test pool file already exists
# this quickly takes a lot of time to run, so I dump it for ease of use since I tend to work pretty iteratively. 
if os.path.exists(test_pool_path) and USE_PRECOMPUTE_FILE:
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
    train_pool = [build_candidate_pool_bioASQ(entry, abstracts, min_lcs_words) for entry in tqdm(train_dataset, desc="Building train pool")]
    test_pool = [build_candidate_pool_bioASQ(entry, abstracts, min_lcs_words) for entry in tqdm(test_dataset, desc="Building test pool")]
    val_pool = [build_candidate_pool_bioASQ(entry, abstracts, min_lcs_words) for entry in tqdm(val_dataset, desc="Building val pool")]
    
    print("Saving built pools to JSON files...")
    
    # Dump the pools for later use
    with open(train_pool_path, "w", encoding="utf-8") as train_p_json:
        json.dump(train_pool, train_p_json, ensure_ascii=False, indent=4)
        
    with open(test_pool_path, "w", encoding="utf-8") as test_p_json:
        json.dump(test_pool, test_p_json, ensure_ascii=False, indent=4)
        
    with open(val_pool_path, "w", encoding="utf-8") as val_p_json:
        json.dump(val_pool, val_p_json, ensure_ascii=False, indent=4)

DEBUG_N = int(len(train_pool) /20) # test with 5% of training set

pool_slice = train_pool if DEBUG_N is None else train_pool[:DEBUG_N]

model, tokenizer = load_reranker(
    model_path_or_name=modelName
)



#----------- Build pairs for training----------------
threshold = 0.5
upper_margin = 0.2
lower_margin = 0.1
max_positives_per_query=10
n_random_negatives = 3
n_hard_negatives = 2
batch_size = 16

all_training_pairs = []
skipped = 0

for entry in tqdm(pool_slice, desc="Building training pairs"):
    pairs = build_training_pairs(
        pool_entry=entry,
        model=model,
        tokenizer=tokenizer,
        threshold=threshold,
        upper_margin=upper_margin,
        lower_margin=lower_margin,
        max_positives_per_query=max_positives_per_query,
        n_random_negatives=n_random_negatives,
        n_hard_negatives=n_hard_negatives,
        batch_size=batch_size
    )
    if not pairs:
        skipped += 1
    all_training_pairs.extend(pairs)

print(f"Total training pairs: {len(all_training_pairs)}")
print(f"Queries contributing zero pairs: {skipped}/{len(pool_slice)}")


# --------------- Create dataset

dataset = RerankerDataset(all_training_pairs)


# ------------ Main method to fine the reranker


epochs = 3
batch_sizeFineTuneReranker = 16
learning_rate = 1e-5
warmup_ratio = 0.1
margin=1
max_grad_norm=1
threshold=0.5
k_values=[5,10,15,20]
val_batch_size=16
early_stopping_patience=2

output_dir = f"finetuningResult/run_lr{learning_rate}_epoch{epochs}warm_{warmup_ratio}_margin{margin}_maxPos{max_positives_per_query}_maxRneg{n_random_negatives}_maxHneg{n_hard_negatives}/" 
modelName.replace(".", "-") # replace . by -, to avoid potential problem with imports and stuff
# TODO ^, or use something like that
metadata = {
    "model" : {
        "output_dir" : output_dir,
        "epochs": epochs,
        "batch_sizeFineTuneReranker": batch_sizeFineTuneReranker,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "margin": margin,
        "max_grad_norm": max_grad_norm,
        "threshold": threshold,
        "k_values": k_values,
        "val_batch_size": val_batch_size,
        "early_stopping_patience": early_stopping_patience
    },
    "dataset" : {
        "threshold" : threshold,
        "upper_margin" : upper_margin,
        "lower_margin" : lower_margin,
        "max_positives_per_query" : max_positives_per_query,
        "n_random_negatives" : n_random_negatives,
        "n_hard_negatives" : n_hard_negatives,
        "batch_size" : batch_size
    }
}

finetune_result = finetune_reranker(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    val_pool=val_pool,
    output_dir = output_dir,
    epochs=epochs,
    batch_size=batch_sizeFineTuneReranker,
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    margin=margin,
    max_grad_norm=max_grad_norm,
    threshold=threshold,
    k_values=k_values,
    val_batch_size=val_batch_size,
    early_stopping_patience=early_stopping_patience,
    metadata=metadata
)

pp(finetune_result)