import spacy
from datasets import load_dataset
import time
from tqdm import tqdm
import json
from pprint import pp
import requests
import sys
import os
from dotenv import load_dotenv
sys.path.append('../')

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

SIBILS_URL = "https://biodiversitypmc.sibils.org/api/search"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

path_name = "qiaojin/PubMedQA"
subset = "pqa_artificial"
split = "train"

ds_artificial = load_dataset(path=path_name, name=subset, split=split, token=HF_TOKEN)
ds_artificial = ds_artificial.select(range(10000))


def fetchAbstractsForQuery(query: str) -> list[dict]:
    params = {
        "q": query,
        "col": "medline"
    }
    try:
        r = requests.post(url=SIBILS_URL, params=params, timeout=30)
        r.raise_for_status()
        response_json = json.loads(r.text)
        hits = response_json['elastic_output']['hits']['hits']
        abstracts = []
        for ab in hits:
            source = ab["_source"]
            abstract_text = source.get("abstract", "").strip()
            if not abstract_text:
                continue
            abstracts.append({
                "id": ab["_id"],
                "sibilScore": ab["_score"],
                "title": source.get("title", ""),
                "fullTextAbstract": abstract_text
            })
        return abstracts
    except requests.exceptions.Timeout:
        print(f"Timeout for query: {query[:60]}...")
        return []
    except Exception as e:
        print(f"Error fetching query '{query[:60]}...': {e}")
        return []
    
nlp = spacy.load("en_core_sci_md")

def tokenize_abstract(text: str) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# --- Main fetching loop ---
# Maps pubmed_id -> list of sentences, deduplicated across queries
abstract_list_tokenized = {}

# Maps query pubid -> list of retrieved pubmed IDs (including empty lists)
query_to_retrieved_ids = {}

for entry in tqdm(ds_artificial, desc="Fetching abstracts"):
    pubid = str(entry['pubid'])
    question = entry['question']

    fetched = fetchAbstractsForQuery(question)
    time.sleep(0.1)

    retrieved_ids = []
    for ab in fetched:
        ab_id = str(ab['id'])
        retrieved_ids.append(ab_id)

        # Only tokenize and store if not already seen
        if ab_id not in abstract_list_tokenized:
            sentences = tokenize_abstract(ab['fullTextAbstract'])
            if sentences:
                abstract_list_tokenized[ab_id] = sentences

    query_to_retrieved_ids[pubid] = retrieved_ids

# --- Save both outputs ---
with open("pubmedqa_abstract_list_tokenized.json", "w", encoding="utf-8") as f:
    json.dump(abstract_list_tokenized, f, ensure_ascii=False, indent=2)

with open("pubmedqa_query_to_retrieved_ids.json", "w", encoding="utf-8") as f:
    json.dump(query_to_retrieved_ids, f, ensure_ascii=False, indent=2)

# --- Summary ---
n_queries_with_results = sum(1 for ids in query_to_retrieved_ids.values() if ids)
n_queries_empty = sum(1 for ids in query_to_retrieved_ids.values() if not ids)
n_unique_abstracts = len(abstract_list_tokenized)

print(f"Queries processed       : {len(query_to_retrieved_ids)}")
print(f"Queries with results    : {n_queries_with_results}")
print(f"Queries with no results : {n_queries_empty}")
print(f"Unique abstracts stored : {n_unique_abstracts}")


pubmedqa_rag_pool = []
skipped_no_abstracts = 0

for entry in tqdm(ds_artificial, desc="Building PubMedQA RAG pool"):
    pubid = str(entry['pubid'])
    question = entry['question']
    long_answer = entry['long_answer']
    final_decision = entry['final_decision']

    retrieved_ids = query_to_retrieved_ids.get(pubid, [])

    # Filter to IDs that are actually present in the tokenized abstracts
    available_ids = [pid for pid in retrieved_ids if pid in abstract_list_tokenized]

    if not available_ids:
        skipped_no_abstracts += 1

    candidates = [
        {"sentence": sentence, "relevance_score": 0.0}
        for pid in available_ids
        for sentence in abstract_list_tokenized[pid]
    ]

    pubmedqa_rag_pool.append({
        "id": pubid,
        "query": question,
        "answer": long_answer,
        "final_decision": final_decision,
        "pubmed_ids": available_ids,
        "candidates": candidates
    })

print(f"Total pool entries      : {len(pubmedqa_rag_pool)}")
print(f"Entries with no abstracts: {skipped_no_abstracts}")
print(f"Entries with candidates : {len(pubmedqa_rag_pool) - skipped_no_abstracts}")

with open("pubmedqa_rag_pool.json", "w", encoding="utf-8") as f:
    json.dump(pubmedqa_rag_pool, f, ensure_ascii=False, indent=2)

print("RAG pool saved to pubmedqa_rag_pool.json")