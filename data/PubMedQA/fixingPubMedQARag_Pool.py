import sys
sys.path.append('../')
import json


sibil_path = "pubmedqa_rag_pool.json"

with open(sibil_path, "r", encoding="utf-8") as sibil_pool_json:
    sibil_pool = json.load(sibil_pool_json)

sibil_pool_fixed = [entry for entry in sibil_pool if len(entry["candidates"]) > 0]
print(f"Kept {len(sibil_pool_fixed)} of {len(sibil_pool)} entries")

output_path = "pubmedqa_rag_pool_fixed.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sibil_pool_fixed, f, indent=2)