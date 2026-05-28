import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.append('../')
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def load_reranker(model_path_or_name: str = "ncbi/MedCPT-Cross-Encoder", device: str = None):
    """
    Loads the MedCPT cross-encoder model and its tokenizer.

    Args:
        model_path_or_name : HuggingFace model name or path to a local finetuned checkpoint.
        device             : Target device string ("cpu", "cuda", "cuda:1", ...).
                             If None, defaults to cuda if available, else cpu.

    Returns:
        model     : The loaded model in eval mode, on the target device.
        tokenizer : The associated tokenizer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name,token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_path_or_name, token=HF_TOKEN)
    model.to(device)
    model.eval()

    print(f"Reranker loaded from '{model_path_or_name}' on device '{device}'.")
    return model, tokenizer