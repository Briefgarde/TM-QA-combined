import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

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

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_path_or_name, token=HF_TOKEN)
    model.to(device)
    if "cuda" in str(device):
        torch.cuda.synchronize()
    model.eval()

    print(f"Reranker loaded from '{model_path_or_name}' on device '{device}'.")
    return model, tokenizer


def load_genLM(model_path_or_name: str = "stanford-crfm/BioMedLM", device: str = None, torch_dtype: torch.dtype = torch.float16):
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

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path_or_name, token=HF_TOKEN, dtype=torch_dtype)
    model.to(device)
    if "cuda" in str(device):
        torch.cuda.synchronize()
    model.eval()

    print(f"Generative model loaded from '{model_path_or_name}' on device '{device}'.")
    return model, tokenizer