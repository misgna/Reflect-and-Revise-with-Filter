import os
import pandas as pd
from sklearn.model_selection import train_test_split 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def to_inference_structure(dataset):
    data = []
    for _, row in dataset.iterrows():
        data.append({
            "essay_prompt": row.get("essay_prompt", ""),
            "response": row.get("essay", row.get("response", "")),
            "answer": int(row.get("domain1_score", row.get("score", 0)))
        })
    return data
    
def prepare_asap_dataset(essay_set, data_dir="./dataset/asap-aes"):
    """
    Loads and formats the ASAP dataset CSV for a given essay set.
    """
    file_path_train = os.path.join(data_dir, f"asap_{essay_set}/train_data.csv")
    train = to_inference_structure(pd.read_csv(file_path_train))

    
    file_path_val = os.path.join(data_dir, f"asap_{essay_set}/val_data.csv")
    val = to_inference_structure(pd.read_csv(file_path_val))
        
    file_path_test = os.path.join(data_dir, f"asap_{essay_set}/test_data.csv")
    test = to_inference_structure(pd.read_csv(file_path_test))
    
    return train, val, test

def prepare_ets_dataset(data_dir="./dataset/asap-ets"):
    file_path = os.path.join(data_dir, "ets_dataset.csv")
    df = pd.read_csv(file_path)
    data = df.to_dict(orient="records")
    return data[:int(len(data)*0.7)], data[int(len(data)*0.7):int(len(data)*0.85)], data[int(len(data)*0.85):]

def prepare_asap2_dataset(data_dir="./dataset/asap-ets"):
    file_path = os.path.join(data_dir, "asap2_dataset.csv")
    df = pd.read_csv(file_path)
    data = df.to_dict(orient="records")
    return data[:int(len(data)*0.7)], data[int(len(data)*0.7):int(len(data)*0.85)], data[int(len(data)*0.85):]

def create_dataset_per_essay_set():
    dataset = pd.read_csv("dataset/asap-aes/training_set_rel3.tsv", sep='\t', encoding='ISO-8859-1') 
    
    for essay_set in range(1, 9):
        asap = dataset[dataset['essay_set'] == essay_set]
        train, test =  train_test_split(asap, train_size=0.66, random_state=42)
        train, val = train_test_split(train, train_size=0.70, random_state=42)
        
        train.to_csv(f"dataset/asap-aes/asap_{essay_set}/train_data.csv", index=False)
        val.to_csv(f"dataset/asap-aes/asap_{essay_set}/val_data.csv", index=False)
        test.to_csv(f"dataset/asap-aes/asap_{essay_set}/test_data.csv", index=False)
            
def load_model_and_tokenizer(checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if torch.cuda.is_available():
        print("Running on GPU")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="balanced",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    else:
        print("Running on CPU")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float32,
        )
        
    #print(f"Device: {model.hf_device_map}")

    return model, tokenizer

MODEL_CHECKPOINTS: dict[str, str] = {
    "gpt": "openai/gpt-oss-20b",
    "gemma_7b": "google/gemma-7b-it",
    "llama3_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral_8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "falcon3_1b": "tiiuae/Falcon3-1B-Instruct",
    "falcon3_3b": "tiiuae/Falcon3-3B-Instruct",
    "falcon3_7b": "tiiuae/Falcon3-7B-Instruct",
    "falcon3_10b": "tiiuae/Falcon3-10B-Instruct",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek_67b": "deepseek-ai/deepseek-llm-67b-chat",
    "deepseek_moe_16b": "deepseek-ai/deepseek-moe-16b-chat",
    "phi3.5_mini": "microsoft/Phi-3.5-mini-instruct",
    "phi3.5_moe": "microsoft/Phi-3.5-MoE-instruct",
    "phi3_mini_4k": "microsoft/Phi-3-mini-4k-instruct",
    "phi3_mini_medium_128k": "microsoft/Phi-3-medium-128k-instruct",
    "phi3_small_8k": "microsoft/Phi-3-small-8k-instruct",
    "phi3_small_128k": "microsoft/Phi-3-small-128k-instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
}


def get_checkpoint(model_id: str) -> str:
    """Retrieves HuggingFace model checkpoint path by model alias."""
    if model_id not in MODEL_CHECKPOINTS:
        raise KeyError(
            f"Unknown model_id: '{model_id}'. Available models: {list(MODEL_CHECKPOINTS.keys())}"
        )
    return MODEL_CHECKPOINTS[model_id]
        
if __name__ == "__main__":
    create_dataset_per_essay_set()
    

    