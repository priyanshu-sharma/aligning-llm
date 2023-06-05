from transformers import AutoTokenizer
from datasets import load_dataset
from model_domain import LengthSampler

def tokenize(sample):
    sample["review"] = sample["review"].replace("/>br", "")
    input_ids = tokenizer.encode(sample["review"])[: input_size()] + [tokenizer.eos_token_id]
    sample["query"] = tokenizer.decode(input_ids)
    return sample

def build_imdb_dataset_test(tokenizer, input_min_text_length=2, input_max_text_length=8):
    dataset = load_dataset("imdb", split="test")
    dataset = dataset.rename_columns({"text": "review"})
    dataset = dataset.filter(lambda x: len(x["review"]) > 200, batched=False)
    input_size = LengthSampler(input_min_text_length, input_max_text_length)
    dataset = dataset.map(tokenize, batched=False)
    return dataset