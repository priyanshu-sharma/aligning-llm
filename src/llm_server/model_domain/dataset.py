from datasets import load_dataset
from model_domain import LengthSampler
from transformers import AutoTokenizer


def build_imdb_dataset_train(tokenizer, input_min_text_length=2, input_max_text_length=8):
    dataset = load_dataset("imdb", split="train")
    dataset = dataset.rename_columns({"text": "review"})
    dataset = dataset.filter(lambda x: len(x["review"]) > 200, batched=False)
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["review"] = sample["review"].replace("/>br", "")
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()] + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")
    return dataset

def build_imdb_dataset_test(tokenizer, input_min_text_length=2, input_max_text_length=8):
    dataset = load_dataset("imdb", split="test")
    dataset = dataset.rename_columns({"text": "review"})
    dataset = dataset.filter(lambda x: len(x["review"]) > 200, batched=False)
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["review"] = sample["review"].replace("/>br", "")
        input_ids = tokenizer.encode(sample["review"])[: input_size()] + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(input_ids)
        return sample

    dataset = dataset.map(tokenize, batched=False)
    return dataset