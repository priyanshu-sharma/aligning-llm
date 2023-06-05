import wandb
import trlx
from transformers import AutoTokenizer
from datasets import load_dataset
from model_domain import (
    llama_ilql_config,
    metric_fn_for_ilql,
    build_imdb_dataset_test,
)
wandb.init(project="Aligning-LLM")

def llama_ilql_learning():
    config = llama_ilql_config
    tokenizer = AutoTokenizer.from_pretrained("jeffwan/llama-7b-hf")
    dataset = load_dataset("imdb", split="train")
    prompts = dataset["text"]
    rewards = dataset["label"]
    val_prompts = build_imdb_dataset_test(tokenizer)["query"][0:100]

    trlx.train(
        samples=prompts,
        rewards=rewards,
        eval_prompts=val_prompts,
        reward_fn=metric_fn_for_ilql,
        config=config,
    )

llama_ilql_learning()