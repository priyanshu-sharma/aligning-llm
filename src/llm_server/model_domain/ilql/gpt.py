import wandb
from datasets import load_dataset
import trlx
from transformers import AutoTokenizer
from model_domain import (
    sentiment_fn,
    metric_fn_for_ilql,
    gpt_ilql_config,
    build_imdb_dataset_test
)
wandb.init(project="Aligning-LLM")

def gpt_ilql_learning():
    config = gpt_ilql_config
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("imdb", split="train")
    prompts = dataset["text"]
    rewards = dataset["label"]
    val_prompts = build_imdb_dataset_test(tokenizer)["query"][0:100]
    trlx.train(
        samples=prompts,
        rewards=rewards,
        eval_prompts=val_prompts,
        metric_fn=metric_fn_for_ilql,
        config=config,
    )

gpt_ilql_learning()