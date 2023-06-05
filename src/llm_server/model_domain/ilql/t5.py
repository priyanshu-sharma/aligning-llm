import wandb
from datasets import load_dataset
from model_domain import (LengthSampler, build_imdb_dataset_test,
                          get_positive_score, metric_fn_for_ilql,
                          t5_ilql_config)
from transformers import AutoTokenizer

import trlx

wandb.init(project="Aligning-LLM")

def t5_ilql_learning():
    config = t5_ilql_config
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/t5-imdb")
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

t5_ilql_learning()