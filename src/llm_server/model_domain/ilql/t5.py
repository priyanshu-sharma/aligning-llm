import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from model_domain import (
    get_positive_score, 
    t5_ilql_config, 
    LengthSampler,
    metric_fn,
    build_imdb_dataset_test
)

import wandb
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
        metric_fn=metric_fn,
        config=config,
    )

t5_ilql_learning()