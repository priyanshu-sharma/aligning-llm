import wandb
from transformers import AutoTokenizer
import trlx
from model_domain import (
    llama_ppo_config,
    metric_fn_for_ppo,
    build_imdb_dataset_train,
    build_imdb_dataset_test,
)
wandb.init(project="Aligning-LLM")

def llama_ppo_learning():
    config = llama_ppo_config
    tokenizer = AutoTokenizer.from_pretrained("jeffwan/llama-7b-hf")
    dataset = build_imdb_dataset_train(tokenizer)
    prompts = dataset["query"]
    val_prompts = build_imdb_dataset_test(tokenizer)["query"][0:100]

    trlx.train(
        prompts=prompts,
        eval_prompts=val_prompts,
        reward_fn=metric_fn_for_ppo,
        config=config,
    )

llama_ppo_learning()