import wandb
from model_domain import (build_imdb_dataset_test,
                          build_imdb_dataset_train,
                          metric_fn_for_ppo, gpt_ppo_config)
from transformers import AutoTokenizer
import trlx
wandb.init(project="Aligning-LLM")

def gpt_ppo_learning():
    config = gpt_ppo_config
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = build_imdb_dataset_train(tokenizer)
    prompts = dataset["query"]
    val_prompts = build_imdb_dataset_test(tokenizer)["query"][0:100]

    trlx.train(
        reward_fn=metric_fn_for_ppo,
        prompts=prompts,
        eval_prompts=val_prompts,
        config=config,
    )

gpt_ppo_learning()