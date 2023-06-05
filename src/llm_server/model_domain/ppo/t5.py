import wandb
from model_domain import (build_imdb_dataset_test,
                          build_imdb_dataset_train, get_positive_score,
                          metric_fn_for_ppo, t5_ppo_config)
from transformers import AutoTokenizer
import trlx
wandb.init(project="Aligning-LLM")

def t5_ppo_learning():
    config = t5_ppo_config
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/t5-imdb")
    dataset = build_imdb_dataset_train(tokenizer)
    prompts = dataset["query"]
    test_dataset = build_imdb_dataset_test(tokenizer)

    val_prompts = test_dataset["query"][0:100]

    trlx.train(
        prompts=prompts,
        eval_prompts=val_prompts,
        reward_fn=metric_fn_for_ppo,
        config=config,
    )

t5_ppo_learning()