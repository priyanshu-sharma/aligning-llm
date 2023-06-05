from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

t5_ppo_config = TRLConfig(
    train=TrainConfig(
        seq_length=128,
        epochs=100,
        total_steps=100000,
        batch_size=12,
        checkpoint_interval=10000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        save_best=False,
    ),
    model=ModelConfig(
        model_path="lvwerra/t5-imdb",
        num_layers_unfrozen=-1,
        model_arch_type="seq2seq",
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="lvwerra/t5-imdb",
        padding_side="right",
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 5.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 100000,
            "eta_min": 5.0e-5,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=128,
        chunk_size=12,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 50,
            "do_sample": True,
            "top_k": 0,
            "top_p": 1,
            "eos_token_id": -1,
        },
    ),
)