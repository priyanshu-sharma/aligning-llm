from trlx.data.configs import (ModelConfig, OptimizerConfig, SchedulerConfig,
                               TokenizerConfig, TrainConfig, TRLConfig)
from trlx.models.modeling_ilql import ILQLConfig

t5_ilql_config = TRLConfig(
    train=TrainConfig(
        seq_length=128,
        epochs=100,
        total_steps=1000,
        batch_size=32,
        checkpoint_interval=1000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
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
    method=ILQLConfig(
        name="ILQLConfig",
        tau=0.7,
        gamma=0.99,
        cql_scale=0.1,
        awac_scale=1,
        alpha=0.001,
        beta=0,
        steps_for_target_q_sync=5,
        two_qs=True,
        gen_kwargs=dict(max_new_tokens=56, top_k=20, beta=4, temperature=1.0),
    ),
)