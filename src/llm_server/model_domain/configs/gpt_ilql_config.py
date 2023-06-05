from trlx.data.configs import (ModelConfig, OptimizerConfig, SchedulerConfig,
                               TokenizerConfig, TrainConfig, TRLConfig)
from trlx.models.modeling_ilql import ILQLConfig

gpt_ilql_config = TRLConfig(
    train=TrainConfig(
        seq_length=64,
        batch_size=128,
        epochs=100,
        total_steps=1000,
        checkpoint_interval=1000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
    ),
    model=ModelConfig(model_path="gpt2", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
    optimizer=OptimizerConfig(
        name="adamw", kwargs=dict(lr=5.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=5.0e-5)  # train.total_steps
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
        gen_kwargs=dict(max_new_tokens=56, top_k=20, beta=1, temperature=1.0),
    ),
)