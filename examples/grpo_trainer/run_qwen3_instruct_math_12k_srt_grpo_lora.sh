set -x

export FULL_BATCH_SIZE=96
export PPO_MINI_BATCH_SIZE=96
export PER_GPU_MINI_BATCH_SIZE=24

# Number of rollouts
export NUM_PER_PROMPT_ROLLOUTS=16
export NUM_PER_PROMPT_ROLLOUTS_VALIDATION=16

# prompt and response length cutoff
export MAX_RESPONSE_LENGTH=2048
export MAX_PROMPT_LENGTH=1024

# Other hyperparameters
export LEARNING_RATE=1e-6
export KL_COEFF=0.001

# Model path and lora rank and alpha
export MODEL_PATH=Qwen/Qwen3-1.7B
# export LORA_RANK=64
# export LORA_ALPHA=32

# SRT hyperparams
# These are the ones that different between SRT and RL with ground truth
export REWARD_MANAGER='self_learning'
export LOG_THRESHOLD_PLOT=False
export SELF_CONSISTENCY_THRESHOLD=0.0
export SOFT_REWARD=False
export REMOVE_KL_LOSS_FROM_UNLABELLED_EXAMPLES=True
export OVERSAMPLING_KEEP_FRACTION=1.0
export TRAIN_FILE=data/math_12k/stage_0/train.parquet
export VAL_FILE=data/math_12k/stage_0/test.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$FULL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEFF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$NUM_PER_PROMPT_ROLLOUTS \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=$NUM_PER_PROMPT_ROLLOUTS_VALIDATION \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=$REWARD_MANAGER \
    reward_model.reward_kwargs.self_consistency_threshold=$SELF_CONSISTENCY_THRESHOLD \
    reward_model.reward_kwargs.soft_reward=$SOFT_REWARD \
    reward_model.reward_kwargs.remove_kl_loss_from_unlabeled_examples=$REMOVE_KL_LOSS_FROM_UNLABELLED_EXAMPLES \
    reward_model.reward_kwargs.oversampling_keep_fraction=$OVERSAMPLING_KEEP_FRACTION \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen3_1_7b_srt_grpo_math_12k_single_stage_rollout_16_fullfinetuning' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 $@

    # actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    # data.train_batch_size=1024 \
    # trainer.n_gpus_per_node=8 \
    # actor_rollout_ref.model.use_shm=True \
