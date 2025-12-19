set -x
pwd=$(pwd)

model_name=("Qwen/Qwen3-1.7B" "infinitylogesh/Qwen3-1.7B-GRPO-SRT-Math-12k-Stage-0" "infinitylogesh/Qwen3-1.7B-GRPO-SRT-Math-12k-Stage-1" "infinitylogesh/Qwen3-1.7B-GRPO-SRT-Math-12k-Stage-2")
task=("aime25,aime24,gsm8k,math" "aime25,aime24,gsm8k,math" "aime25,aime24,gsm8k,math" "aime25,aime24,gsm8k,math")
export WANDB_PROJECT="verl-srt-eval-${task[0]}"
export HF_HUB_ENABLE_HF_TRANSFER=0

for i in "${!model_name[@]}"; do
    # get only model name without path
    model_name_only=$(basename ${model_name[$i]})
    export WANDB_RUN_ID=${model_name_only}
    python3 ${pwd}/main.py ${model_name[$i]} ${task[$i]}
done