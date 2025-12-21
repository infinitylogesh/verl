"""

For FSDP checkpoints (sharded across multiple ranks), use verl's model_merger:

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /workspace/verl/checkpoints/verl_grpo_example_gsm8k/qwen3_1_7b_srt_grpo_math_12k_single_stage_rollout_16_fullfinetuning/global_step_124/actor \
    --target_dir /workspace/verl/checkpoints/qwen3_1_7b_srt_grpo_math_12k_single_stage_rollout_16_fullfinetuning_merged/ \
    --hf_upload_path infinitylogesh/Qwen3-1.7B-GRPO-SRT-Math-12k-Single-Stage-Rollout-16-Full-Finetuning

For standard HuggingFace checkpoints with optional LoRA adapters:

python3 merge_lora.py --model_path Qwen/Qwen3-1.7B \
    --lora_path /workspace/verl/checkpoints/verl_grpo_example_gsm8k/qwen3_1_7b_srt_grpo_math_12k_stage_0_lora_rollout_32_lora_64_32/global_step_200/actor/lora_adapter/ \
    --output_path /workspace/verl/checkpoints/qwen3_1_7b_srt_grpo_math_12k_stage_0_lora_rollout_32_lora_64_32_merged/ \
    --hub_path infinitylogesh/Qwen3-1.7B-GRPO-SRT-Math-12k-Stage-0
"""


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from argparse import ArgumentParser
import os

def merge_lora(model_path, lora_path, output_path, hub_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    model.push_to_hub(hub_path, private=False)
    tokenizer.push_to_hub(hub_path, private=False)

if __name__ == "__main__":
    parser = ArgumentParser(description="Merge LoRA adapters and push to HuggingFace Hub")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base HuggingFace model")
    parser.add_argument("--lora_path", type=str, required=False, help="Path to LoRA adapter (optional)")
    parser.add_argument("--output_path", type=str, required=False, help="Path to save merged model")
    parser.add_argument("--hub_path", type=str, required=False, help="HuggingFace Hub repo ID")
    parser.add_argument("--private", action="store_true", help="Create private Hub repository")
    args = parser.parse_args()
    
    if not args.output_path and not args.hub_path:
        parser.error("At least one of --output_path or --hub_path must be provided")
    
    merge_lora(args.model_path, args.lora_path, args.output_path, args.hub_path, args.private)