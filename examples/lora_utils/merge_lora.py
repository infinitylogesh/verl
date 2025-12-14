"""
python3 merge_lora.py --model_path Qwen/Qwen3-1.7B \
    --lora_path /workspace/verl/checkpoints/verl_grpo_example_gsm8k/qwen3_1_7b_srt_grpo_math_12k_stage_0_lora_rollout_32_lora_64_32/global_step_200/actor/lora_adapter/ \
    --output_path /workspace/verl/checkpoints/qwen3_1_7b_srt_grpo_math_12k_stage_0_lora_rollout_32_lora_64_32_merged/ \
    --hub_path infinitylogesh/Qwen3-1.7B-GRPO-SRT-Math-12k-Stage-0
"""


from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from argparse import ArgumentParser

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
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--hub_path", type=str, required=True)
    args = parser.parse_args()
    merge_lora(args.model_path, args.lora_path, args.output_path, args.hub_path)