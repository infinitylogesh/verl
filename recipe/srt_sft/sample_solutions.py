#%%

from openai import OpenAI
from openai import AsyncOpenAI
from utils import last_boxed_only_string, remove_boxed
import os
import aiohttp
import asyncio

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    )

SYSTEM_PROMPT = """
You are a helpful assistant that can solve math problems. You should reason step by step and output the final answer within \\boxed{}.
"""



def extract_solution(solution_str: str) -> str:
    solution_substr = last_boxed_only_string(solution_str)
    if solution_substr is None:
        return None
    try:
        box_removed = remove_boxed(solution_substr)
    except:
        box_removed = None

    return box_removed

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    return str1 == str2

def format_response(response_choice: dict) -> str:
    response_text = response_choice.message.content
    reasoning_details = response_choice.message.reasoning_details[0]['text']
    return f"<think>\n{reasoning_details}\n</think>\n{response_text}\n"

async def sample_solution(question: str) -> str:
    
    # First API call with reasoning
    response = await client.chat.completions.create(
        model="deepseek/deepseek-v3.2",
        messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
                ],
        extra_body={"reasoning": {"enabled": True}},
        temperature=1.0,
        max_tokens=4096
    )

    return response

async def search_solutions(question: str,ground_truth: str,n: int = 10) -> str:
    correct_solutions = []
    incorrect_solutions = []
    for i in range(n):
        response = await sample_solution(question)
        response_choice = response.choices[0]
        solution = extract_solution(response_choice.message.content)
        if is_equiv(solution, ground_truth):
            correct_solutions.append(format_response(response_choice))
            break
        else:
            incorrect_solutions.append(format_response(response_choice))
    return {"question": question, "correct_solutions": correct_solutions, "incorrect_solutions": incorrect_solutions}


async def batch_search_solutions(samples: list[dict],n: int = 10) -> dict:
    tasks = []
    for sample in samples:
        question = sample["prompt"]
        ground_truth = sample['reward_model']['solution_hidden_during_training']
        tasks.append(search_solutions(question, ground_truth, n))
    return await asyncio.gather(*tasks)

# %%

from datasets import load_dataset

dataset = load_dataset("infinitylogesh/math_12k_srt_splits","stage_0")

train_dataset = dataset["train"]
test_dataset = dataset["test"]

# %%