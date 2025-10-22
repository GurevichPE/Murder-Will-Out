import json
import os
from pathlib import Path
from typing import Union, Dict, List
import torch
import transformers
from huggingface_hub import login
from tqdm.auto import tqdm
from transformers import AutoTokenizer

SYSTEM_PROMPT = """
Your job is to answer an entity-centric question.
You need to answer with the correct entity, without any additional information.
"""

DATA_PATH = "./data/final_dataset"
SAVE_PATH = "./data/sampled_answers_1000_temp1"


def get_user_prompt(question:str) -> str:
    query = f"""
    Here is the question. Simply reply with the correct entity. If you cannot answer for any reason, output None. But
    do try your best to find the correct answer.
    ```
    Question: {question}
    ```
    Just return the answer, with no text around it.
    """
    return query


def get_messages(question:str) -> List[Dict[str,str]]:
    user_query = get_user_prompt(question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    return messages


def load_json(path:Union[str, Path]) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def make_multiple_predictions(pipeline:transformers.Pipeline, question:str, num_answers:int = 500):
    """Generate multiple answers for a question using batch processing with temperature=1"""
    messages = get_messages(question)

    # Create a batch of identical messages for parallel generation
    batch_messages = [messages] * num_answers

    # Process all messages in a single batch
    outputs = pipeline(
        batch_messages,
        max_new_tokens=8,
        batch_size=min(num_answers, 500),  # Use reasonable batch size for memory efficiency
    )

    # Extract answers from batch results
    answers = []
    for output in outputs:
        answer = output[0]["generated_text"][-1]["content"]
        answers.append(answer)

    return answers


def load_pipeline() -> transformers.Pipeline:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hf_token = # INSERT TOKEN
    login(hf_token)
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
        # Set sampling with temperature=1
        do_sample=True,
        temperature=2.0,
    )

    # Fix for batch processing: set pad_token and pad_token_id to enable batching
    if pipeline.tokenizer.pad_token_id is None:
        pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]

    return pipeline


def process_one_file(pipeline, data:List[Dict[str, str]]) -> List[Dict[str, str]]:
    for i in tqdm(range(len(data))):
        entry = data[i]
        question = entry["question"]
        answers = make_multiple_predictions(pipeline, question, num_answers=500)
        entry["sampled_answers"] = answers
        data[i] = entry
    return data


def main():
    pipeline = load_pipeline()
    
    data_path = Path(DATA_PATH)
    save_path = Path(SAVE_PATH)

    # Only process test and dev datasets
    modes_to_process = ["train"]
    codes_to_process = ["P176", "P264", "P40", "P50"]

    for mode in modes_to_process:
        if not (data_path / mode).exists():
            print(f"Warning: {mode} directory not found, skipping...")
            continue

        print(f"Generation for {mode}...")
        os.makedirs(save_path / mode, exist_ok=True)

        mode_path = data_path / mode

        for file in os.listdir(mode_path):
            if any([(code in file) for code in codes_to_process]):
                data = load_json(mode_path / file)
                data_with_answers = process_one_file(pipeline, data)

                savefile = save_path / mode / file

                with open(savefile, "w", encoding="utf-8") as f:
                    json.dump(data_with_answers, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()