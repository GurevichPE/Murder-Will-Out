import json
import os
from pathlib import Path
from typing import Union, Dict, List
import torch
import transformers
from huggingface_hub import login
from tqdm.auto import tqdm
import random

SYSTEM_PROMPT = """
Your job is to answer an entity-centric question.
You need to answer with the correct entity, without any additional information.
"""

DATA_PATH = "./data/final_dataset"
SAVE_PATH = "./data/greedy_answers"
MAX_TRAIN_SIZE = 500


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


def load_json(path: Union[str, Path]) -> List[Dict]:
    """Load JSON data from file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def is_exact_match(greedy_answer: str, gold_answers: List[str]) -> bool:
    """Check if greedy answer exactly matches any of the gold answers."""
    if not greedy_answer or not gold_answers:
        return False
    
    # Normalize answers for comparison (strip whitespace, lowercase)
    normalized_greedy = greedy_answer.strip().lower()
    normalized_gold = [answer.strip().lower() for answer in gold_answers]
    
    return normalized_greedy in normalized_gold


def filter_data_by_split(data: List[Dict[str, str]], split: str) -> List[Dict[str, str]]:
    """Filter data based on split type and requirements."""
    filtered_data = []
    
    for entry in data:
        if "greedy_answer" not in entry or "answers" not in entry:
            continue
            
        if split == "train":
            # For training split, keep only exact matches
            if is_exact_match(entry["greedy_answer"], entry["answers"]):
                filtered_data.append(entry)
        else:
            # For dev and test splits, keep all entries
            filtered_data.append(entry)
    
    # For test split, limit to MAX_TEST_SIZE
    if split == "train" and len(filtered_data) > MAX_TRAIN_SIZE:
        # Shuffle to ensure random selection
        random.shuffle(filtered_data)
        filtered_data = filtered_data[:MAX_TRAIN_SIZE]
    
    return filtered_data


def make_one_prediction(pipeline:transformers.Pipeline, question:str):
    messages = get_messages(question)
    print(messages)
    outputs = pipeline(
        messages,
        max_new_tokens=8,
    )
    return outputs[0]["generated_text"][-1]


def load_pipeline() -> transformers.Pipeline:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    from key import KEY
    hf_token = KEY
    login(hf_token)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
        # Set greedy decoding: num_beams=1 and do_sample=False
        num_beams=1,
        do_sample=False,
    )
    return pipeline


def process_one_file(pipeline, data: List[Dict[str, str]], split: str) -> List[Dict[str, str]]:
    """Process one file by generating greedy answers and applying split-specific filtering."""
    # Generate greedy answers for all entries
    for i in tqdm(range(len(data))):
        entry = data[i]
        question = entry["question"]
        pred = make_one_prediction(pipeline, question)
        entry["greedy_answer"] = pred["content"]
        data[i] = entry
    
    # Apply filtering based on split type
    filtered_data = filter_data_by_split(data, split)
    
    print(f"Original size: {len(data)}, Filtered size: {len(filtered_data)}")
    
    return filtered_data


def main():
    """Main function to process all data splits with greedy answer generation and filtering."""
    # Set random seed for reproducibility
    random.seed(42)
    
    pipeline = load_pipeline()
    data_path = Path(DATA_PATH)
    save_path = Path(SAVE_PATH)
    splits = ["train"]


    for mode in os.listdir(data_path):
        print(f"Generation for {mode}...")
        os.makedirs(save_path / mode, exist_ok=True)

        mode_path = data_path / mode
        
        # Determine split type from directory name
        split = mode.lower()

        if split not in splits:
            continue
        
        for file in os.listdir(mode_path):
            print(f"Processing {file} in {split} split...")
            data = load_json(mode_path / file)
            data_with_greedy_answer = process_one_file(pipeline, data, split)

            savefile = save_path / mode / file 

            with open(savefile, "w", encoding="utf-8") as f: 
                json.dump(data_with_greedy_answer, f, indent=2, ensure_ascii=False)
                
            print(f"Saved {len(data_with_greedy_answer)} entries to {savefile}")

if __name__ == "__main__":
    main()