import json
import os
from pathlib import Path
from typing import Union, Dict, List
import torch
import transformers
from huggingface_hub import login
from tqdm.auto import tqdm

SYSTEM_PROMPT = """
Your job is to answer an entity-centric question.
You need to answer with the correct entity, without any additional information.
"""

DATA_PATH = "./data/final_dataset"
SAVE_PATH = "./data/greedy_answers"


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


def make_one_prediction(pipeline:transformers.Pipeline, question:str):
    messages = get_messages(question)
    print(messages)
    outputs = pipeline(
        messages,
        max_new_tokens=32,
    )
    return outputs[0]["generated_text"][-1]


def load_pipeline() -> transformers.Pipeline:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hf_token = # INSERT TOKEN
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


def process_one_file(pipeline, data:List[Dict[str, str]]) -> List[Dict[str, str]]:
    for i in tqdm(range(len(data))):
        entry = data[i]
        question = entry["question"]
        pred = make_one_prediction(pipeline, question)
        entry["greedy_answer"] = pred["content"]
        data[i] = entry
    return data


def main():
    pipeline = load_pipeline()
    data_path = Path(DATA_PATH)
    save_path = Path(SAVE_PATH)

    for mode in os.listdir(data_path):
        print(f"Generation for {mode}...")
        os.makedirs(save_path / mode, exist_ok=True)

        mode_path = data_path / mode

        for file in os.listdir(mode_path):
            data = load_json(mode_path / file)
            data_with_greedy_answer = process_one_file(pipeline, data)

            savefile = save_path / mode / file 

            with open(savefile, "w", encoding="utf-8") as f: 
                json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()