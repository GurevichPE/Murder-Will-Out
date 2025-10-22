import json
import os
from pathlib import Path
from typing import Union, Dict, List, Tuple
import torch
import transformers
from huggingface_hub import login
from tqdm.auto import tqdm
import re

SYSTEM_PROMPT = "Your job is to evaluate if a proposed answer to an entity-centric question is correct."

def get_user_prompt(question, answer):
    prompt =f"""Here is the question and the proposed answer.
        ```
        Question: {question}
        Proposed Answer: {answer}
        ```
        Is the proposed answer:
        A: CORRECT
        B: INCORRECT
        Just return the letters ”A” or ”B”, with no text around it."""
    return prompt

def get_messages(question:str, answer:str) -> List[Dict[str,str]]:
    user_query = get_user_prompt(question, answer)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    return messages


def get_p_true(model, tokenizer, question: str, answers: List[str], batch_size: int = 160) -> List[float]:
    """
    Calculate P(True) for each proposed answer - the probability that the model
    would respond "A" (CORRECT) when asked to evaluate if the answer is correct.

    Returns list of probabilities, one for each answer in the answers list.
    Each probability represents P(model responds "A" | question, proposed_answer)
    """
    import numpy as np
    # Create messages for each question-answer pair
    messages_list = [get_messages(question, answer) for answer in answers]

    final_probas = np.zeros(len(answers))

    # Process in batches to handle memory constraints
    for i in range(0, len(messages_list), batch_size):
        batch_messages = messages_list[i:i + batch_size]

        # Apply chat template and tokenize batch
        batch_texts = [
            tokenizer.apply_chat_template(
                [msg_list],
                tokenize=False,
                add_generation_prompt=True,
            )[0]  # apply_chat_template returns a list, take first element
            for msg_list in batch_messages
        ]

        # Tokenize the prompts (without adding "A" yet)
        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False  # Don't add special tokens since they're already in the chat template
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits
            token_a = tokenizer.encode("A", add_special_tokens=False)[0]
            token_b = tokenizer.encode("B", add_special_tokens=False)[0]
            last_logits = logits[:, -1, :]
            a_logits = last_logits[:, [token_a, token_b]]
            token_probs = torch.softmax(a_logits, dim=-1)
            a_probs = token_probs[:, 0]
            final_probas[i:i + batch_size] = a_probs.cpu().to(torch.float16).numpy()
    return final_probas.tolist()


def load_pipeline() -> Tuple[transformers.PreTrainedTokenizer, transformers.PreTrainedModel]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hf_token = # INSERT TOKEN
    login(hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        # Handle different types of eos_token_id (int, tensor, list)
        eos_token_id = model.config.eos_token_id
        if isinstance(eos_token_id, torch.Tensor):
            tokenizer.pad_token_id = eos_token_id.item()
        elif isinstance(eos_token_id, (list, tuple)) and len(eos_token_id) > 0:
            tokenizer.pad_token_id = eos_token_id[0]
        else:
            # Assume it's a scalar value
            tokenizer.pad_token_id = eos_token_id
    return tokenizer, model

def load_json(path:Union[str, Path]) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def process_one_code(code_path: Path, model, tokenizer) -> List[Dict]:
    """
    Process a single JSON file containing question-answer pairs.

    Args:
        code_path: Path to the JSON file
        model: The loaded language model
        tokenizer: The model's tokenizer

    Returns:
        Updated data with p_true probabilities added
    """
    data = load_json(code_path)
    for i in tqdm(range(len(data))):
        entry = data[i]
        question = entry["question"]
        answers = entry["answer_labels"]
        results = get_p_true(model, tokenizer, question, answers)
        entry["p_true"] = results
        data[i] = entry
    return data

def save_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    DATA_PATH = Path("./data/sampled_labeled_answers_1000_temp1")

    tokenizer, model = load_pipeline()

    for split in os.listdir(DATA_PATH):
        split_path = DATA_PATH / split
        if split != "test":
            continue
        for code in os.listdir(split_path):
            if ("stats" not in code) and (("176" in code) or ("264" in code)):
                code_path = split_path / code
                print(f"Processing {code}...")
                data = process_one_code(code_path, model, tokenizer)
                save_data(data, code_path)

if __name__ == "__main__":
    main()