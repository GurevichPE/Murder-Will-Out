import json
import os
from pathlib import Path
from typing import Union, Dict, List, Tuple
import torch
import transformers
from huggingface_hub import login
from tqdm.auto import tqdm
import re

DATA_PATH = "./data/sampled_answers_1000_temp1"
SAVE_PATH = "./data/sampled_labeled_answers_1000_temp1"


SYSTEM_INSTRUCTION = (
    "You are a strict judge. I will show you some QUESTION."
    "Next, I will show you CANDIDATE answer and GOLDEN answer."
    "Compare a GOLDEN answer and a CANDIDATE answer. "
    "If a CANDIDATE and GOLDEN answer are the same, return True. "
    "If a CANDIDATE answer differs with GOLDEN answer, return False. "
    "Spelling errors, synonyms, abbreviations, or hedging (e.g., ”it is possible that”) should not alter the judgement if the subject referred to in the CANDIDATE answer matches the GOLDEN answer."
    "Output exactly one token — either:\n\nTrue\n\nor\n\nFalse\n\n"
    "Output nothing else, no punctuation, no explanation, no list. Output must be exactly one of the two tokens: True of False."
)


FEW_SHOT_EXAMPLES = (
    "Example 1:\nQUESTION: \"Who is the author of The Monk?\"\nGOLDEN: \"Matthew Gregory Lewis\"\nCANDIDATE: \"Matthew Lewis\"\nTrue\n\n"
    "Example 2:\nQUESTION: \"Which company is British Rail 18000 produced by?\"\nGOLDEN: \"Brown, Boveri & Cie\"\nCANDIDATE: \"Adtranz\"\nFalse\n\n"
)


def get_prompt(question:str, gold_answer:str, answer:str, code:str) -> str:
    q = f"QUESTION: {question}"
    g = f"GOLDEN: {gold_answer}"
    c = f"CANDIDATE: {answer}"
    postfix = "Are GOLDEN and CANDIDATE equivalent? Reply with exactly one token: True or False."
    prompt = f"{SYSTEM_INSTRUCTION}\n{FEW_SHOT_EXAMPLES}\n```{q}\n{g}\n{c}\n```\n{postfix}"
    return prompt

def get_messages(question:str, gold_answer:str, answer:str, code:str) -> List[Dict[str,str]]:
    user_query = get_prompt(question, gold_answer, answer, code)
    messages = [
        {"role": "user", "content": user_query},
    ]
    return messages

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.strip()
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation (simple)
    return s


def load_pipeline() -> transformers.Pipeline:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hf_token = # INSERT TOKEN
    login(hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        do_sample=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = model.config.eos_token_id[0]
    return tokenizer, model

def load_json(path:Union[str, Path]) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def llm_judge(model, tokenizer, messages_batch, batch_size: int = 32):
    from transformers import pipeline
    """
    Process a batch of messages using the language model with configurable batch size.

    Args:
        model: The language model
        tokenizer: The tokenizer
        messages_batch: List of message lists, where each message list is a conversation
        batch_size: Number of conversations to process in each batch (default: 32)

    Returns:
        List of strings, each being the model's judgment for the corresponding conversation
    """
    all_results = []

    generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",   # pipeline will use the already-loaded model/device map
            trust_remote_code=True,
        )

    # Process messages in batches
    for i in range(0, len(messages_batch), batch_size):
        batch_messages = messages_batch[i:i + batch_size]
        results = generator(
                batch_messages,
                max_new_tokens=3,
                do_sample=False,
                temperature=0,
                top_p=0.1,
                return_full_text=False,
                batch_size=min(len(batch_messages), 500)
            )
        
        for r in results:
            gen = r[0]["generated_text"].strip()
            token = None
            # split by whitespace and punctuation
            for part in gen.replace("\n", " ").split():
                if part == "True" or part == "False":
                    token = part
                    break
                # sometimes the model outputs "→" or "True." so strip punctuation:
                pstripped = part.strip('.,;:→-')
                if pstripped in ("True", "False"):
                    token = pstripped
                    break
            if token is None:
                # fallback deterministic decision: treat as False (or you can retry)
                token = "False"
            all_results.append(token)
    return all_results


def process_one_question(data: Dict[str,str], code: str, model, tokenizer, batch_size: int = 32) -> Dict:
    golden_answer = data["answers"][0]
    sampled_answers = data["sampled_answers"]
    question = data["question"]
    normed_gold = normalize_text(golden_answer)
    pre_selected_labels = []
    pre_selected_ids = []
    ids_for_llm_judge = []
    messages_for_llm_judge = []

    for i, a in enumerate(sampled_answers):
        a_norm = normalize_text(a)
        if (a_norm == normed_gold) or (a_norm in normed_gold) or(normed_gold in a_norm):
            pre_selected_labels.append("EXACT_MATCH")
            pre_selected_ids.append(i)
        else:
            msg = get_messages(question, golden_answer, a, code)
            messages_for_llm_judge.append(msg)
            ids_for_llm_judge.append(i)
        
    llm_judge_answers = llm_judge(model, tokenizer, messages_for_llm_judge, batch_size)

    # Combine pre-selected labels with LLM judge results
    final_labels = [""] * len(sampled_answers)
    for label, idx in zip(pre_selected_labels, pre_selected_ids):
        final_labels[idx] = label

    for answer, idx in zip(llm_judge_answers, ids_for_llm_judge):
        final_labels[idx] = answer

    answer_label_dict = [(a, l) for a, l in zip(sampled_answers, final_labels)]
    result = {
        "question": question,
        "golden_answer": golden_answer,
        "answer_labels": answer_label_dict,
    }

    return result


def main(batch_size: int = 32):
    tokenizer, model = load_pipeline()

    data_path = Path(DATA_PATH)
    save_path = Path(SAVE_PATH)

    # Only process test and dev datasets
    modes_to_process = ["dev"]

    codes_to_process = ["P176"]

    print(f"Using batch size: {batch_size}")

    for mode in modes_to_process:
        if not (data_path / mode).exists():
            print(f"Warning: {mode} directory not found, skipping...")
            continue

        print(f"Processing {mode}...")
        os.makedirs(save_path / mode, exist_ok=True)

        mode_path = data_path / mode

        for file in os.listdir(mode_path):
            code = file[:file.find(".")]
            if code in codes_to_process:

                # Load data - expecting array of questions
                data_list = load_json(mode_path / file)

                # Process each question in the file
                processed_questions = []
                for data in tqdm(data_list):
                    result = process_one_question(data, code, model, tokenizer, batch_size)
                    processed_questions.append(result)

                # Save individual file results
                savefile = save_path / mode / file
                with open(savefile, "w", encoding="utf-8") as f:
                    json.dump(processed_questions, f, indent=2, ensure_ascii=False)

    print("Processing completed!")

if __name__ == "__main__":
    main(batch_size=200)