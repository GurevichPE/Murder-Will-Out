"""
Generate training data for linear probes (internal knowledge scoring).

This script implements KNOWLEDGE-AWARE PROBING as described in the paper's Appendix A.8.2:
- Train ONLY on questions where the greedy answer is correct (model knows the answer)
- For each such question, create MULTIPLE training pairs:
  * Multiple positive examples: greedy correct + any other sampled correct answers
  * Multiple negative examples: several sampled incorrect answers
  
This ensures the probe learns from questions where the model has knowledge,
providing stronger discriminative signal for distinguishing correct from incorrect answers.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_json(path: Path) -> List[Dict]:
    """Load JSON data from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: List[Dict], path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_pickle(data: List[Dict], path: Path) -> None:
    """Save data to pickle file."""
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_stats(stats: Dict, path: Path) -> None:
    """Save statistics to JSON file."""
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def is_greedy_correct(
    greedy_answer: str,
    answer_labels: List[List[str]]
) -> bool:
    """
    Check if the greedy answer is correct.
    
    Args:
        greedy_answer: Answer from greedy decoding
        answer_labels: List of [answer, label] pairs from sampling
        
    Returns:
        True if greedy answer is correct
    """
    for answer, label in answer_labels:
        if answer.strip().lower() == greedy_answer.strip().lower():
            return label == "EXACT_MATCH"
    return False


def get_correct_answers(
    greedy_answer: str,
    answer_labels: List[List[str]]
) -> List[str]:
    """
    Get all correct answers (greedy + any correct sampled ones).
    
    Args:
        greedy_answer: Answer from greedy decoding (already verified as correct)
        answer_labels: List of [answer, label] pairs from sampling
        
    Returns:
        List of unique correct answers
    """
    correct_answers = [greedy_answer]
    
    # Add other sampled correct answers (different from greedy)
    for answer, label in answer_labels:
        if label == "EXACT_MATCH":
            # Normalize for comparison
            if answer.strip().lower() != greedy_answer.strip().lower():
                if answer not in correct_answers:
                    correct_answers.append(answer)
    
    return correct_answers


def get_incorrect_answers(
    answer_labels: List[List[str]],
    max_negatives: int = 5
) -> List[str]:
    """
    Get multiple incorrect answers from the labeled answers.
    
    Args:
        answer_labels: List of [answer, label] pairs
        max_negatives: Maximum number of negative examples to return
        
    Returns:
        List of incorrect answers (up to max_negatives)
    """
    incorrect_answers = []
    for answer, label in answer_labels:
        if label == "False":
            if answer not in incorrect_answers:
                incorrect_answers.append(answer)
                if len(incorrect_answers) >= max_negatives:
                    break
    return incorrect_answers


def extract_hidden_states(
    model,
    tokenizer,
    question: str,
    answer: str,
    device: str = "cuda"
) -> Dict[int, np.ndarray]:
    """
    Extract hidden states from all layers for a (question, answer) pair.
    
    From paper Appendix A.8: "In early experiments, we also tested the sequence 
    from §A.8.2, where the model is prompted to verify a as the answer to q, 
    but the classifier's performance was similar in both cases."
    
    Uses the P(True) verification prompt from Appendix A.8.2 (for Llama/Mistral).
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        question: The question text
        answer: The answer text
        device: Device to run on
        
    Returns:
        Dictionary mapping layer index to hidden state (numpy array)
    """
    # Prompts from Appendix A.8.2 (P(True) verification prompts for Llama and Mistral)
    system_prompt = "Your job is to evaluate if a proposed answer to an entity-centric question is correct."
    
    user_prompt = ("Here is the question and the proposed answer.\n"
                  "```\n"
                  f"Question: {question}\n"
                  f"Proposed Answer: {answer}\n"
                  "```\n"
                  "Is the proposed answer:\n"
                  "A: CORRECT\n"
                  "B: INCORRECT\n"
                  "Just return the letters \"A\" or \"B\", with no text around it.")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Apply chat template for verification sequence
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Get hidden states from all layers
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    # Extract hidden states from the last token position
    hidden_states_dict = {}
    for layer_idx, layer_hidden_state in enumerate(outputs.hidden_states):
        # Take the last token's hidden state
        last_hidden = layer_hidden_state[0, -1, :].cpu().to(torch.float16).numpy()
        hidden_states_dict[layer_idx] = last_hidden
    
    return hidden_states_dict


def process_one_relation_train(
    greedy_data: List[Dict],
    labeled_data: List[Dict],
    model,
    tokenizer,
    device: str = "cuda",
    max_negatives_per_question: int = 5,
    max_questions: int = None
) -> Tuple[List[Dict], Dict]:
    """
    Process train split: KNOWLEDGE-AWARE PROBING.
    
    Strategy:
    - ONLY use questions where greedy answer is correct (model knows the answer)
    - For each such question, create multiple training pairs:
      * Positive examples: greedy correct answer + any other sampled correct answers
      * Negative examples: multiple sampled incorrect answers (up to max_negatives_per_question)
    
    Args:
        greedy_data: Data with greedy answers
        labeled_data: Data with labeled sampled answers
        model: The language model
        tokenizer: The tokenizer
        device: Device to run on
        max_negatives_per_question: Max number of negative examples per question
        max_questions: Maximum number of questions to process (for limiting training set size)
        
    Returns:
        Tuple of (training_examples, statistics)
    """
    training_examples = []
    stats = {
        "total_questions_examined": 0,
        "greedy_correct_questions": 0,
        "questions_with_multiple_correct": 0,
        "questions_with_insufficient_incorrect": 0,
        "final_questions_used": 0,
        "total_positive_examples": 0,
        "total_negative_examples": 0
    }
    
    # Create a mapping from question to labeled data
    labeled_dict = {entry["question"]: entry for entry in labeled_data}
    
    questions_used = 0
    
    for greedy_entry in tqdm(greedy_data, desc="Processing questions"):
        if max_questions and questions_used >= max_questions:
            break
            
        question = greedy_entry["question"]
        greedy_answer = greedy_entry["greedy_answer"]
        
        # Find corresponding labeled entry
        if question not in labeled_dict:
            continue
        
        stats["total_questions_examined"] += 1
        labeled_entry = labeled_dict[question]
        answer_labels = labeled_entry["answer_labels"]
        golden_answer = labeled_entry["golden_answer"]
        
        # KNOWLEDGE-AWARE FILTERING: Only use questions where greedy is correct
        if not is_greedy_correct(greedy_answer, answer_labels):
            continue
        
        stats["greedy_correct_questions"] += 1
        
        # Get all correct answers (greedy + any other sampled correct ones)
        correct_answers = get_correct_answers(greedy_answer, answer_labels)
        
        if len(correct_answers) > 1:
            stats["questions_with_multiple_correct"] += 1
        
        # Get multiple incorrect answers
        incorrect_answers = get_incorrect_answers(answer_labels, max_negatives_per_question)
        
        if len(incorrect_answers) == 0:
            stats["questions_with_insufficient_incorrect"] += 1
            continue
        
        # Extract hidden states for all correct answers (positives)
        correct_hidden_states_list = []
        for correct_ans in correct_answers:
            hidden_states = extract_hidden_states(
                model, tokenizer, question, correct_ans, device
            )
            correct_hidden_states_list.append((correct_ans, hidden_states))
        
        # Extract hidden states for all incorrect answers (negatives)
        incorrect_hidden_states_list = []
        for incorrect_ans in incorrect_answers:
            hidden_states = extract_hidden_states(
                model, tokenizer, question, incorrect_ans, device
            )
            incorrect_hidden_states_list.append((incorrect_ans, hidden_states))
        
        # Create training examples
        # Positive examples
        for correct_ans, hidden_states in correct_hidden_states_list:
            training_examples.append({
                "question": question,
                "answer": correct_ans,
                "label": 1,  # Correct
                "hidden_states": {k: v.tolist() for k, v in hidden_states.items()},
                "golden_answer": golden_answer,
                "is_greedy": (correct_ans == greedy_answer)
            })
            stats["total_positive_examples"] += 1
        
        # Negative examples
        for incorrect_ans, hidden_states in incorrect_hidden_states_list:
            training_examples.append({
                "question": question,
                "answer": incorrect_ans,
                "label": 0,  # Incorrect
                "hidden_states": {k: v.tolist() for k, v in hidden_states.items()},
                "golden_answer": golden_answer,
                "is_greedy": False
            })
            stats["total_negative_examples"] += 1
        
        stats["final_questions_used"] += 1
        questions_used += 1
    
    return training_examples, stats


def process_one_relation_dev_test(
    labeled_data: List[Dict],
    model,
    tokenizer,
    device: str = "cuda",
    max_answers_per_question: int = 100,
    max_questions: int = None
) -> Tuple[List[Dict], Dict]:
    """
    Process dev/test split: ALL questions with up to 100 answers each.
    
    Strategy:
    - Process ALL questions (no filtering)
    - For each question:
      * Take up to 100 sampled answers
      * If no positive answers found, add golden answer as positive
      * Extract hidden states for all answers
    
    Args:
        labeled_data: Data with labeled sampled answers
        model: The language model
        tokenizer: The tokenizer
        device: Device to run on
        max_answers_per_question: Maximum number of answers per question (default 100)
        max_questions: Maximum number of questions to process
        
    Returns:
        Tuple of (training_examples, statistics)
    """
    training_examples = []
    stats = {
        "total_questions_examined": 0,
        "questions_with_positives": 0,
        "questions_without_positives": 0,
        "golden_answer_added": 0,
        "final_questions_used": 0,
        "total_positive_examples": 0,
        "total_negative_examples": 0,
        "total_examples": 0
    }
    
    questions_used = 0
    
    for labeled_entry in tqdm(labeled_data, desc="Processing questions"):
        if max_questions and questions_used >= max_questions:
            break
        
        question = labeled_entry["question"]
        answer_labels = labeled_entry["answer_labels"]
        golden_answer = labeled_entry["golden_answer"]
        
        stats["total_questions_examined"] += 1
        
        # Take up to max_answers_per_question
        answers_to_process = answer_labels[:max_answers_per_question]
        
        # Check if we have any positive answers
        has_positive = any(label == "EXACT_MATCH" for _, label in answers_to_process)
        
        if has_positive:
            stats["questions_with_positives"] += 1
        else:
            stats["questions_without_positives"] += 1
            # Add golden answer as positive if not already present
            golden_already_present = any(
                answer.strip().lower() == golden_answer.strip().lower() 
                for answer, _ in answers_to_process
            )
            
            if not golden_already_present:
                # Add golden answer at the beginning
                answers_to_process = [(golden_answer, "EXACT_MATCH")] + answers_to_process[:max_answers_per_question-1]
                stats["golden_answer_added"] += 1
        
        # Extract hidden states for all answers
        for answer, label in answers_to_process:
            hidden_states = extract_hidden_states(
                model, tokenizer, question, answer, device
            )
            
            is_correct = (label == "EXACT_MATCH")
            
            training_examples.append({
                "question": question,
                "answer": answer,
                "label": 1 if is_correct else 0,
                "hidden_states": {k: v.tolist() for k, v in hidden_states.items()},
                "golden_answer": golden_answer,
                "is_greedy": False  # Not using greedy for dev/test
            })
            
            if is_correct:
                stats["total_positive_examples"] += 1
            else:
                stats["total_negative_examples"] += 1
            
            stats["total_examples"] += 1
        
        stats["final_questions_used"] += 1
        questions_used += 1
    
    return training_examples, stats


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16
    )
    
    model.eval()
    print("Model loaded successfully")
    
    return model, tokenizer


def main(splits=["train", "dev"]):
    """
    Main function to generate probe training data.
    
    Strategies by split:
    - train: Knowledge-aware probing (only greedy correct questions)
    - dev/test: All questions, up to 100 answers each (add golden if no positives)
    
    Args:
        splits: List of splits to process (e.g., ["train", "dev", "test"])
    """
    # Configuration
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device = "cuda"
    from huggingface_hub import login
    hf_token = # INSERT TOKEN
    login(hf_token)
    relations = ["P264"]
    
    # Train-specific settings
    max_negatives_per_question = 5  # Multiple negatives per question for train
    max_questions_per_relation_train = 500  # Limit to 500 questions per relation for train
    
    # Dev/test-specific settings
    max_answers_per_question_dev_test = 100  # Up to 100 answers per question for dev/test
    
    greedy_dir = Path("./data/greedy_answers")
    labeled_dir = Path("./data/sampled_labeled_answers_1000_temp1")
    output_dir = Path("./data/probe_training_data")
    
    # Parse command-line arguments for splits
    if len(sys.argv) > 1:
        splits = sys.argv[1:]
    else:
        splits = ["train"]  # Default to train only
    
    print(f"Generating probe data for splits: {splits}")
    print(f"\nStrategies:")
    print(f"  train: Knowledge-aware (greedy correct only, multiple pos/neg per question)")
    print(f"  dev/test: All questions (up to {max_answers_per_question_dev_test} answers, add golden if needed)")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # Process each relation and split
    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}")
        
        for relation in relations:
            print(f"\nProcessing {relation} ({split})...")
            
            # Load data
            labeled_path = labeled_dir / split / f"{relation}.{split}.json"
            
            if not labeled_path.exists():
                print(f"  Skipping {relation} ({split}): labeled file not found")
                continue
            
            labeled_data = load_json(labeled_path)
            
            # Process based on split type
            if split == "train":
                # Need greedy data for train
                greedy_path = greedy_dir / split / f"{relation}.{split}.json"
                if not greedy_path.exists():
                    print(f"  Skipping {relation} ({split}): greedy file not found")
                    continue
                
                greedy_data = load_json(greedy_path)
                
                # Knowledge-aware probing
                training_examples, stats = process_one_relation_train(
                    greedy_data,
                    labeled_data,
                    model,
                    tokenizer,
                    device,
                    max_negatives_per_question,
                    max_questions_per_relation_train
                )
                
                # Print train-specific stats
                print(f"\n  Statistics for {relation} ({split}):")
                print(f"    Questions examined: {stats['total_questions_examined']}")
                print(f"    Greedy correct: {stats['greedy_correct_questions']}")
                print(f"    Questions with multiple correct answers: {stats['questions_with_multiple_correct']}")
                print(f"    Questions with insufficient incorrect: {stats['questions_with_insufficient_incorrect']}")
                print(f"    Final questions used: {stats['final_questions_used']}")
                print(f"    Total positive examples: {stats['total_positive_examples']}")
                print(f"    Total negative examples: {stats['total_negative_examples']}")
                print(f"    Total examples: {len(training_examples)}")
                
            else:  # dev or test
                # All questions with up to 100 answers
                training_examples, stats = process_one_relation_dev_test(
                    labeled_data,
                    model,
                    tokenizer,
                    device,
                    max_answers_per_question_dev_test,
                    max_questions=None  # Process all questions
                )
                
                # Print dev/test-specific stats
                print(f"\n  Statistics for {relation} ({split}):")
                print(f"    Questions examined: {stats['total_questions_examined']}")
                print(f"    Questions with positives: {stats['questions_with_positives']}")
                print(f"    Questions without positives: {stats['questions_without_positives']}")
                print(f"    Golden answer added: {stats['golden_answer_added']}")
                print(f"    Final questions used: {stats['final_questions_used']}")
                print(f"    Total positive examples: {stats['total_positive_examples']}")
                print(f"    Total negative examples: {stats['total_negative_examples']}")
                print(f"    Total examples: {stats['total_examples']}")
            
            # Save results
            # Use pickle for dev/test (large files), JSON for train (smaller, human-readable)
            if split == "train":
                output_path = output_dir / f"{relation}.{split}.probe_data.json"
                save_json(training_examples, output_path)
                print(f"\n  Saved to: {output_path} (JSON)")
            else:
                output_path = output_dir / f"{relation}.{split}.probe_data.pkl"
                save_pickle(training_examples, output_path)
                print(f"\n  Saved to: {output_path} (Pickle)")
            
            stats_path = output_dir / f"{relation}.{split}.probe_data.stats.json"
            save_stats(stats, stats_path)
    
    print(f"\n{'='*70}")
    print("Probe data generation complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
