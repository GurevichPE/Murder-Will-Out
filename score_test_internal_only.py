"""
Simplified test scoring script for internal knowledge measurement only.

This script:
1. Loads the trained probe (best layer) from train_linear_probe.py 
2. Loads test data (with existing P(True) scores)
3. Extracts hidden states for test questions on-the-fly
4. Scores using the probe (internal method)
5. Calculates K metrics and compares internal vs P(True)
6. Saves results for analysis

Key assumptions:
- Test data already has P(True) scores calculated
- Trained probe exists in models/probes/
- Uses the best layer identified during training
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

from knowledge_calculation import calculate_k_q_with_answers


def load_json(path: Path) -> List[Dict]:
    """Load JSON data from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_trained_probe(models_dir: Path) -> Tuple[LogisticRegression, StandardScaler, int]:
    """Load the trained probe, scaler, and best layer."""
    # Load metrics to get best layer
    metrics_path = models_dir / "all_layers_metrics.json"
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
    
    best_layer = metrics_data["best_layer"]
    print(f"Loading trained probe from best layer: {best_layer}")
    
    # Load probe and scaler
    with open(models_dir / f"probe_layer_{best_layer}.pkl", "rb") as f:
        probe = pickle.load(f)
    
    with open(models_dir / f"scaler_layer_{best_layer}.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return probe, scaler, best_layer


def extract_hidden_states(
    model,
    tokenizer,
    question: str,
    answer: str,
    device: str = "cuda"
) -> Dict[int, np.ndarray]:
    """Extract hidden states for a (question, answer) pair using verification prompt."""
    # Same verification prompt as used in probe training
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
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    # Extract hidden states from all layers at the last token position
    hidden_states_dict = {}
    for layer_idx, layer_hidden_state in enumerate(outputs.hidden_states):
        last_hidden = layer_hidden_state[0, -1, :].cpu().to(torch.float16).numpy()
        hidden_states_dict[layer_idx] = last_hidden
    
    return hidden_states_dict


def score_test_questions(
    test_data: List[Dict],
    model,
    tokenizer,
    probe: LogisticRegression,
    scaler: StandardScaler,
    best_layer: int,
    device: str = "cuda"
) -> Tuple[List[Dict], Dict[str, float], Dict[str, float]]:
    """Score test questions using the trained probe."""
    print(f"\nScoring {len(test_data)} test questions using probe (layer {best_layer})...")
    
    scored_questions = []
    k_values_internal = []
    k_star_values_internal = []
    k_values_ptrue = []
    k_star_values_ptrue = []
    
    for entry in tqdm(test_data, desc="Processing test questions"):
        question = entry["question"]
        answer_labels = entry["answer_labels"]  # List of [answer, label] pairs
        golden_answer = entry["golden_answer"]
        p_true_scores = entry["p_true"]
        
        # Extract answers and labels
        answers = [pair[0] for pair in answer_labels]
        labels = [pair[1] == "EXACT_MATCH" for pair in answer_labels]
        
        # Skip questions with no correct/incorrect pairs
        num_correct = sum(labels)
        num_incorrect = len(labels) - num_correct
        if num_correct == 0 or num_incorrect == 0:
            continue
        
        # Calculate internal scores (probe)
        probe_scores = []
        for answer in answers:
            hidden_states = extract_hidden_states(model, tokenizer, question, answer, device)
            layer_hidden_state = hidden_states[best_layer].reshape(1, -1)
            scaled_hidden_state = scaler.transform(layer_hidden_state)
            probe_score = probe.predict_proba(scaled_hidden_state)[0, 1]
            probe_scores.append(float(probe_score))
        
        # Calculate K metrics for internal (probe) method
        k_q_internal = calculate_k_q_with_answers(answers, probe_scores, labels, aggregation_method="max")
        
        # Calculate K metrics for P(True) method  
        k_q_ptrue = calculate_k_q_with_answers(answers, p_true_scores, labels, aggregation_method="max")
        
        # Store results if both K values are valid
        if not (np.isnan(k_q_internal) or np.isnan(k_q_ptrue)):
            k_values_internal.append(k_q_internal)
            k_star_internal = 1.0 if k_q_internal == 1.0 else 0.0
            k_star_values_internal.append(k_star_internal)
            
            k_values_ptrue.append(k_q_ptrue)
            k_star_ptrue = 1.0 if k_q_ptrue == 1.0 else 0.0
            k_star_values_ptrue.append(k_star_ptrue)
            
            scored_questions.append({
                "question": question,
                "golden_answer": golden_answer,
                "num_answers": len(answers),
                "num_correct": num_correct,
                "num_incorrect": num_incorrect,
                "probe_scores": probe_scores,
                "p_true_scores": p_true_scores,
                "k_q_internal": float(k_q_internal),
                "k_star_internal": float(k_star_internal),
                "k_q_ptrue": float(k_q_ptrue),
                "k_star_ptrue": float(k_star_ptrue)
            })
    
    # Calculate overall metrics
    internal_metrics = {
        "num_questions": len(k_values_internal),
        "mean_k": float(np.mean(k_values_internal)) if k_values_internal else 0.0,
        "mean_k_star": float(np.mean(k_star_values_internal)) if k_star_values_internal else 0.0,
        "std_k": float(np.std(k_values_internal)) if k_values_internal else 0.0
    }
    
    ptrue_metrics = {
        "num_questions": len(k_values_ptrue),
        "mean_k": float(np.mean(k_values_ptrue)) if k_values_ptrue else 0.0,
        "mean_k_star": float(np.mean(k_star_values_ptrue)) if k_star_values_ptrue else 0.0,
        "std_k": float(np.std(k_values_ptrue)) if k_values_ptrue else 0.0
    }
    
    return scored_questions, internal_metrics, ptrue_metrics


def print_comparison_results(
    internal_metrics: Dict[str, float],
    ptrue_metrics: Dict[str, float],
    best_layer: int
):
    """Print comparison results showing internal vs P(True) performance."""
    print("\n" + "="*80)
    print("INTERNAL KNOWLEDGE EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nTest Set Results (Best Layer: {best_layer}):")
    print(f"{'Method':<20} {'Num Questions':<15} {'Mean K':<10} {'Mean K*':<10} {'Std K':<10}")
    print("-" * 75)
    
    # Results
    print(f"{'Internal (Probe)':<20} {internal_metrics['num_questions']:<15} "
          f"{internal_metrics['mean_k']:<10.4f} {internal_metrics['mean_k_star']:<10.4f} "
          f"{internal_metrics['std_k']:<10.4f}")
    
    print(f"{'P(True|q,a)':<20} {ptrue_metrics['num_questions']:<15} "
          f"{ptrue_metrics['mean_k']:<10.4f} {ptrue_metrics['mean_k_star']:<10.4f} "
          f"{ptrue_metrics['std_k']:<10.4f}")
    
    # Analysis
    print("\n" + "="*80)
    print("HIDDEN KNOWLEDGE ANALYSIS")
    print("="*80)
    
    internal_k = internal_metrics['mean_k']
    internal_k_star = internal_metrics['mean_k_star']
    ptrue_k = ptrue_metrics['mean_k']
    ptrue_k_star = ptrue_metrics['mean_k_star']
    
    gap_k = internal_k - ptrue_k
    gap_k_star = internal_k_star - ptrue_k_star
    
    relative_improvement = ((internal_k / ptrue_k) - 1) * 100 if ptrue_k > 0 else float('inf')
    
    print(f"Internal Knowledge (Probe):     K = {internal_k:.4f}, K* = {internal_k_star:.4f}")
    print(f"External Knowledge P(True):     K = {ptrue_k:.4f}, K* = {ptrue_k_star:.4f}")
    print(f"\nHidden Knowledge Gap:")
    print(f"  ΔK  = {gap_k:+.4f}")
    print(f"  ΔK* = {gap_k_star:+.4f}")
    print(f"  Relative improvement = {relative_improvement:.1f}%")
    
    # Detection
    hidden_knowledge_detected = internal_k > ptrue_k
    print(f"\nHidden Knowledge Detected: {'YES' if hidden_knowledge_detected else 'NO'}")
    
    if hidden_knowledge_detected:
        print(f"Evidence: Internal probe (K={internal_k:.4f}) exceeds P(True) method (K={ptrue_k:.4f})")
    else:
        print(f"No evidence: Internal probe (K={internal_k:.4f}) does not exceed P(True) method (K={ptrue_k:.4f})")


def main():
    """Main function to evaluate test set using trained probe."""
    # Configuration
    models_dir = Path("./models/probes")
    test_data_dir = Path("./data/sampled_labeled_answers_1000_temp1/test")
    output_dir = Path("./results")
    relations = ["P40", "P50", "P176", "P264"]
    
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device = "cuda"
    
    print("="*80)
    print("INTERNAL KNOWLEDGE TEST EVALUATION")
    print("="*80)
    
    # Load trained probe
    print("\n1. Loading trained probe...")
    probe, scaler, best_layer = load_trained_probe(models_dir)
    
    # Load model
    print(f"\n2. Loading model ({model_name})...")
    
    # Try to load HuggingFace token
    try:
        from key import KEY
        from huggingface_hub import login
        login(KEY)
    except ImportError:
        print("Warning: Could not load HuggingFace token from key.py")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16
    )
    model.eval()
    
    # Load test data
    print("\n3. Loading test data with P(True) scores...")
    all_test_data = []
    
    for relation in relations:
        test_file = test_data_dir / f"{relation}.test.json"
        if test_file.exists():
            print(f"  Loading {relation} test data...")
            data = load_json(test_file)
            # Add relation info
            for entry in data:
                entry["relation"] = relation
            all_test_data.extend(data)
        else:
            print(f"  Warning: {test_file} not found")
    
    print(f"  Total test questions: {len(all_test_data)}")
    
    # Score test data
    print(f"\n4. Scoring test data...")
    scored_questions, internal_metrics, ptrue_metrics = score_test_questions(
        all_test_data, model, tokenizer, probe, scaler, best_layer, device
    )
    
    # Print results
    print_comparison_results(internal_metrics, ptrue_metrics, best_layer)
    
    # Save results
    print(f"\n5. Saving results...")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results = {
        "metadata": {
            "model_name": model_name,
            "best_layer": best_layer,
            "relations": relations,
            "num_test_questions": len(all_test_data),
            "num_scored_questions": len(scored_questions)
        },
        "internal_metrics": internal_metrics,
        "ptrue_metrics": ptrue_metrics,
        "comparison": {
            "gap_k": internal_metrics["mean_k"] - ptrue_metrics["mean_k"],
            "gap_k_star": internal_metrics["mean_k_star"] - ptrue_metrics["mean_k_star"],
            "relative_improvement_percent": ((internal_metrics["mean_k"] / ptrue_metrics["mean_k"]) - 1) * 100 if ptrue_metrics["mean_k"] > 0 else 0,
            "hidden_knowledge_detected": internal_metrics["mean_k"] > ptrue_metrics["mean_k"]
        }
    }
    
    # Save summary results
    results_path = output_dir / "internal_test_evaluation.json"
    save_json(results, results_path)
    print(f"Summary results saved to: {results_path}")
    
    # Save detailed per-question results
    detailed_path = output_dir / "internal_test_questions_detailed.json"
    save_json(scored_questions, detailed_path)
    print(f"Detailed question results saved to: {detailed_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
