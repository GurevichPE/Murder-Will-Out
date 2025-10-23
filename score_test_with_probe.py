"""
Test set evaluation script using trained linear probes for internal knowledge scoring.

This script implements the paper's methodology for measuring hidden knowledge by comparing:
- Internal knowledge: Measured using trained linear probes on hidden states
- External knowledge: Measured using token-level probabilities (P(a|q), P(True|q,a))

The script:
1. Loads the trained probe (best layer) and scaler from train_linear_probe.py
2. Loads test data (either pre-computed hidden states or computes on-the-fly)  
3. Scores all test answers using both internal and external methods
4. Calculates knowledge metrics (K, K*) for comparison
5. Generates results showing evidence of hidden knowledge

Key Features:
- Supports both pre-computed and on-the-fly hidden state extraction
- Implements all external scoring baselines from the paper
- Proper answer deduplication for accurate K/K* calculations
- Comprehensive comparison analysis following paper methodology
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

from knowledge_calculation import calculate_k_q_with_answers
from external_scoring import get_p_true, load_pipeline


def load_json(path: Path) -> List[Dict]:
    """Load JSON data from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> List[Dict]:
    """Load pickle data from file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: dict, path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_trained_probe(models_dir: Path) -> Tuple[LogisticRegression, StandardScaler, int, Dict]:
    """
    Load the trained probe, scaler, and metadata from train_linear_probe.py output.
    
    Args:
        models_dir: Directory containing trained probe files
        
    Returns:
        Tuple of (probe, scaler, best_layer, metrics)
    """
    # Load metrics to get best layer
    metrics_path = models_dir / "all_layers_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
    
    best_layer = metrics_data["best_layer"]
    best_metrics = metrics_data["best_metrics"]
    
    print(f"Loading trained probe from best layer: {best_layer}")
    print(f"Best layer dev performance: K={best_metrics['dev_k_full']:.4f}, K*={best_metrics['dev_k_star_full']:.4f}")
    
    # Load probe and scaler
    probe_path = models_dir / f"probe_layer_{best_layer}.pkl"
    scaler_path = models_dir / f"scaler_layer_{best_layer}.pkl"
    
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    with open(probe_path, "rb") as f:
        probe = pickle.load(f)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    return probe, scaler, best_layer, best_metrics


def load_test_data_precomputed(
    probe_data_dir: Path,
    relations: List[str]
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[Dict]]:
    """
    Load pre-computed test hidden states from probe_training_data/ directory.
    
    Args:
        probe_data_dir: Directory containing probe data files
        relations: List of relation codes
        
    Returns:
        Tuple of (hidden_states_by_layer, labels, metadata)
    """
    print("\nLoading pre-computed test data...")
    
    all_examples = []
    
    for relation in relations:
        file_path = probe_data_dir / f"{relation}.test.probe_data.pkl"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
        
        print(f"Loading {relation} test data from pickle...")
        data = load_pickle(file_path)
        all_examples.extend(data)
    
    if not all_examples:
        raise FileNotFoundError("No test probe data found! Run generate_data_for_probe.py with 'test' split first.")
    
    print(f"Loaded {len(all_examples)} total test examples from {len(relations)} relations")
    
    # Organize by layer (same structure as train_linear_probe.py)
    num_layers = len(all_examples[0]["hidden_states"])
    hidden_states_by_layer = {i: [] for i in range(num_layers)}
    labels = []
    metadata = []
    
    for example in all_examples:
        # Extract hidden states for each layer
        hs_dict = example["hidden_states"]
        for layer_idx in range(num_layers):
            if layer_idx in hs_dict:
                hidden_state = hs_dict[layer_idx]
            else:
                hidden_state = hs_dict[str(layer_idx)]
            hidden_states_by_layer[layer_idx].append(hidden_state)
        
        labels.append(example["label"])
        metadata.append({
            "question": example["question"],
            "answer": example["answer"],
            "golden_answer": example["golden_answer"]
        })
    
    # Convert to numpy arrays
    for layer_idx in range(num_layers):
        hidden_states_by_layer[layer_idx] = np.array(hidden_states_by_layer[layer_idx])
    
    labels = np.array(labels)
    
    print(f"Test hidden states shape per layer: {hidden_states_by_layer[0].shape}")
    print(f"Test labels shape: {labels.shape}")
    print(f"Test label distribution: {np.sum(labels == 1)} positive, {np.sum(labels == 0)} negative")
    
    return hidden_states_by_layer, labels, metadata


def load_test_data_original(
    labeled_data_dir: Path,
    relations: List[str]
) -> List[Dict]:
    """
    Load original test data from labeled answers files.
    
    Args:
        labeled_data_dir: Directory containing labeled answer files
        relations: List of relation codes
        
    Returns:
        List of test entries with question, answer_labels, golden_answer
    """
    print("\nLoading original test data...")
    
    all_test_data = []
    
    for relation in relations:
        file_path = labeled_data_dir / "test" / f"{relation}.test.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
        
        print(f"Loading {relation} test data...")
        data = load_json(file_path)
        
        # Add relation info for tracking
        for entry in data:
            entry["relation"] = relation
        
        all_test_data.extend(data)
    
    print(f"Loaded {len(all_test_data)} test questions from {len(relations)} relations")
    
    return all_test_data


def extract_hidden_states_for_qa_pair(
    model,
    tokenizer,
    question: str,
    answer: str,
    device: str = "cuda"
) -> Dict[int, np.ndarray]:
    """
    Extract hidden states from all layers for a (question, answer) pair.
    Uses the same verification prompt as in generate_data_for_probe.py.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        question: Question text
        answer: Answer text
        device: Device to run on
        
    Returns:
        Dictionary mapping layer index to hidden state
    """
    # Same verification prompt as in generate_data_for_probe.py
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
    
    # Extract hidden states from the last token position
    hidden_states_dict = {}
    for layer_idx, layer_hidden_state in enumerate(outputs.hidden_states):
        last_hidden = layer_hidden_state[0, -1, :].cpu().to(torch.float16).numpy()
        hidden_states_dict[layer_idx] = last_hidden
    
    return hidden_states_dict


def calculate_external_scores(
    model,
    tokenizer, 
    question: str,
    answers: List[str],
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    Calculate external scoring baselines: P(a|q) and P(True|q,a).
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        question: Question text
        answers: List of answer texts
        device: Device to run on
        
    Returns:
        Dictionary with external scores for each method
    """
    # P(True|q,a) - verification score using external_scoring.py
    p_true_scores = get_p_true(model, tokenizer, question, answers, batch_size=32)
    
    # P(a|q) and P(a|q)^(1/|a|) - generation likelihood scores
    p_aq_scores = []
    p_aq_normalized_scores = []
    
    # Create generation prompt (same as in paper methodology)
    generation_prompt = f"Question: {question}\nAnswer:"
    
    for answer in answers:
        # Tokenize prompt and answer separately
        prompt_inputs = tokenizer(generation_prompt, return_tensors="pt", add_special_tokens=True).to(device)
        
        # Create the full sequence (prompt + answer)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        full_input_ids = torch.cat([
            prompt_inputs.input_ids,
            torch.tensor([answer_tokens], device=device)
        ], dim=1)
        
        # Calculate token-level probabilities
        with torch.no_grad():
            outputs = model(input_ids=full_input_ids)
            logits = outputs.logits
            
            # Get probabilities for answer tokens
            answer_start_idx = prompt_inputs.input_ids.shape[1]
            answer_logits = logits[0, answer_start_idx-1:answer_start_idx-1+len(answer_tokens), :]
            answer_probs = torch.softmax(answer_logits, dim=-1)
            
            # Calculate P(a|q) = ∏ P(a_i | q, a_{<i})
            token_probs = []
            for i, token_id in enumerate(answer_tokens):
                token_prob = answer_probs[i, token_id].item()
                token_probs.append(token_prob)
            
            # P(a|q)
            p_aq = np.prod(token_probs) if token_probs else 1e-10
            p_aq_scores.append(float(p_aq))
            
            # P(a|q)^(1/|a|) - length normalized
            p_aq_norm = np.power(p_aq, 1.0 / len(token_probs)) if token_probs else 1e-10
            p_aq_normalized_scores.append(float(p_aq_norm))
    
    return {
        "p_aq": p_aq_scores,
        "p_aq_normalized": p_aq_normalized_scores,
        "p_true": p_true_scores
    }


def score_test_questions_precomputed(
    test_hidden_states: Dict[int, np.ndarray],
    test_labels: np.ndarray,
    test_metadata: List[Dict],
    probe: LogisticRegression,
    scaler: StandardScaler,
    best_layer: int
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Score test questions using pre-computed hidden states.
    
    Args:
        test_hidden_states: Pre-computed hidden states by layer
        test_labels: Test labels
        test_metadata: Test metadata
        probe: Trained probe
        scaler: Fitted scaler
        best_layer: Best layer index
        
    Returns:
        Tuple of (scored_questions, overall_metrics)
    """
    print(f"\nScoring test data using pre-computed hidden states (layer {best_layer})...")
    
    # Get hidden states for the best layer and scale them
    layer_hidden_states = test_hidden_states[best_layer]
    scaled_hidden_states = scaler.transform(layer_hidden_states)
    
    # Get probe scores (probability of being correct)
    probe_scores = probe.predict_proba(scaled_hidden_states)[:, 1]
    
    # Group by question for K metric calculation
    question_groups = {}
    for idx, meta in enumerate(test_metadata):
        question = meta["question"]
        if question not in question_groups:
            question_groups[question] = {
                "indices": [],
                "probe_scores": [],
                "labels": [],
                "answers": [],
                "golden_answer": meta["golden_answer"]
            }
        
        question_groups[question]["indices"].append(idx)
        question_groups[question]["probe_scores"].append(float(probe_scores[idx]))
        question_groups[question]["labels"].append(bool(test_labels[idx]))
        question_groups[question]["answers"].append(meta["answer"])
    
    # Calculate K metrics per question
    scored_questions = []
    k_values = []
    k_star_values = []
    
    for question, data in question_groups.items():
        answers = data["answers"]
        scores = data["probe_scores"]
        labels = data["labels"]
        
        # Calculate K_q with proper answer deduplication
        k_q = calculate_k_q_with_answers(answers, scores, labels, aggregation_method="max")
        
        if not np.isnan(k_q):
            k_values.append(k_q)
            k_star = 1.0 if k_q == 1.0 else 0.0
            k_star_values.append(k_star)
            
            scored_questions.append({
                "question": question,
                "golden_answer": data["golden_answer"],
                "num_answers": len(answers),
                "num_correct": sum(labels),
                "num_incorrect": sum(1-l for l in labels),
                "probe_scores": scores,
                "answers": answers,
                "labels": labels,
                "k_q": float(k_q),
                "k_star": float(k_star)
            })
    
    # Overall metrics
    overall_metrics = {
        "num_questions": len(scored_questions),
        "mean_k": float(np.mean(k_values)) if k_values else 0.0,
        "mean_k_star": float(np.mean(k_star_values)) if k_star_values else 0.0,
        "std_k": float(np.std(k_values)) if k_values else 0.0
    }
    
    print(f"Processed {len(scored_questions)} test questions")
    print(f"Internal knowledge - Mean K: {overall_metrics['mean_k']:.4f}, Mean K*: {overall_metrics['mean_k_star']:.4f}")
    
    return scored_questions, overall_metrics


def score_test_questions_onthefly(
    test_data: List[Dict],
    model,
    tokenizer,
    probe: LogisticRegression,
    scaler: StandardScaler,
    best_layer: int,
    device: str = "cuda",
    max_answers_per_question: int = 100
) -> Tuple[List[Dict], Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Score test questions by computing hidden states on-the-fly and external scores.
    
    Args:
        test_data: List of test entries
        model: Language model
        tokenizer: Tokenizer
        probe: Trained probe
        scaler: Fitted scaler
        best_layer: Best layer index
        device: Device to run on
        max_answers_per_question: Max answers to process per question
        
    Returns:
        Tuple of (scored_questions, internal_metrics, external_metrics)
    """
    print(f"\nScoring test data on-the-fly (layer {best_layer})...")
    
    scored_questions = []
    k_values_internal = []
    k_star_values_internal = []
    
    # Track external method performance
    external_k_values = {"p_aq": [], "p_aq_normalized": [], "p_true": []}
    external_k_star_values = {"p_aq": [], "p_aq_normalized": [], "p_true": []}
    
    for entry in tqdm(test_data, desc="Processing test questions"):
        question = entry["question"]
        answer_labels = entry["answer_labels"]  # List of [answer, label] pairs
        golden_answer = entry["golden_answer"]
        
        # Limit answers per question
        answers_to_process = answer_labels[:max_answers_per_question]
        
        # Extract answers and labels
        answers = [pair[0] for pair in answers_to_process]
        labels = [pair[1] == "EXACT_MATCH" for pair in answers_to_process]
        
        # Skip questions with no correct/incorrect pairs
        if not any(labels) or all(labels):
            continue
        
        # Calculate external scores
        external_scores = calculate_external_scores(model, tokenizer, question, answers, device)
        
        # Calculate internal scores (probe)
        probe_scores = []
        for answer in answers:
            hidden_states = extract_hidden_states_for_qa_pair(model, tokenizer, question, answer, device)
            layer_hidden_state = hidden_states[best_layer].reshape(1, -1)
            scaled_hidden_state = scaler.transform(layer_hidden_state)
            probe_score = probe.predict_proba(scaled_hidden_state)[0, 1]
            probe_scores.append(float(probe_score))
        
        # Calculate K metrics for internal (probe) method
        k_q_internal = calculate_k_q_with_answers(answers, probe_scores, labels, aggregation_method="max")
        
        # Calculate K metrics for external methods
        k_q_external = {}
        for method_name, scores in external_scores.items():
            k_q_ext = calculate_k_q_with_answers(answers, scores, labels, aggregation_method="max")
            k_q_external[method_name] = k_q_ext
            
            if not np.isnan(k_q_ext):
                external_k_values[method_name].append(k_q_ext)
                k_star_ext = 1.0 if k_q_ext == 1.0 else 0.0
                external_k_star_values[method_name].append(k_star_ext)
        
        # Store results if internal K is valid
        if not np.isnan(k_q_internal):
            k_values_internal.append(k_q_internal)
            k_star_internal = 1.0 if k_q_internal == 1.0 else 0.0
            k_star_values_internal.append(k_star_internal)
            
            scored_questions.append({
                "question": question,
                "golden_answer": golden_answer,
                "num_answers": len(answers),
                "num_correct": sum(labels),
                "num_incorrect": sum(1-l for l in labels),
                "answers": answers,
                "labels": labels,
                "internal_scores": probe_scores,
                "external_scores": external_scores,
                "k_q_internal": float(k_q_internal),
                "k_star_internal": float(k_star_internal),
                "k_q_external": {k: float(v) if not np.isnan(v) else None for k, v in k_q_external.items()}
            })
    
    # Calculate overall metrics
    internal_metrics = {
        "num_questions": len(scored_questions),
        "mean_k": float(np.mean(k_values_internal)) if k_values_internal else 0.0,
        "mean_k_star": float(np.mean(k_star_values_internal)) if k_star_values_internal else 0.0,
        "std_k": float(np.std(k_values_internal)) if k_values_internal else 0.0
    }
    
    external_metrics = {}
    for method_name in external_k_values.keys():
        k_vals = external_k_values[method_name]
        k_star_vals = external_k_star_values[method_name]
        external_metrics[method_name] = {
            "num_questions": len(k_vals),
            "mean_k": float(np.mean(k_vals)) if k_vals else 0.0,
            "mean_k_star": float(np.mean(k_star_vals)) if k_star_vals else 0.0,
            "std_k": float(np.std(k_vals)) if k_vals else 0.0
        }
    
    return scored_questions, internal_metrics, external_metrics


def print_comparison_results(
    internal_metrics: Dict[str, float],
    external_metrics: Dict[str, Dict[str, float]],
    best_layer_metrics: Dict
):
    """
    Print comparison results showing hidden knowledge evidence.
    
    Args:
        internal_metrics: Internal (probe) performance metrics
        external_metrics: External methods performance metrics  
        best_layer_metrics: Dev set performance of the best layer
    """
    print("\n" + "="*80)
    print("HIDDEN KNOWLEDGE EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nTrained Probe Performance (Layer {best_layer_metrics.get('layer', 'Unknown')}):")
    print(f"  Dev Set K:  {best_layer_metrics.get('dev_k_full', 0):.4f}")
    print(f"  Dev Set K*: {best_layer_metrics.get('dev_k_star_full', 0):.4f}")
    
    print(f"\nTest Set Results:")
    print(f"{'Method':<20} {'Num Questions':<15} {'Mean K':<10} {'Mean K*':<10} {'Std K':<10}")
    print("-" * 75)
    
    # Internal (probe) results
    print(f"{'Internal (Probe)':<20} {internal_metrics['num_questions']:<15} "
          f"{internal_metrics['mean_k']:<10.4f} {internal_metrics['mean_k_star']:<10.4f} "
          f"{internal_metrics['std_k']:<10.4f}")
    
    # External method results
    method_names = {
        "p_aq": "P(a|q)",
        "p_aq_normalized": "P(a|q)^(1/|a|)",
        "p_true": "P(True|q,a)"
    }
    
    for method_key, method_display in method_names.items():
        if method_key in external_metrics:
            metrics = external_metrics[method_key]
            print(f"{method_display:<20} {metrics['num_questions']:<15} "
                  f"{metrics['mean_k']:<10.4f} {metrics['mean_k_star']:<10.4f} "
                  f"{metrics['std_k']:<10.4f}")
    
    # Calculate and show hidden knowledge evidence
    print("\n" + "="*80)
    print("HIDDEN KNOWLEDGE ANALYSIS")
    print("="*80)
    
    internal_k = internal_metrics['mean_k']
    internal_k_star = internal_metrics['mean_k_star']
    
    print(f"Internal Knowledge (Probe):     K = {internal_k:.4f}, K* = {internal_k_star:.4f}")
    
    for method_key, method_display in method_names.items():
        if method_key in external_metrics:
            external_k = external_metrics[method_key]['mean_k']
            external_k_star = external_metrics[method_key]['mean_k_star']
            
            gap_k = internal_k - external_k
            gap_k_star = internal_k_star - external_k_star
            
            relative_improvement = ((internal_k / external_k) - 1) * 100 if external_k > 0 else float('inf')
            
            print(f"{method_display:<20}: K = {external_k:.4f}, K* = {external_k_star:.4f}")
            print(f"                     Gap: ΔK = {gap_k:+.4f}, ΔK* = {gap_k_star:+.4f}")
            print(f"                     Relative improvement: {relative_improvement:.1f}%")
            print()
    
    # Statistical significance (basic check)
    best_external_k = max(external_metrics[method]['mean_k'] for method in external_metrics)
    hidden_knowledge_detected = internal_k > best_external_k
    
    print(f"Hidden Knowledge Detected: {'YES' if hidden_knowledge_detected else 'NO'}")
    if hidden_knowledge_detected:
        print(f"Evidence: Internal knowledge (K={internal_k:.4f}) exceeds all external methods")
        print(f"         (best external: K={best_external_k:.4f})")


def main():
    """
    Main function to evaluate test set using trained linear probe.
    """
    # Configuration
    models_dir = Path("./models/probes")
    probe_data_dir = Path("./data/probe_training_data")  
    labeled_data_dir = Path("./data/sampled_labeled_answers_1000_temp1")
    output_dir = Path("./results")
    relations = ["P40", "P50", "P176", "P264"]
    
    print("="*80)
    print("TEST SET EVALUATION WITH LINEAR PROBE")
    print("="*80)
    
    # Load trained probe
    print("\n1. Loading trained probe...")
    probe, scaler, best_layer, best_metrics = load_trained_probe(models_dir)
    
    # Try loading pre-computed test data first
    use_precomputed = True
    try:
        test_hidden_states, test_labels, test_metadata = load_test_data_precomputed(
            probe_data_dir, relations
        )
        print("\nUsing pre-computed test hidden states (faster)...")
        
    except FileNotFoundError as e:
        print(f"\nPre-computed test data not found: {e}")
        print("Falling back to on-the-fly computation (slower but includes external baselines)...")
        use_precomputed = False
        
        # Load original test data
        test_data = load_test_data_original(labeled_data_dir, relations)
        
        # Load model for on-the-fly computation
        print("\n2. Loading model for on-the-fly computation...")
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        device = "cuda"
        
        from huggingface_hub import login
        try:
            from key import KEY
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
    
    # Score test data
    print(f"\n3. Scoring test data...")
    
    if use_precomputed:
        # Use pre-computed hidden states (internal scoring only)
        scored_questions, internal_metrics = score_test_questions_precomputed(
            test_hidden_states, test_labels, test_metadata, probe, scaler, best_layer
        )
        external_metrics = {}
        
    else:
        # Compute on-the-fly (internal + external scoring)  
        scored_questions, internal_metrics, external_metrics = score_test_questions_onthefly(
            test_data, model, tokenizer, probe, scaler, best_layer, device
        )
    
    # Print results
    print("\n4. Results...")
    print_comparison_results(internal_metrics, external_metrics, best_metrics)
    
    # Save detailed results
    print(f"\n5. Saving results...")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results = {
        "metadata": {
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "best_layer": best_layer,
            "best_layer_dev_metrics": best_metrics,
            "relations": relations,
            "use_precomputed": use_precomputed
        },
        "internal_metrics": internal_metrics,
        "external_metrics": external_metrics,
        "scored_questions": scored_questions[:10]  # Save first 10 for inspection
    }
    
    results_path = output_dir / "test_evaluation_results.json"
    save_json(results, results_path)
    print(f"Results saved to: {results_path}")
    
    # Save full scored questions separately (can be large)
    questions_path = output_dir / "test_scored_questions.json"  
    save_json(scored_questions, questions_path)
    print(f"Detailed question scores saved to: {questions_path}")
    
    print("\n" + "="*80)
    print("TEST EVALUATION COMPLETE!")
    print("="*80)
    
    if use_precomputed:
        print("\nNote: Only internal scoring was performed (using pre-computed hidden states).")
        print("To get external baseline comparisons, run generate_data_for_probe.py with 'test' split")
        print("or delete the pre-computed files to force on-the-fly computation.")


if __name__ == "__main__":
    main()
