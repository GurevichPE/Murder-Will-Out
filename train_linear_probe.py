"""
Train logistic regression probes for internal knowledge scoring using pre-computed hidden states.

This script implements a comparative experiment evaluating probes on:
1. Full dev dataset (all questions)
2. Greedy-correct dev subset (only questions where greedy decoding is correct)

The script:
1. Loads pre-computed train/dev hidden states from probe_training_data/
2. For each layer, trains a logistic regression classifier
3. Evaluates on both full dev set and greedy-correct subset
4. Compares performance to understand how probe quality varies with model "knowledge"
5. Selects the best layer and saves the trained probe

Features:
- All hidden states are pre-computed - no model loading needed!
- Dual evaluation strategy: full vs greedy-correct dev sets
- Answer deduplication for accurate K/K* calculations
- Comprehensive comparison analysis showing performance differences
- Visualizations comparing full vs greedy-correct evaluation results
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from knowledge_calculation import calculate_k_q, calculate_k_q_with_answers


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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_plotting_data(all_metrics: List[Dict], output_path: Path) -> None:
    """
    Save all plotting data to enable plot generation without retraining.
    
    Args:
        all_metrics: List of metrics dicts for each layer
        output_path: Path to save the plotting data
    """
    plotting_data = {
        "layers": [m["layer"] for m in all_metrics],
        "train_auc": [m["train_auc"] for m in all_metrics],
        "train_accuracy": [m["train_accuracy"] for m in all_metrics],
        "dev_auc_full": [m["dev_auc_full"] for m in all_metrics],
        "dev_accuracy_full": [m["dev_accuracy_full"] for m in all_metrics],
        "dev_k_full": [m["dev_k_full"] for m in all_metrics],
        "dev_k_star_full": [m["dev_k_star_full"] for m in all_metrics],
        "dev_auc_greedy": [m["dev_auc_greedy"] for m in all_metrics],
        "dev_accuracy_greedy": [m["dev_accuracy_greedy"] for m in all_metrics],
        "dev_k_greedy": [m["dev_k_greedy"] for m in all_metrics],
        "dev_k_star_greedy": [m["dev_k_star_greedy"] for m in all_metrics]
    }
    
    save_json(plotting_data, output_path)
    print(f"Plotting data saved to: {output_path}")


def generate_plots_from_saved_data(plotting_data_path: Path, output_dir: Path) -> None:
    """
    Generate all plots using previously saved plotting data.
    
    This function allows you to regenerate plots without retraining probes.
    
    Args:
        plotting_data_path: Path to the saved plotting data JSON file
        output_dir: Directory to save the generated plots
        
    Example:
        plotting_data_path = Path("./models/probes/plotting_data.json")
        output_dir = Path("./models/probes/plots")
        generate_plots_from_saved_data(plotting_data_path, output_dir)
    """
    if not plotting_data_path.exists():
        raise FileNotFoundError(f"Plotting data file not found: {plotting_data_path}")
    
    # Load plotting data
    plotting_data = load_json(plotting_data_path)
    print(f"Loaded plotting data from: {plotting_data_path}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Generating plots in: {output_dir}")
    
    # Generate all plots
    print("Generating full dev set plots...")
    plot_full_dev_metrics(plotting_data, output_dir)
    
    print("Generating comparison plots...")
    plot_full_vs_greedy_comparison(plotting_data, output_dir)
    
    print("All plots generated successfully!")


def load_probe_training_data(
    data_dir: Path,
    relations: List[str],
    split: str = "train"
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[Dict]]:
    """
    Load pre-computed probe training data from all relations and organize by layer.
    
    Args:
        data_dir: Directory containing probe data files
        relations: List of relation codes (e.g., ["P40", "P50"])
        split: Data split ("train" or "dev")
        
    Returns:
        Tuple of:
        - hidden_states_by_layer: Dict mapping layer_idx -> array of hidden states
        - labels: Array of labels (0 or 1)
        - metadata: List of dicts with question, answer, etc.
    """
    all_examples = []
    
    # Load data from all relations
    # Train uses JSON, dev/test use pickle
    for relation in relations:
        if split == "train":
            file_path = data_dir / f"{relation}.{split}.probe_data.json"
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping...")
                continue
            print(f"Loading {relation} {split} data from JSON...")
            data = load_json(file_path)
        else:
            file_path = data_dir / f"{relation}.{split}.probe_data.pkl"
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping...")
                continue
            print(f"Loading {relation} {split} data from pickle...")
            data = load_pickle(file_path)
        
        all_examples.extend(data)
    
    print(f"Loaded {len(all_examples)} total examples from {len(relations)} relations")
    
    # Organize by layer
    num_layers = len(all_examples[0]["hidden_states"])
    print(f"Number of layers: {num_layers}")
    
    # Initialize storage
    hidden_states_by_layer = {i: [] for i in range(num_layers)}
    labels = []
    metadata = []
    
    # Extract hidden states and labels
    for example in all_examples:
        # Extract hidden states for each layer
        # Note: JSON has string keys, pickle has int keys
        hs_dict = example["hidden_states"]
        for layer_idx in range(num_layers):
            # Try both int and string keys for compatibility
            if layer_idx in hs_dict:
                hidden_state = hs_dict[layer_idx]
            else:
                hidden_state = hs_dict[str(layer_idx)]
            hidden_states_by_layer[layer_idx].append(hidden_state)
        
        # Extract label
        labels.append(example["label"])
        
        # Store metadata
        metadata.append({
            "question": example["question"],
            "answer": example["answer"],
            "golden_answer": example["golden_answer"]
        })
    
    # Convert to numpy arrays
    for layer_idx in range(num_layers):
        hidden_states_by_layer[layer_idx] = np.array(hidden_states_by_layer[layer_idx])
    
    labels = np.array(labels)
    
    print(f"Hidden state shape per layer: {hidden_states_by_layer[0].shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {np.sum(labels == 1)} positive, {np.sum(labels == 0)} negative")
    
    return hidden_states_by_layer, labels, metadata


def load_and_filter_greedy_correct_dev(
    dev_hidden_states: Dict[int, np.ndarray],
    dev_labels: np.ndarray,
    dev_metadata: List[Dict],
    relations: List[str]
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[Dict]]:
    """
    Load greedy answers and filter dev set to only include examples where 
    greedy answer exactly matches the golden answer.
    
    Args:
        dev_hidden_states: Dict mapping layer_idx -> dev hidden states
        dev_labels: Dev labels
        dev_metadata: Dev metadata
        relations: List of relation codes
        
    Returns:
        Tuple of (filtered_hidden_states_dict, filtered_labels, filtered_metadata)
    """
    # Load all greedy answer data
    greedy_data_by_question = {}
    
    for relation in relations:
        greedy_file = Path(f"./data/greedy_answers/dev/{relation}.dev.json")
        if not greedy_file.exists():
            print(f"Warning: Greedy answers file not found: {greedy_file}")
            continue
            
        print(f"Loading greedy answers from {greedy_file}")
        with open(greedy_file, 'r', encoding='utf-8') as f:
            greedy_data = json.load(f)
        
        for entry in greedy_data:
            question = entry["question"]
            greedy_answer = entry["greedy_answer"]
            golden_answers = entry["answers"]  # These are the golden/correct answers
            
            # Store the greedy data for this question
            greedy_data_by_question[question] = {
                "greedy_answer": greedy_answer,
                "golden_answers": golden_answers
            }
    
    print(f"Loaded greedy data for {len(greedy_data_by_question)} questions")
    
    # Filter dev examples to only keep those where greedy answer exactly matches golden answer
    filtered_indices = []
    greedy_correct_count = 0
    
    for idx, meta in enumerate(dev_metadata):
        question = meta["question"]
        
        if question in greedy_data_by_question:
            greedy_info = greedy_data_by_question[question]
            greedy_answer = greedy_info["greedy_answer"]
            golden_answers = greedy_info["golden_answers"]
            
            # Check if greedy answer exactly matches any golden answer
            is_exact_match = any(
                greedy_answer.strip() == golden.strip()
                for golden in golden_answers
            )
            
            if is_exact_match:
                filtered_indices.append(idx)
                if idx == len([i for i in range(idx) if dev_metadata[i]["question"] == question]):
                    # Only count once per question (first occurrence)
                    greedy_correct_count += 1
    
    print(f"Found {greedy_correct_count} questions with exact greedy matches")
    print(f"Filtering {len(filtered_indices)}/{len(dev_metadata)} examples ({len(filtered_indices)/len(dev_metadata):.1%})")
    
    if len(filtered_indices) == 0:
        print("Warning: No examples found where greedy answer exactly matches golden answer!")
        return {}, np.array([]), []
    
    # Filter all data
    filtered_indices = np.array(filtered_indices)
    filtered_labels = dev_labels[filtered_indices]
    filtered_metadata = [dev_metadata[i] for i in filtered_indices]
    
    # Filter hidden states for all layers
    filtered_hidden_states = {}
    for layer_idx, layer_states in dev_hidden_states.items():
        filtered_hidden_states[layer_idx] = layer_states[filtered_indices]
    
    return filtered_hidden_states, filtered_labels, filtered_metadata


def calculate_k_metrics_for_dev(
    dev_hidden_states: np.ndarray,
    dev_labels: np.ndarray,
    dev_metadata: List[Dict],
    probe,
    scaler
) -> Tuple[float, float]:
    """
    Calculate K and K* metrics on dev set using pre-computed hidden states.
    
    Groups examples by question and calculates knowledge metrics using answer deduplication.
    
    Args:
        dev_hidden_states: Hidden states for all dev examples (n_examples, hidden_dim)
        dev_labels: Labels (0 or 1) for all dev examples
        dev_metadata: Metadata with question, answer, golden_answer for each example
        probe: Trained MLP classifier
        scaler: Fitted StandardScaler
        
    Returns:
        Tuple of (mean_k, mean_k_star)
    """
    # Scale hidden states
    dev_hidden_states_scaled = scaler.transform(dev_hidden_states)
    
    # Get probe scores
    probe_scores = probe.predict_proba(dev_hidden_states_scaled)[:, 1]
    
    # Group by question
    question_groups = {}
    for idx, meta in enumerate(dev_metadata):
        question = meta["question"]
        if question not in question_groups:
            question_groups[question] = {
                "indices": [],
                "scores": [],
                "is_correct": [],
                "answers": []
            }
        
        # Use the actual label from the data (not golden answer comparison)
        is_correct = (dev_labels[idx] == 1)
        
        question_groups[question]["indices"].append(idx)
        question_groups[question]["scores"].append(probe_scores[idx])
        question_groups[question]["is_correct"].append(is_correct)
        question_groups[question]["answers"].append(meta["answer"])
    
    # Calculate K_q for each question
    k_values = []
    k_star_values = []
    skipped_questions = 0
    
    for question, data in question_groups.items():
        indices = np.array(data["indices"])
        scores = np.array(data["scores"])
        is_correct = np.array(data["is_correct"])
        answers = data["answers"]
        
        # Check if we have both correct and incorrect answers
        correct_indices = np.where(is_correct)[0]
        incorrect_indices = np.where(~is_correct)[0]
        
        if len(correct_indices) == 0 or len(incorrect_indices) == 0:
            skipped_questions += 1
            if len(correct_indices) == 0:
                reason = "no correct answers"
            else:
                reason = "no incorrect answers"
            print(f"Warning: Question '{question[:50]}...' has {reason} - skipping per paper Section 4.2")
            continue
        
        # Calculate K_q with answer deduplication
        # This properly handles cases where the same answer appears multiple times
        k_q = calculate_k_q_with_answers(answers, scores, is_correct, aggregation_method="max")
        
        # Only include valid K_q values (not NaN)
        if not np.isnan(k_q):
            k_values.append(k_q)
            
            # K* = 1 if K = 1, else 0
            k_star = 1.0 if k_q == 1.0 else 0.0
            k_star_values.append(k_star)
    
    print(f"Processed {len(question_groups)} questions, skipped {skipped_questions} questions")
    print(f"Valid K_q calculations: {len(k_values)}")
    
    # Compute means only over valid questions
    mean_k = np.mean(k_values) if k_values else 0.0
    mean_k_star = np.mean(k_star_values) if k_star_values else 0.0
    
    return mean_k, mean_k_star


def train_probe_for_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev_full: np.ndarray,
    y_dev_full: np.ndarray,
    dev_metadata_full: List[Dict],
    X_dev_greedy: np.ndarray,
    y_dev_greedy: np.ndarray,
    dev_metadata_greedy: List[Dict],
    layer_idx: int,
    random_state: int = 42
) -> Tuple[LogisticRegression, StandardScaler, Dict[str, float]]:
    """
    Train a logistic regression probe for a specific layer using pre-computed hidden states.
    Evaluates on both full dev set and greedy-correct dev subset.
    
    Args:
        X_train: Training hidden states
        y_train: Training labels
        X_dev_full: Full dev hidden states
        y_dev_full: Full dev labels
        dev_metadata_full: Full dev metadata
        X_dev_greedy: Greedy-correct dev hidden states
        y_dev_greedy: Greedy-correct dev labels
        dev_metadata_greedy: Greedy-correct dev metadata
        layer_idx: Layer index (for logging)
        random_state: Random seed
        
    Returns:
        Tuple of (trained_probe, scaler, metrics_dict)
    """

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_full_scaled = scaler.transform(X_dev_full)
    
    print(f"  Layer {layer_idx}: Training logistic regression on {X_train_scaled.shape[1]} features")
    
    # Train logistic regression
    probe = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs',
        class_weight='balanced'
    )
    
    probe.fit(X_train_scaled, y_train)
    
    # Evaluate on train set (binary classification metrics)
    y_train_pred_proba = probe.predict_proba(X_train_scaled)[:, 1]
    y_train_pred = probe.predict(X_train_scaled)
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Evaluate on full dev set
    print(f"  Layer {layer_idx}: Evaluating on full dev set...")
    y_dev_full_pred_proba = probe.predict_proba(X_dev_full_scaled)[:, 1]
    y_dev_full_pred = probe.predict(X_dev_full_scaled)
    dev_auc_full = roc_auc_score(y_dev_full, y_dev_full_pred_proba)
    dev_accuracy_full = accuracy_score(y_dev_full, y_dev_full_pred)
    
    dev_k_full, dev_k_star_full = calculate_k_metrics_for_dev(
        X_dev_full, y_dev_full, dev_metadata_full, probe, scaler
    )
    
    # Evaluate on greedy-correct subset of dev set
    print(f"  Layer {layer_idx}: Evaluating on greedy-correct dev subset...")
    if len(y_dev_greedy) > 0:
        X_dev_greedy_scaled = scaler.transform(X_dev_greedy)
        y_dev_greedy_pred_proba = probe.predict_proba(X_dev_greedy_scaled)[:, 1]
        y_dev_greedy_pred = probe.predict(X_dev_greedy_scaled)
        dev_auc_greedy = roc_auc_score(y_dev_greedy, y_dev_greedy_pred_proba)
        dev_accuracy_greedy = accuracy_score(y_dev_greedy, y_dev_greedy_pred)
        
        dev_k_greedy, dev_k_star_greedy = calculate_k_metrics_for_dev(
            X_dev_greedy, y_dev_greedy, dev_metadata_greedy, probe, scaler
        )
    else:
        dev_auc_greedy = dev_accuracy_greedy = dev_k_greedy = dev_k_star_greedy = 0.0
    
    print(f"  Layer {layer_idx} Results:")
    print(f"    Train AUC: {train_auc:.3f}, Train Acc: {train_accuracy:.3f}")
    print(f"    Dev (Full) AUC: {dev_auc_full:.3f}, Acc: {dev_accuracy_full:.3f}, K: {dev_k_full:.3f}, K*: {dev_k_star_full:.3f}")
    print(f"    Dev (Greedy) AUC: {dev_auc_greedy:.3f}, Acc: {dev_accuracy_greedy:.3f}, K: {dev_k_greedy:.3f}, K*: {dev_k_star_greedy:.3f}")
    
    metrics = {
        "layer": layer_idx,
        "train_auc": float(train_auc),
        "train_accuracy": float(train_accuracy),
        "dev_auc_full": float(dev_auc_full),
        "dev_accuracy_full": float(dev_accuracy_full),
        "dev_k_full": float(dev_k_full),
        "dev_k_star_full": float(dev_k_star_full),
        "dev_auc_greedy": float(dev_auc_greedy),
        "dev_accuracy_greedy": float(dev_accuracy_greedy),
        "dev_k_greedy": float(dev_k_greedy),
        "dev_k_star_greedy": float(dev_k_star_greedy)
    }
    
    return probe, scaler, metrics


def train_all_layers(
    train_hidden_states: Dict[int, np.ndarray],
    train_labels: np.ndarray,
    dev_hidden_states_full: Dict[int, np.ndarray],
    dev_labels_full: np.ndarray,
    dev_metadata_full: List[Dict],
    dev_hidden_states_greedy: Dict[int, np.ndarray],
    dev_labels_greedy: np.ndarray,
    dev_metadata_greedy: List[Dict]
) -> Tuple[Dict[int, Tuple], List[Dict]]:
    """
    Train probes for all layers using pre-computed hidden states.
    Evaluates on both full dev set and greedy-correct dev subset.
    
    Args:
        train_hidden_states: Dict mapping layer_idx -> train hidden states
        train_labels: Train labels
        dev_hidden_states_full: Dict mapping layer_idx -> full dev hidden states
        dev_labels_full: Full dev labels
        dev_metadata_full: Full dev metadata
        dev_hidden_states_greedy: Dict mapping layer_idx -> greedy-correct dev hidden states
        dev_labels_greedy: Greedy-correct dev labels
        dev_metadata_greedy: Greedy-correct dev metadata
        
    Returns:
        Tuple of (probes_dict, all_metrics)
        - probes_dict: {layer_idx: (probe, scaler)}
        - all_metrics: List of metrics dicts
    """
    num_layers = len(train_hidden_states)
    probes_dict = {}
    all_metrics = []
    
    print(f"\nTraining probes for {num_layers} layers using pre-computed hidden states...")
    
    for layer_idx in tqdm(range(num_layers), desc="Training layers"):
        probe, scaler, metrics = train_probe_for_layer(
            train_hidden_states[layer_idx],
            train_labels,
            dev_hidden_states_full[layer_idx],
            dev_labels_full,
            dev_metadata_full,
            dev_hidden_states_greedy[layer_idx],
            dev_labels_greedy,
            dev_metadata_greedy,
            layer_idx
        )
        
        probes_dict[layer_idx] = (probe, scaler)
        all_metrics.append(metrics)
    
    return probes_dict, all_metrics


def select_best_layer(
    all_metrics: List[Dict],
    criterion: str = "dev_k_full"
) -> int:
    """
    Select the best layer based on dev set performance.
    
    Args:
        all_metrics: List of metrics dicts for each layer
        criterion: Metric to use for selection (default: "dev_k")
        
    Returns:
        Index of the best layer
    """
    best_layer = max(all_metrics, key=lambda x: x[criterion])
    return best_layer["layer"]


def plot_layer_performance(
    all_metrics: List[Dict],
    output_path: Path
):
    """
    Plot performance metrics comparing full dev vs greedy-correct dev evaluation.
    
    Args:
        all_metrics: List of metrics dicts for each layer
        output_path: Path to save the plot
    """
    layers = [m["layer"] for m in all_metrics]
    train_auc = [m["train_auc"] for m in all_metrics]
    train_accuracy = [m["train_accuracy"] for m in all_metrics]
    
    dev_auc_full = [m["dev_auc_full"] for m in all_metrics]
    dev_accuracy_full = [m["dev_accuracy_full"] for m in all_metrics]
    dev_k_full = [m["dev_k_full"] for m in all_metrics]
    dev_k_star_full = [m["dev_k_star_full"] for m in all_metrics]
    
    dev_auc_greedy = [m["dev_auc_greedy"] for m in all_metrics]
    dev_accuracy_greedy = [m["dev_accuracy_greedy"] for m in all_metrics]
    dev_k_greedy = [m["dev_k_greedy"] for m in all_metrics]
    dev_k_star_greedy = [m["dev_k_star_greedy"] for m in all_metrics]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot AUC comparison: Full vs Greedy-Correct
    ax1.plot(layers, train_auc, 'o-', label='Train AUC', linewidth=2, markersize=4, color='blue')
    ax1.plot(layers, dev_auc_full, 's-', label='Dev AUC (Full)', linewidth=2, markersize=4, color='orange')
    ax1.plot(layers, dev_auc_greedy, '^-', label='Dev AUC (Greedy)', linewidth=2, markersize=4, color='red')
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax1.set_title('AUC: Full vs Greedy-Correct Dev', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot K metrics comparison: Full vs Greedy-Correct
    ax2.plot(layers, dev_k_full, 'o-', label='Dev K (Full)', linewidth=2, markersize=4, color='green')
    ax2.plot(layers, dev_k_greedy, 's-', label='Dev K (Greedy)', linewidth=2, markersize=4, color='darkgreen')
    ax2.plot(layers, dev_k_star_full, '^-', label='Dev K* (Full)', linewidth=2, markersize=4, color='red')
    ax2.plot(layers, dev_k_star_greedy, 'v-', label='Dev K* (Greedy)', linewidth=2, markersize=4, color='darkred')
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Knowledge Metric', fontsize=12, fontweight='bold')
    ax2.set_title('K Metrics: Full vs Greedy-Correct Dev', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot Accuracy comparison: Full vs Greedy-Correct  
    ax3.plot(layers, train_accuracy, 'o-', label='Train Accuracy', linewidth=2, markersize=4, color='blue')
    ax3.plot(layers, dev_accuracy_full, 's-', label='Dev Acc (Full)', linewidth=2, markersize=4, color='orange')
    ax3.plot(layers, dev_accuracy_greedy, '^-', label='Dev Acc (Greedy)', linewidth=2, markersize=4, color='red')
    ax3.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy: Full vs Greedy-Correct Dev', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot Performance improvement on Greedy subset
    k_improvement = [g - f for f, g in zip(dev_k_full, dev_k_greedy)]
    auc_improvement = [g - f for f, g in zip(dev_auc_full, dev_auc_greedy)]
    acc_improvement = [g - f for f, g in zip(dev_accuracy_full, dev_accuracy_greedy)]
    
    ax4.plot(layers, k_improvement, 'o-', label='K Improvement', linewidth=2, markersize=4, color='green')
    ax4.plot(layers, auc_improvement, 's-', label='AUC Improvement', linewidth=2, markersize=4, color='blue')
    ax4.plot(layers, acc_improvement, '^-', label='Acc Improvement', linewidth=2, markersize=4, color='orange')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Performance Improvement', fontsize=12, fontweight='bold')
    ax4.set_title('Greedy-Correct vs Full Dev Improvement', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    # Highlight best layer based on full dev K
    best_idx = np.argmax(dev_k_full)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(x=layers[best_idx], color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_full_dev_metrics(plotting_data: Dict, output_dir: Path) -> None:
    """
    Create separate plots for accuracy, AUC, and K/K* metrics using only full dev set evaluation.
    
    Args:
        plotting_data: Dictionary containing all plotting data
        output_dir: Directory to save the plots
    """
    layers = plotting_data["layers"]
    train_auc = plotting_data["train_auc"]
    train_accuracy = plotting_data["train_accuracy"]
    dev_auc_full = plotting_data["dev_auc_full"]
    dev_accuracy_full = plotting_data["dev_accuracy_full"]
    dev_k_full = plotting_data["dev_k_full"]
    dev_k_star_full = plotting_data["dev_k_star_full"]
    
    # Find best layer for highlighting
    best_idx = np.argmax(dev_k_full)
    
    # Plot 1: Accuracy (Full Dev Only)
    plt.figure(figsize=(10, 6))
    plt.plot(layers, train_accuracy, 'o-', label='Train Accuracy', linewidth=2, markersize=6, color='blue')
    plt.plot(layers, dev_accuracy_full, 's-', label='Dev Accuracy', linewidth=2, markersize=6, color='orange')
    plt.axvline(x=layers[best_idx], color='red', linestyle='--', alpha=0.5, linewidth=2, label=f'Best Layer ({layers[best_idx]})')
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Accuracy by Layer (Full Dev Set)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    accuracy_path = output_dir / "accuracy_full_dev.png"
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy plot (full dev) saved to: {accuracy_path}")
    plt.close()
    
    # Plot 2: AUC (Full Dev Only)
    plt.figure(figsize=(10, 6))
    plt.plot(layers, train_auc, 'o-', label='Train AUC', linewidth=2, markersize=6, color='blue')
    plt.plot(layers, dev_auc_full, 's-', label='Dev AUC', linewidth=2, markersize=6, color='orange')
    plt.axvline(x=layers[best_idx], color='red', linestyle='--', alpha=0.5, linewidth=2, label=f'Best Layer ({layers[best_idx]})')
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel('AUC Score', fontsize=12, fontweight='bold')
    plt.title('AUC by Layer (Full Dev Set)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    auc_path = output_dir / "auc_full_dev.png"
    plt.savefig(auc_path, dpi=300, bbox_inches='tight')
    print(f"AUC plot (full dev) saved to: {auc_path}")
    plt.close()
    
    # Plot 3: K and K* Metrics (Full Dev Only)
    plt.figure(figsize=(10, 6))
    plt.plot(layers, dev_k_full, 'o-', label='K (Knowledge)', linewidth=2, markersize=6, color='green')
    plt.plot(layers, dev_k_star_full, 's-', label='K* (Perfect Knowledge)', linewidth=2, markersize=6, color='darkgreen')
    plt.axvline(x=layers[best_idx], color='red', linestyle='--', alpha=0.5, linewidth=2, label=f'Best Layer ({layers[best_idx]})')
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel('Knowledge Metric', fontsize=12, fontweight='bold')
    plt.title('Knowledge Metrics by Layer (Full Dev Set)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    k_metrics_path = output_dir / "k_metrics_full_dev.png"
    plt.savefig(k_metrics_path, dpi=300, bbox_inches='tight')
    print(f"K metrics plot (full dev) saved to: {k_metrics_path}")
    plt.close()


def plot_full_vs_greedy_comparison(plotting_data: Dict, output_dir: Path) -> None:
    """
    Create separate plots comparing full dev vs greedy-correct dev evaluation for accuracy, AUC, and K/K*.
    
    Args:
        plotting_data: Dictionary containing all plotting data
        output_dir: Directory to save the plots
    """
    layers = plotting_data["layers"]
    dev_auc_full = plotting_data["dev_auc_full"]
    dev_accuracy_full = plotting_data["dev_accuracy_full"]
    dev_k_full = plotting_data["dev_k_full"]
    dev_k_star_full = plotting_data["dev_k_star_full"]
    dev_auc_greedy = plotting_data["dev_auc_greedy"]
    dev_accuracy_greedy = plotting_data["dev_accuracy_greedy"]
    dev_k_greedy = plotting_data["dev_k_greedy"]
    dev_k_star_greedy = plotting_data["dev_k_star_greedy"]
    
    # Find best layer for highlighting
    best_idx = np.argmax(dev_k_full)
    
    # Plot 1: Accuracy Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(layers, dev_accuracy_full, 'o-', label='Full Dev Set', linewidth=2, markersize=6, color='blue')
    plt.plot(layers, dev_accuracy_greedy, 's-', label='Greedy-Correct Subset', linewidth=2, markersize=6, color='red')
    plt.axvline(x=layers[best_idx], color='gray', linestyle='--', alpha=0.5, linewidth=2, label=f'Best Layer ({layers[best_idx]})')
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Accuracy Comparison: Full Dev vs Greedy-Correct Subset', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    accuracy_comp_path = output_dir / "accuracy_comparison.png"
    plt.savefig(accuracy_comp_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy comparison plot saved to: {accuracy_comp_path}")
    plt.close()
    
    # Plot 2: AUC Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(layers, dev_auc_full, 'o-', label='Full Dev Set', linewidth=2, markersize=6, color='blue')
    plt.plot(layers, dev_auc_greedy, 's-', label='Greedy-Correct Subset', linewidth=2, markersize=6, color='red')
    plt.axvline(x=layers[best_idx], color='gray', linestyle='--', alpha=0.5, linewidth=2, label=f'Best Layer ({layers[best_idx]})')
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel('AUC Score', fontsize=12, fontweight='bold')
    plt.title('AUC Comparison: Full Dev vs Greedy-Correct Subset', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    auc_comp_path = output_dir / "auc_comparison.png"
    plt.savefig(auc_comp_path, dpi=300, bbox_inches='tight')
    print(f"AUC comparison plot saved to: {auc_comp_path}")
    plt.close()
    
    # Plot 3: K Metrics Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # K metric comparison
    ax1.plot(layers, dev_k_full, 'o-', label='Full Dev Set', linewidth=2, markersize=6, color='green')
    ax1.plot(layers, dev_k_greedy, 's-', label='Greedy-Correct Subset', linewidth=2, markersize=6, color='darkgreen')
    ax1.axvline(x=layers[best_idx], color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('K (Knowledge Metric)', fontsize=12, fontweight='bold')
    ax1.set_title('K Metric Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # K* metric comparison
    ax2.plot(layers, dev_k_star_full, 'o-', label='Full Dev Set', linewidth=2, markersize=6, color='red')
    ax2.plot(layers, dev_k_star_greedy, 's-', label='Greedy-Correct Subset', linewidth=2, markersize=6, color='darkred')
    ax2.axvline(x=layers[best_idx], color='gray', linestyle='--', alpha=0.5, linewidth=2, label=f'Best Layer ({layers[best_idx]})')
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('K* (Perfect Knowledge)', fontsize=12, fontweight='bold')
    ax2.set_title('K* Metric Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    k_comp_path = output_dir / "k_metrics_comparison.png"
    plt.savefig(k_comp_path, dpi=300, bbox_inches='tight')
    print(f"K metrics comparison plot saved to: {k_comp_path}")
    plt.close()
    
    # Plot 4: Performance Improvement (Greedy vs Full)
    k_improvement = [g - f for f, g in zip(dev_k_full, dev_k_greedy)]
    auc_improvement = [g - f for f, g in zip(dev_auc_full, dev_auc_greedy)]
    acc_improvement = [g - f for f, g in zip(dev_accuracy_full, dev_accuracy_greedy)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, k_improvement, 'o-', label='K Improvement', linewidth=2, markersize=6, color='green')
    plt.plot(layers, auc_improvement, 's-', label='AUC Improvement', linewidth=2, markersize=6, color='blue')
    plt.plot(layers, acc_improvement, '^-', label='Accuracy Improvement', linewidth=2, markersize=6, color='orange')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=layers[best_idx], color='gray', linestyle='--', alpha=0.5, linewidth=2, label=f'Best Layer ({layers[best_idx]})')
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel('Performance Improvement', fontsize=12, fontweight='bold')
    plt.title('Performance Improvement: Greedy-Correct vs Full Dev', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    improvement_path = output_dir / "performance_improvement.png"
    plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
    print(f"Performance improvement plot saved to: {improvement_path}")
    plt.close()


def save_probe(
    probe: LogisticRegression,
    scaler: StandardScaler,
    layer_idx: int,
    output_dir: Path
):
    """
    Save trained probe and scaler.
    
    Args:
        probe: Trained logistic regression classifier
        scaler: Fitted StandardScaler
        layer_idx: Layer index
        output_dir: Directory to save the models
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save probe
    probe_path = output_dir / f"probe_layer_{layer_idx}.pkl"
    with open(probe_path, 'wb') as f:
        pickle.dump(probe, f)
    
    # Save scaler
    scaler_path = output_dir / f"scaler_layer_{layer_idx}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Saved probe to: {probe_path}")
    print(f"Saved scaler to: {scaler_path}")


def main():
    """
    Main function to train linear probes using pre-computed hidden states.
    """
    # Configuration
    probe_data_dir = Path("./data/probe_training_data")
    output_dir = Path("./models/probes")
    relations = ["P40", "P50", "P176", "P264"]  # Updated to match generate_data_for_probe.py
    
    print("="*70)
    print("Linear Probe Training with Pre-Computed Hidden States")
    print("="*70)
    
    # Load training data
    print("\n" + "="*70)
    print("Loading Pre-Computed Training Data")
    print("="*70)
    train_hidden_states, train_labels, train_metadata = load_probe_training_data(
        probe_data_dir, relations, split="train"
    )
    
    # Load dev data
    print("\n" + "="*70)
    print("Loading Pre-Computed Dev Data")
    print("="*70)
    dev_hidden_states, dev_labels, dev_metadata = load_probe_training_data(
        probe_data_dir, relations, split="dev"
    )
    
    # Create greedy-correct dev subset
    print("\n" + "="*70)
    print("Creating Greedy-Correct Dev Subset")
    print("="*70)
    dev_hidden_states_greedy, dev_labels_greedy, dev_metadata_greedy = load_and_filter_greedy_correct_dev(
        dev_hidden_states, dev_labels, dev_metadata, relations
    )
    
    # Train probes for all layers
    print("\n" + "="*70)
    print("Training Probes")
    print("="*70)
    probes_dict, all_metrics = train_all_layers(
        train_hidden_states,
        train_labels,
        dev_hidden_states,
        dev_labels,
        dev_metadata,
        dev_hidden_states_greedy,
        dev_labels_greedy,
        dev_metadata_greedy
    )
    
    # Print results
    print("\n" + "="*140)
    print("Results Summary - Logistic Regression Probes (Full vs Greedy-Correct Dev Evaluation)")
    print("="*140)
    print(f"{'Layer':<6} {'Train':<8} {'Train':<8} {'Dev (Full)':<12} {'Dev (Full)':<12} {'Dev (Full)':<12} {'Dev (Greedy)':<14} {'Dev (Greedy)':<14} {'Dev (Greedy)':<14}")
    print(f"{'':^6} {'AUC':<8} {'Acc':<8} {'AUC':<12} {'Acc':<12} {'K':<12} {'AUC':<14} {'Acc':<14} {'K':<14}")
    print("-"*140)
    for metrics in all_metrics:
        print(f"{metrics['layer']:<6} "
              f"{metrics['train_auc']:<8.3f} "
              f"{metrics['train_accuracy']:<8.3f} "
              f"{metrics['dev_auc_full']:<12.3f} "
              f"{metrics['dev_accuracy_full']:<12.3f} "
              f"{metrics['dev_k_full']:<12.3f} "
              f"{metrics['dev_auc_greedy']:<14.3f} "
              f"{metrics['dev_accuracy_greedy']:<14.3f} "
              f"{metrics['dev_k_greedy']:<14.3f}")
    
    # Select best layer
    best_layer = select_best_layer(all_metrics, criterion="dev_k_full")
    best_metrics = all_metrics[best_layer]
    
    print("\n" + "="*80)
    print(f"Best Layer: {best_layer} (selected based on full dev K)")
    print("="*80)
    print(f"Train AUC:               {best_metrics['train_auc']:.4f}")
    print(f"Train Accuracy:          {best_metrics['train_accuracy']:.4f}")
    print("\nDev (Full Dataset):")
    print(f"  AUC:                   {best_metrics['dev_auc_full']:.4f}")
    print(f"  Accuracy:              {best_metrics['dev_accuracy_full']:.4f}")
    print(f"  K:                     {best_metrics['dev_k_full']:.4f}")
    print(f"  K*:                    {best_metrics['dev_k_star_full']:.4f}")
    print("\nDev (Greedy-Correct Subset):")
    print(f"  AUC:                   {best_metrics['dev_auc_greedy']:.4f}")
    print(f"  Accuracy:              {best_metrics['dev_accuracy_greedy']:.4f}")
    print(f"  K:                     {best_metrics['dev_k_greedy']:.4f}")
    print(f"  K*:                    {best_metrics['dev_k_star_greedy']:.4f}")
    
    # Compare full vs greedy-correct dev performance
    print("\n" + "="*80)
    print("Experimental Analysis: Full Dev vs Greedy-Correct Dev Subset")
    print("="*80)
    
    full_k_values = [m['dev_k_full'] for m in all_metrics]
    greedy_k_values = [m['dev_k_greedy'] for m in all_metrics]
    full_auc_values = [m['dev_auc_full'] for m in all_metrics]
    greedy_auc_values = [m['dev_auc_greedy'] for m in all_metrics]
    
    avg_k_full = np.mean(full_k_values)
    avg_k_greedy = np.mean(greedy_k_values)
    avg_auc_full = np.mean(full_auc_values)
    avg_auc_greedy = np.mean(greedy_auc_values)
    
    print(f"Average K (Full Dev):        {avg_k_full:.4f}")
    print(f"Average K (Greedy Dev):      {avg_k_greedy:.4f}")
    print(f"K Improvement on Greedy:     {avg_k_greedy - avg_k_full:.4f} ({((avg_k_greedy/avg_k_full - 1) * 100) if avg_k_full > 0 else 0:.1f}%)")
    print(f"\nAverage AUC (Full Dev):      {avg_auc_full:.4f}")
    print(f"Average AUC (Greedy Dev):    {avg_auc_greedy:.4f}")
    print(f"AUC Improvement on Greedy:   {avg_auc_greedy - avg_auc_full:.4f} ({((avg_auc_greedy/avg_auc_full - 1) * 100) if avg_auc_full > 0 else 0:.1f}%)")
    
    # Check for overfitting
    train_aucs = [m['train_auc'] for m in all_metrics]
    overfitting_layers = sum(1 for auc in train_aucs if auc > 0.99)
    print(f"\nOverfitting analysis:")
    print(f"Layers with train AUC > 0.99: {overfitting_layers}/{len(all_metrics)}")
    if overfitting_layers > 0:
        print("Warning: Some layers show potential overfitting (train AUC > 0.99)")
    else:
        print("No significant overfitting detected.")
    
    # Save metrics
    output_dir.mkdir(exist_ok=True, parents=True)
    metrics_path = output_dir / "all_layers_metrics.json"
    save_json({
        "all_metrics": all_metrics,
        "best_layer": best_layer,
        "best_metrics": best_metrics
    }, metrics_path)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save plotting data for independent plot generation
    plotting_data_path = output_dir / "plotting_data.json"
    save_plotting_data(all_metrics, plotting_data_path)
    
    # Generate all plots
    print("\n" + "="*70)
    print("Generating Plots")
    print("="*70)
    
    # Extract plotting data for use with plotting functions
    plotting_data = {
        "layers": [m["layer"] for m in all_metrics],
        "train_auc": [m["train_auc"] for m in all_metrics],
        "train_accuracy": [m["train_accuracy"] for m in all_metrics],
        "dev_auc_full": [m["dev_auc_full"] for m in all_metrics],
        "dev_accuracy_full": [m["dev_accuracy_full"] for m in all_metrics],
        "dev_k_full": [m["dev_k_full"] for m in all_metrics],
        "dev_k_star_full": [m["dev_k_star_full"] for m in all_metrics],
        "dev_auc_greedy": [m["dev_auc_greedy"] for m in all_metrics],
        "dev_accuracy_greedy": [m["dev_accuracy_greedy"] for m in all_metrics],
        "dev_k_greedy": [m["dev_k_greedy"] for m in all_metrics],
        "dev_k_star_greedy": [m["dev_k_star_greedy"] for m in all_metrics]
    }
    
    # 1. Original combined plot
    plot_path = output_dir / "layer_performance_combined.png"
    plot_layer_performance(all_metrics, plot_path)
    
    # 2. Full dev set only plots
    print("Generating full dev set plots...")
    plot_full_dev_metrics(plotting_data, output_dir)
    
    # 3. Full vs greedy comparison plots
    print("Generating comparison plots...")
    plot_full_vs_greedy_comparison(plotting_data, output_dir)
    
    # Save best probe
    best_probe, best_scaler = probes_dict[best_layer]
    save_probe(best_probe, best_scaler, best_layer, output_dir)
    
    # Save all probes (optional, useful for analysis)
    print("\n" + "="*70)
    print("Saving All Probes (for analysis)")
    print("="*70)
    all_probes_dir = output_dir / "all_layers"
    for layer_idx, (probe, scaler) in probes_dict.items():
        save_probe(probe, scaler, layer_idx, all_probes_dir)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nBest probe (layer {best_layer}) saved to: {output_dir}")
    print(f"Use this probe for internal knowledge scoring.")
    
    print(f"\nGenerated Plots:")
    print(f"  - Combined overview: layer_performance_combined.png")
    print(f"  - Full dev set only:")
    print(f"    * accuracy_full_dev.png")
    print(f"    * auc_full_dev.png")
    print(f"    * k_metrics_full_dev.png")
    print(f"  - Full vs Greedy comparison:")
    print(f"    * accuracy_comparison.png")
    print(f"    * auc_comparison.png")
    print(f"    * k_metrics_comparison.png")
    print(f"    * performance_improvement.png")
    
    print(f"\nPlotting data saved to: {plotting_data_path}")
    print(f"To regenerate plots later, use:")
    print(f"  from train_linear_probe import generate_plots_from_saved_data")
    print(f"  generate_plots_from_saved_data(Path('{plotting_data_path}'), Path('output_directory'))")


if __name__ == "__main__":
    main()
