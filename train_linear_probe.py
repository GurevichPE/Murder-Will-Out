"""
Train linear probes for internal knowledge scoring using pre-computed hidden states.

This script:
1. Loads pre-computed train hidden states from probe_training_data/
2. Loads pre-computed dev hidden states from probe_training_data/
3. For each layer, trains a logistic regression classifier
4. Evaluates probe on dev set and calculates K, K* metrics
5. Selects the best layer based on dev K
6. Saves the trained probe and scaler for the best layer

All hidden states are pre-computed - no model loading needed!
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

from knowledge_calculation import calculate_k_q


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


def calculate_k_metrics_for_dev(
    dev_hidden_states: np.ndarray,
    dev_labels: np.ndarray,
    dev_metadata: List[Dict],
    probe,
    scaler,
    balanced_evaluation: bool = True
) -> Tuple[float, float]:
    """
    Calculate K and K* metrics on dev set using pre-computed hidden states.
    
    Groups examples by question and calculates knowledge metrics.
    
    Args:
        dev_hidden_states: Hidden states for all dev examples (n_examples, hidden_dim)
        dev_labels: Labels (0 or 1) for all dev examples
        dev_metadata: Metadata with question, answer, golden_answer for each example
        probe: Trained probe
        scaler: Fitted scaler
        balanced_evaluation: If True, balance labels per question (like in appendix)
        
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
                "is_correct": []
            }
        
        # Use the actual label from the data (not golden answer comparison)
        is_correct = (dev_labels[idx] == 1)
        
        question_groups[question]["indices"].append(idx)
        question_groups[question]["scores"].append(probe_scores[idx])
        question_groups[question]["is_correct"].append(is_correct)
    
    # Calculate K_q for each question
    k_values = []
    k_star_values = []
    skipped_questions = 0
    
    for question, data in question_groups.items():
        indices = np.array(data["indices"])
        scores = np.array(data["scores"])
        is_correct = np.array(data["is_correct"])
        
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
        
        if balanced_evaluation:
            # BALANCED EVALUATION: Sample equal numbers of correct/incorrect 
            # Per appendix: "balance test labels by sampling one correct and one incorrect answer per question"
            
            # Sample equal numbers (minimum of the two)
            n_samples = min(len(correct_indices), len(incorrect_indices), 10)  # Max 10 each for efficiency
            
            if n_samples > 0:
                # Randomly sample n_samples from each group
                np.random.seed(42)  # For reproducibility
                sampled_correct = np.random.choice(correct_indices, size=n_samples, replace=False)
                sampled_incorrect = np.random.choice(incorrect_indices, size=n_samples, replace=False)
                
                # Combine samples
                sampled_indices = np.concatenate([sampled_correct, sampled_incorrect])
                balanced_scores = scores[sampled_indices]
                balanced_labels = is_correct[sampled_indices]
                
                # Calculate K_q on balanced data
                k_q = calculate_k_q(balanced_scores, balanced_labels)
            else:
                k_q = np.nan
        else:
            # UNBALANCED EVALUATION: Use all data (original approach)
            k_q = calculate_k_q(scores, is_correct)
        
        # Only include valid K_q values (not NaN)
        if not np.isnan(k_q):
            k_values.append(k_q)
            
            # K* = 1 if K = 1, else 0
            k_star = 1.0 if k_q == 1.0 else 0.0
            k_star_values.append(k_star)
    
    balance_mode = "BALANCED" if balanced_evaluation else "UNBALANCED"
    print(f"Evaluation mode: {balance_mode}")
    print(f"Processed {len(question_groups)} questions, skipped {skipped_questions} questions")
    print(f"Valid K_q calculations: {len(k_values)}")
    
    # Compute means only over valid questions
    mean_k = np.mean(k_values) if k_values else 0.0
    mean_k_star = np.mean(k_star_values) if k_star_values else 0.0
    
    return mean_k, mean_k_star


def train_probe_for_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    dev_metadata: List[Dict],
    layer_idx: int,
    random_state: int = 42
) -> Tuple[LogisticRegression, StandardScaler, Dict[str, float]]:
    """
    Train a logistic regression probe for a specific layer using pre-computed hidden states.
    
    Args:
        X_train: Training hidden states
        y_train: Training labels
        X_dev: Dev hidden states
        y_dev: Dev labels
        dev_metadata: Dev metadata for calculating K metrics
        layer_idx: Layer index (for logging)
        random_state: Random seed
        
    Returns:
        Tuple of (trained_probe, scaler, metrics_dict)
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    
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
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    
    # Evaluate on dev set (K metrics) - use balanced evaluation to match paper
    print(f"  Layer {layer_idx}: Evaluating K metrics...")
    dev_k_balanced, dev_k_star_balanced = calculate_k_metrics_for_dev(
        X_dev, y_dev, dev_metadata, probe, scaler, balanced_evaluation=True
    )
    
    # Also run unbalanced for comparison
    print(f"  Layer {layer_idx}: Running unbalanced evaluation for comparison...")
    dev_k_unbalanced, dev_k_star_unbalanced = calculate_k_metrics_for_dev(
        X_dev, y_dev, dev_metadata, probe, scaler, balanced_evaluation=False
    )
    
    print(f"  Layer {layer_idx} Results:")
    print(f"    Balanced:   K = {dev_k_balanced:.3f}, K* = {dev_k_star_balanced:.3f}")  
    print(f"    Unbalanced: K = {dev_k_unbalanced:.3f}, K* = {dev_k_star_unbalanced:.3f}")
    
    # Use balanced for layer selection (following paper)
    dev_k, dev_k_star = dev_k_balanced, dev_k_star_balanced
    
    metrics = {
        "layer": layer_idx,
        "train_auc": float(train_auc),
        "dev_k": float(dev_k),
        "dev_k_star": float(dev_k_star),
        "dev_k_balanced": float(dev_k_balanced),
        "dev_k_star_balanced": float(dev_k_star_balanced), 
        "dev_k_unbalanced": float(dev_k_unbalanced),
        "dev_k_star_unbalanced": float(dev_k_star_unbalanced)
    }
    
    return probe, scaler, metrics


def train_all_layers(
    train_hidden_states: Dict[int, np.ndarray],
    train_labels: np.ndarray,
    dev_hidden_states: Dict[int, np.ndarray],
    dev_labels: np.ndarray,
    dev_metadata: List[Dict]
) -> Tuple[Dict[int, Tuple], List[Dict]]:
    """
    Train probes for all layers using pre-computed hidden states.
    
    Args:
        train_hidden_states: Dict mapping layer_idx -> train hidden states
        train_labels: Train labels
        dev_hidden_states: Dict mapping layer_idx -> dev hidden states
        dev_labels: Dev labels
        dev_metadata: Dev metadata for K calculations
        
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
            dev_hidden_states[layer_idx],
            dev_labels,
            dev_metadata,
            layer_idx
        )
        
        probes_dict[layer_idx] = (probe, scaler)
        all_metrics.append(metrics)
    
    return probes_dict, all_metrics


def select_best_layer(
    all_metrics: List[Dict],
    criterion: str = "dev_k"
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
    Plot K performance metrics across all layers.
    
    Args:
        all_metrics: List of metrics dicts for each layer
        output_path: Path to save the plot
    """
    layers = [m["layer"] for m in all_metrics]
    train_auc = [m["train_auc"] for m in all_metrics]
    dev_k = [m["dev_k"] for m in all_metrics]
    dev_k_star = [m["dev_k_star"] for m in all_metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Train AUC vs Dev K
    ax1.plot(layers, train_auc, 'o-', label='Train AUC', linewidth=2, markersize=4, color='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(layers, dev_k, 's-', label='Dev K', linewidth=2, markersize=4, color='green')
    
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Train AUC', fontsize=12, fontweight='bold', color='blue')
    ax1_twin.set_ylabel('Dev K', fontsize=12, fontweight='bold', color='green')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    ax1.set_title('Train AUC vs Dev K by Layer', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Highlight best layer
    best_idx = np.argmax(dev_k)
    ax1.axvline(x=layers[best_idx], color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Plot K and K*
    ax2.plot(layers, dev_k, 'o-', label='Dev K', linewidth=2, markersize=4)
    ax2.plot(layers, dev_k_star, 's-', label='Dev K*', linewidth=2, markersize=4)
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Knowledge Metric', fontsize=12, fontweight='bold')
    ax2.set_title('Dev Knowledge by Layer', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Highlight best layer
    ax2.axvline(x=layers[best_idx], color='red', linestyle='--', alpha=0.5, linewidth=2, label=f'Best: Layer {layers[best_idx]}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
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
        probe: Trained logistic regression model
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
    relations = ["P40", "P50", "P176", "P264"]
    
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
    
    # Train probes for all layers
    print("\n" + "="*70)
    print("Training Probes")
    print("="*70)
    probes_dict, all_metrics = train_all_layers(
        train_hidden_states,
        train_labels,
        dev_hidden_states,
        dev_labels,
        dev_metadata
    )
    
    # Print results
    print("\n" + "="*80)
    print("Results Summary - BALANCED EVALUATION (Paper Method)")
    print("="*80)
    print(f"{'Layer':<8} {'Train AUC':<12} {'Dev K (Bal)':<12} {'Dev K* (Bal)':<12} {'Dev K (Unbal)':<14} {'Dev K* (Unbal)':<14}")
    print("-"*80)
    for metrics in all_metrics:
        print(f"{metrics['layer']:<8} "
              f"{metrics['train_auc']:<12.4f} "
              f"{metrics['dev_k_balanced']:<12.4f} "
              f"{metrics['dev_k_star_balanced']:<12.4f} "
              f"{metrics['dev_k_unbalanced']:<14.4f} "
              f"{metrics['dev_k_star_unbalanced']:<14.4f}")
    
    # Select best layer
    best_layer = select_best_layer(all_metrics, criterion="dev_k")
    best_metrics = all_metrics[best_layer]
    
    print("\n" + "="*70)
    print(f"Best Layer: {best_layer} (selected based on balanced K)")
    print("="*70)
    print(f"Train AUC: {best_metrics['train_auc']:.4f}")
    print(f"Dev K (Balanced):   {best_metrics['dev_k_balanced']:.4f}")
    print(f"Dev K* (Balanced):  {best_metrics['dev_k_star_balanced']:.4f}")
    print(f"Dev K (Unbalanced): {best_metrics['dev_k_unbalanced']:.4f}")
    print(f"Dev K* (Unbalanced):{best_metrics['dev_k_star_unbalanced']:.4f}")
    print("")
    print("NOTE: Balanced evaluation should give results closer to paper (~0.8)")
    print("      Unbalanced shows the effect of class imbalance (~0.5)")
    
    # Save metrics
    output_dir.mkdir(exist_ok=True, parents=True)
    metrics_path = output_dir / "all_layers_metrics.json"
    save_json({
        "all_metrics": all_metrics,
        "best_layer": best_layer,
        "best_metrics": best_metrics
    }, metrics_path)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Plot performance
    plot_path = output_dir / "layer_performance.png"
    plot_layer_performance(all_metrics, plot_path)
    
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


if __name__ == "__main__":
    main()
