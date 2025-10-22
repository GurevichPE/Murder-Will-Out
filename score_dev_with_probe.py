"""
Score dev dataset with trained linear probe (internal scoring).

This script:
1. Loads the trained probe and scaler (best layer)
2. Loads pre-computed dev hidden states from probe_training_data (pickle format)
3. For each entry in dev data, looks up the pre-computed hidden states
4. Scores them using the trained probe
5. Adds probe_scores to the data
6. Calculates internal knowledge metrics (K, K*)
7. Compares internal vs external knowledge (hidden knowledge gap)

Note: Uses pre-computed hidden states from generate_data_for_probe.py,
not extracting them on-the-fly. This ensures consistency with training.
Dev/test data is stored as pickle for efficiency (much smaller than JSON).
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm.auto import tqdm

from knowledge_calculation import calculate_knowledge_for_dataset


def load_json(path: Path) -> List[Dict]:
    """Load JSON data from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> List[Dict]:
    """Load pickle data from file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data, path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_probe(probe_dir: Path, layer_idx: int) -> Tuple:
    """
    Load trained probe and scaler for a specific layer.
    
    Args:
        probe_dir: Directory containing saved probe models
        layer_idx: Layer index to load
        
    Returns:
        Tuple of (probe, scaler)
    """
    probe_path = probe_dir / f"probe_layer_{layer_idx}.pkl"
    scaler_path = probe_dir / f"scaler_layer_{layer_idx}.pkl"
    
    if not probe_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Probe or scaler not found for layer {layer_idx} in {probe_dir}"
        )
    
    with open(probe_path, 'rb') as f:
        probe = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return probe, scaler


def load_precomputed_hidden_states(
    probe_data_dir: Path,
    relations: List[str],
    split: str = "dev"
) -> Dict[Tuple[str, str], Dict[int, np.ndarray]]:
    """
    Load pre-computed hidden states from probe training data.
    
    Args:
        probe_data_dir: Directory containing probe data files
        relations: List of relation codes
        split: Data split ("dev" or "test")
        
    Returns:
        Dictionary mapping (question, answer) -> {layer_idx: hidden_state}
    """
    hidden_states_lookup = {}
    
    print(f"\nLoading pre-computed {split} hidden states from pickle files...")
    for relation in relations:
        # Dev/test data is stored as pickle (much smaller than JSON)
        file_path = probe_data_dir / f"{relation}.{split}.probe_data.pkl"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
        
        print(f"  Loading {relation}...")
        data = load_pickle(file_path)
        
        for example in data:
            question = example["question"]
            answer = example["answer"]
            hidden_states = example["hidden_states"]
            
            # Convert hidden states from dict of lists to dict of numpy arrays
            hidden_states_np = {
                int(layer_idx): np.array(hs, dtype=np.float16)
                for layer_idx, hs in hidden_states.items()
            }
            
            # Key: (question, answer) tuple
            key = (question, answer)
            hidden_states_lookup[key] = hidden_states_np
    
    print(f"Loaded hidden states for {len(hidden_states_lookup)} (question, answer) pairs")
    return hidden_states_lookup


def score_entry_with_probe(
    entry: Dict,
    hidden_states_lookup: Dict[Tuple[str, str], Dict[int, np.ndarray]],
    probe,
    scaler,
    layer_idx: int
) -> List[float]:
    """
    Score all answer candidates for an entry using the trained probe and pre-computed hidden states.
    
    Args:
        entry: Data entry with question and answer_labels
        hidden_states_lookup: Pre-computed hidden states lookup
        probe: Trained logistic regression probe
        scaler: Fitted StandardScaler
        layer_idx: Layer index to use
        
    Returns:
        List of probe scores (probabilities) for each answer
    """
    question = entry["question"]
    answers = [answer for answer, _ in entry["answer_labels"]]
    
    # Look up pre-computed hidden states for all answers
    hidden_states_list = []
    for answer in answers:
        key = (question, answer)
        if key not in hidden_states_lookup:
            raise KeyError(
                f"Hidden states not found for question-answer pair:\n"
                f"  Question: {question}\n"
                f"  Answer: {answer}\n"
                f"Make sure you've run generate_data_for_probe.py for dev split!"
            )
        
        # Get hidden state for this layer
        hidden_state = hidden_states_lookup[key][layer_idx]
        hidden_states_list.append(hidden_state)
    
    # Stack into array: (num_answers, hidden_dim)
    hidden_states = np.vstack(hidden_states_list)
    
    # Standardize all at once
    hidden_states_scaled = scaler.transform(hidden_states)
    
    # Get probe scores for all answers at once (probability of being correct)
    probe_scores = probe.predict_proba(hidden_states_scaled)[:, 1]
    
    return probe_scores.tolist()


def process_one_relation(
    relation: str,
    data_path: Path,
    hidden_states_lookup: Dict[Tuple[str, str], Dict[int, np.ndarray]],
    probe,
    scaler,
    layer_idx: int
) -> List[Dict]:
    """
    Process one relation file and add probe scores using pre-computed hidden states.
    
    Args:
        relation: Relation code (e.g., "P264")
        data_path: Path to dev data directory
        hidden_states_lookup: Pre-computed hidden states
        probe: Trained probe
        scaler: Fitted scaler
        layer_idx: Layer index
        
    Returns:
        Data with probe_scores added
    """
    file_path = data_path / f"{relation}.dev.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"\nLoading {relation} dev data...")
    data = load_json(file_path)
    print(f"Loaded {len(data)} entries")
    
    print(f"Scoring with probe (layer {layer_idx}) using pre-computed hidden states...")
    for entry in tqdm(data, desc=f"Processing {relation}"):
        probe_scores = score_entry_with_probe(
            entry, hidden_states_lookup, probe, scaler, layer_idx
        )
        entry["probe_scores"] = probe_scores
    
    return data


def calculate_knowledge_metrics(
    data: List[Dict],
    relation: str
) -> Dict[str, Dict[str, float]]:
    """
    Calculate knowledge metrics using both external and internal scoring.
    
    Args:
        data: List of entries with p_true and probe_scores
        relation: Relation code
        
    Returns:
        Dictionary with metrics for both scoring methods
    """
    # Calculate with external scoring (p_true)
    metrics_external = calculate_knowledge_for_dataset(data, score_key="p_true")
    
    # Calculate with internal scoring (probe)
    metrics_internal = calculate_knowledge_for_dataset(data, score_key="probe_scores")
    
    # Calculate gaps
    k_gap_absolute = metrics_internal["mean_k"] - metrics_external["mean_k"]
    k_gap_relative = (
        k_gap_absolute / metrics_external["mean_k"] 
        if metrics_external["mean_k"] > 0 else 0.0
    )
    
    k_star_gap_absolute = metrics_internal["mean_k_star"] - metrics_external["mean_k_star"]
    k_star_gap_relative = (
        k_star_gap_absolute / metrics_external["mean_k_star"]
        if metrics_external["mean_k_star"] > 0 else 0.0
    )
    
    return {
        "relation": relation,
        "external": metrics_external,
        "internal": metrics_internal,
        "gaps": {
            "k_absolute": k_gap_absolute,
            "k_relative": k_gap_relative,
            "k_star_absolute": k_star_gap_absolute,
            "k_star_gap_relative": k_star_gap_relative
        }
    }


def print_comparison_table(all_metrics: List[Dict]):
    """
    Print a formatted comparison table of internal vs external knowledge.
    
    Args:
        all_metrics: List of metrics dicts for each relation
    """
    print("\n" + "="*90)
    print("Internal vs External Knowledge Comparison")
    print("="*90)
    print(f"{'Relation':<12} {'Ext K':<10} {'Int K':<10} {'Gap %':<10} "
          f"{'Ext K*':<10} {'Int K*':<10} {'Gap %':<10}")
    print("-"*90)
    
    for metrics in all_metrics:
        rel = metrics["relation"]
        ext_k = metrics["external"]["mean_k"]
        int_k = metrics["internal"]["mean_k"]
        k_gap = metrics["gaps"]["k_relative"] * 100
        
        ext_k_star = metrics["external"]["mean_k_star"]
        int_k_star = metrics["internal"]["mean_k_star"]
        k_star_gap = metrics["gaps"]["k_star_gap_relative"] * 100
        
        print(f"{rel:<12} {ext_k:<10.4f} {int_k:<10.4f} {k_gap:<10.1f} "
              f"{ext_k_star:<10.4f} {int_k_star:<10.4f} {k_star_gap:<10.1f}")
    
    # Calculate averages
    avg_ext_k = np.mean([m["external"]["mean_k"] for m in all_metrics])
    avg_int_k = np.mean([m["internal"]["mean_k"] for m in all_metrics])
    avg_k_gap = np.mean([m["gaps"]["k_relative"] for m in all_metrics]) * 100
    
    avg_ext_k_star = np.mean([m["external"]["mean_k_star"] for m in all_metrics])
    avg_int_k_star = np.mean([m["internal"]["mean_k_star"] for m in all_metrics])
    avg_k_star_gap = np.mean([m["gaps"]["k_star_gap_relative"] for m in all_metrics]) * 100
    
    print("-"*90)
    print(f"{'Average':<12} {avg_ext_k:<10.4f} {avg_int_k:<10.4f} {avg_k_gap:<10.1f} "
          f"{avg_ext_k_star:<10.4f} {avg_int_k_star:<10.4f} {avg_k_star_gap:<10.1f}")
    print("="*90)
    
    print("\nKey:")
    print("  Ext K / Int K: External / Internal knowledge (mean)")
    print("  Gap %: Relative improvement of internal over external")
    print("  K*: Perfect knowledge indicator")


def main():
    """
    Main function to score dev data with probe and calculate internal knowledge.
    Uses pre-computed hidden states from generate_data_for_probe.py.
    """
    # Configuration
    probe_dir = Path("./models/probes")
    probe_data_dir = Path("./data/probe_training_data")
    dev_data_path = Path("./data/sampled_labeled_answers_1000_temp1/dev")
    output_path = Path("./data/scored_with_probe/dev")
    output_path.mkdir(exist_ok=True, parents=True)
    
    relations = ["P40", "P50", "P176", "P264"]
    
    print("="*70)
    print("Internal Knowledge Scoring with Linear Probe")
    print("="*70)
    print("\nUsing pre-computed hidden states from generate_data_for_probe.py")
    
    # Load best probe
    print("\nLoading best probe...")
    metrics_file = probe_dir / "all_layers_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_file}\n"
            "Please run train_linear_probe.py first!"
        )
    
    probe_metrics = load_json(metrics_file)
    best_layer = probe_metrics["best_layer"]
    best_metrics = probe_metrics["best_metrics"]
    
    print(f"Best layer: {best_layer}")
    print(f"Dev AUC: {best_metrics['dev_auc']:.4f}")
    print(f"Dev Accuracy: {best_metrics['dev_accuracy']:.4f}")
    
    probe, scaler = load_probe(probe_dir, best_layer)
    print("Probe and scaler loaded successfully!")
    
    # Load pre-computed hidden states
    hidden_states_lookup = load_precomputed_hidden_states(
        probe_data_dir, relations, split="dev"
    )
    
    # Process each relation
    all_metrics = []
    for relation in relations:
        print("\n" + "="*70)
        print(f"Processing {relation}")
        print("="*70)
        
        # Score with probe using pre-computed hidden states
        data_with_scores = process_one_relation(
            relation,
            dev_data_path,
            hidden_states_lookup,
            probe,
            scaler,
            best_layer
        )
        
        # Save data with probe scores
        output_file = output_path / f"{relation}.dev.json"
        save_json(data_with_scores, output_file)
        print(f"Saved scored data to: {output_file}")
        
        # Calculate knowledge metrics
        print("\nCalculating knowledge metrics...")
        metrics = calculate_knowledge_metrics(data_with_scores, relation)
        all_metrics.append(metrics)
        
        # Print individual metrics
        print(f"\n{relation} Results:")
        print(f"  External K (p_true):  {metrics['external']['mean_k']:.4f}")
        print(f"  Internal K (probe):   {metrics['internal']['mean_k']:.4f}")
        print(f"  Hidden knowledge gap: {metrics['gaps']['k_relative']*100:.1f}%")
        print(f"  External K*:          {metrics['external']['mean_k_star']:.4f}")
        print(f"  Internal K*:          {metrics['internal']['mean_k_star']:.4f}")
    
    # Save all metrics
    metrics_output = output_path / "knowledge_comparison_metrics.json"
    save_json({
        "best_layer": best_layer,
        "probe_performance": best_metrics,
        "relations": all_metrics,
        "summary": {
            "avg_external_k": float(np.mean([m["external"]["mean_k"] for m in all_metrics])),
            "avg_internal_k": float(np.mean([m["internal"]["mean_k"] for m in all_metrics])),
            "avg_k_gap_relative": float(np.mean([m["gaps"]["k_relative"] for m in all_metrics])),
            "avg_external_k_star": float(np.mean([m["external"]["mean_k_star"] for m in all_metrics])),
            "avg_internal_k_star": float(np.mean([m["internal"]["mean_k_star"] for m in all_metrics])),
        }
    }, metrics_output)
    print(f"\n\nAll metrics saved to: {metrics_output}")
    
    # Print comparison table
    print_comparison_table(all_metrics)
    
    # Print conclusion
    avg_gap = np.mean([m["gaps"]["k_relative"] for m in all_metrics]) * 100
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print(f"Average hidden knowledge gap: {avg_gap:.1f}%")
    
    if avg_gap > 5:
        print("\n✓ HIDDEN KNOWLEDGE DETECTED!")
        print("  The model encodes more knowledge internally than it expresses externally.")
        print(f"  Internal scoring (probe) shows {avg_gap:.1f}% relative improvement over")
        print("  external scoring (p_true).")
    else:
        print("\n✗ Limited hidden knowledge detected.")
        print("  Internal and external knowledge are similar.")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()

