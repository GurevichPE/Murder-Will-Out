import numpy as np
import json
from pathlib import Path
from typing import Union, List, Dict, Tuple
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

def load_json(path:Union[str, Path]) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def get_accuracy(answer_labels: List[Tuple[str]]) -> Tuple[float, str]:
    ts, fs = 0, 0
    for _, label in answer_labels:
        if label in ["EXACT_MATCH", "True"]:
            ts += 1
        else:
            fs += 1
    if fs == 0:
        flag = "all_true"
    elif ts == 0:
        flag = "all_false"
    else:
        flag = "none"
    return ts / (ts + fs), flag

def get_roc_auc(answer_labels: List[Tuple[str]], p_true: List[float]) -> float:
    """
    Calculate ROC AUC score using predicted probabilities and true labels.
    """
    if not p_true or len(p_true) != len(answer_labels):
        print("Not p_true or len(p_true) != len(answer_labels)")
        return None

    # Debug: check the structure
    # print(f"Debug: p_true length={len(p_true)}, answer_labels length={len(answer_labels)}")
    # print(f"Debug: first few p_true values: {p_true[:3]}")
    # print(f"Debug: first few labels: {[label for _, label in answer_labels[:3]]}")

    y_true = []
    for _, label in answer_labels:
        if label in ["EXACT_MATCH", "True"]:
            y_true.append(1)
        else:
            y_true.append(0)

    # Convert probabilities to numpy array
    y_scores = np.array(p_true)

    # Handle cases where all predictions are the same
    if len(np.unique(y_true)) < 2:
        print("len(np.unique(y_true)) < 2")
        return None

    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError as e:
        print(f"ROC AUC calculation error: {e}")
        return None

def evaluate_one_entry(entry:Dict) -> Tuple[Dict, float, str, float]:
    answer_labels = entry["answer_labels"]
    accuracy, flag = get_accuracy(answer_labels)

    # Calculate ROC AUC if p_true exists
    roc_auc = None
    if "p_true" in entry:
        roc_auc = get_roc_auc(answer_labels, entry["p_true"])


    entry["accuracy"] = accuracy
    entry["flag"] = flag
    entry["roc_auc"] = roc_auc
    return entry, accuracy, flag, roc_auc

def save_all(path: Union[str, Path], data: Dict, stats: Dict) -> None:
    # Save the main data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # Save statistics to a parallel file with '.stats.json' appended to the stem
    path = Path(path)
    stats_path = path.with_name(path.stem + ".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def evaluate_one_code(path:Union[str, Path]) -> Dict:
    """Evaluate one code file and return comprehensive statistics."""
    data = load_json(path)
    accs = []
    roc_aucs = []
    flag_stats = ddict(float)
    valid_roc_count = 0

    for i in range(len(data)):
        entry = data[i]
        entry, accuracy, flag, roc_auc = evaluate_one_entry(entry)
        accs.append(accuracy)
        flag_stats[flag] += 1
        data[i] = entry

        if roc_auc is not None:
            roc_aucs.append(roc_auc)
            valid_roc_count += 1

    mean_acc = sum(accs) / len(accs)
    mean_roc_auc = sum(roc_aucs) / valid_roc_count if valid_roc_count > 0 else None

    for key in flag_stats.keys():
        flag_stats[key] /= len(data)
    flag_stats["accuracy"] = mean_acc
    flag_stats["roc_auc"] = mean_roc_auc
    flag_stats["valid_roc_count"] = valid_roc_count

    save_all(path, data, flag_stats)
    return flag_stats

def create_comparison_plots(modes_stats: Dict[str, Dict]) -> None:
    """Create comparison plots for accuracy and ROC AUC across modes."""

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    modes = list(modes_stats.keys())
    accuracies = [stats["accuracy"] for stats in modes_stats.values()]
    roc_aucs = [stats.get("roc_auc") if stats.get("roc_auc") is not None else 0 for stats in modes_stats.values()]

    # Accuracy comparison plot
    bars1 = axes[0].bar(modes, accuracies, color='skyblue', alpha=0.8)
    axes[0].set_title('Accuracy Comparison Across Modes', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_ylim(0, max(accuracies) * 1.1 if accuracies else 1)

    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        if height is not None and height > 0:
            y_pos = height + max(accuracies) * 0.01 if accuracies else 0.01
            axes[0].text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    # ROC AUC comparison plot
    bars2 = axes[1].bar(modes, roc_aucs, color='lightcoral', alpha=0.8)
    axes[1].set_title('ROC AUC Comparison Across Modes', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('ROC AUC', fontsize=12)
    max_roc = max(roc_aucs) if roc_aucs and max(roc_aucs) > 0 else 1.0
    axes[1].set_ylim(0, max_roc * 1.1)

    # Add value labels on bars for ROC AUC
    for bar, roc in zip(bars2, roc_aucs):
        height = bar.get_height()
        if height is not None and height > 0:
            y_pos = height + max_roc * 0.01
            if roc > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2., y_pos,
                            f'{roc:.3f}', ha='center', va='bottom', fontsize=10)
            else:
                axes[1].text(bar.get_x() + bar.get_width()/2., y_pos,
                            'N/A', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('./comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create a detailed table showing all statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS COMPARISON")
    print("="*80)
    print(f"{'Mode':<10} {'Accuracy':<12} {'ROC AUC':<12} {'Valid ROC':<12} {'All True':<12} {'All False':<12} {'Mixed':<12}")
    print("-"*80)
    for mode, stats in modes_stats.items():
        acc = f"{stats['accuracy']:.4f}"
        roc_auc = f"{stats.get('roc_auc', 'N/A'):.4f}" if stats.get('roc_auc') else "N/A"
        valid_roc = str(int(stats.get('valid_roc_count', 0)))
        all_true = f"{stats.get('all_true', 0):.3f}"
        all_false = f"{stats.get('all_false', 0):.3f}"
        mixed = f"{stats.get('none', 0):.3f}"
        print(f"{mode:<10} {acc:<12} {roc_auc:<12} {valid_roc:<12} {all_true:<12} {all_false:<12} {mixed:<12}")
    print("="*80)

def evaluate_all_modes(data_path: Union[str, Path]) -> Dict[str, Dict]:
    """Evaluate all modes and return comprehensive statistics for dev split only."""
    modes_stats = {}
    data_path = Path(data_path)

    # Process only dev split
    dev_path = data_path / "test"
    if not dev_path.exists():
        raise ValueError("Dev directory not found in data path")

    print("Processing dev split only...")
    for mode_file in dev_path.glob("P*.test.json"):
        mode = mode_file.stem.split('.')[0]  # Extract P40, P50, etc.
        print(f"Processing {mode} in test...")
        stats = evaluate_one_code(mode_file)
        modes_stats[mode] = stats

        # Verify that ROC AUC is calculated properly
        roc_auc = stats.get("roc_auc")
        if roc_auc is None:
            print(f"Warning: {mode} has no ROC AUC calculated (p_true may be missing)")
        else:
            print(f"ROC AUC for {mode}: {roc_auc:.4f}")

    return modes_stats

def main():
    import os
    DATA_PATH = Path("./data/sampled_labeled_answers_1000_temp1")

    print("Evaluating all modes and generating comparison plots...")
    modes_stats = evaluate_all_modes(DATA_PATH)
    create_comparison_plots(modes_stats)
    print("Evaluation complete! Check 'comparison_plots.png' for visualizations.")

if __name__ == "__main__":
    main()

