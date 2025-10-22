"""
Plot heatmaps of probe performance (ROC AUC) across layers and relations.

This script creates visualizations showing:
1. Overall probe performance across layers
2. Per-relation probe performance across layers (if evaluated separately)
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


def load_json(path: Path) -> dict:
    """Load JSON data from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_probe_data(
    data_dir: Path,
    relation: str,
    split: str = "dev"
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Load probe data for a single relation.
    
    Returns:
        Tuple of (hidden_states_by_layer, labels)
    """
    file_path = data_dir / f"{relation}.{split}.probe_data.json"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = load_json(file_path)
    
    # Organize by layer
    num_layers = len(data[0]["hidden_states"])
    hidden_states_by_layer = {i: [] for i in range(num_layers)}
    labels = []
    
    for example in data:
        for layer_idx in range(num_layers):
            hidden_state = example["hidden_states"][str(layer_idx)]
            hidden_states_by_layer[layer_idx].append(hidden_state)
        labels.append(example["label"])
    
    # Convert to numpy arrays
    for layer_idx in range(num_layers):
        hidden_states_by_layer[layer_idx] = np.array(hidden_states_by_layer[layer_idx])
    labels = np.array(labels)
    
    return hidden_states_by_layer, labels


def load_probe_and_scaler(models_dir: Path, layer_idx: int):
    """Load probe and scaler for a specific layer."""
    probe_path = models_dir / "all_layers" / f"probe_layer_{layer_idx}.pkl"
    scaler_path = models_dir / "all_layers" / f"scaler_layer_{layer_idx}.pkl"
    
    with open(probe_path, 'rb') as f:
        probe = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return probe, scaler


def evaluate_probe_on_relation(
    probe,
    scaler,
    hidden_states: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """Evaluate a probe on a specific dataset."""
    # Scale features
    X_scaled = scaler.transform(hidden_states)
    
    # Get predictions
    y_pred = probe.predict(X_scaled)
    y_proba = probe.predict_proba(X_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(labels, y_pred)
    auc = roc_auc_score(labels, y_proba)
    
    return {
        "accuracy": accuracy,
        "auc": auc
    }


def compute_per_relation_metrics(
    probe_data_dir: Path,
    models_dir: Path,
    relations: List[str],
    split: str = "dev"
) -> Dict[str, List[Dict]]:
    """
    Compute metrics for each relation and layer.
    
    Returns:
        Dict mapping relation -> list of metrics per layer
    """
    print(f"\n{'='*70}")
    print(f"Computing Per-Relation Metrics on {split.upper()} set")
    print(f"{'='*70}\n")
    
    relation_metrics = {}
    
    for relation in relations:
        print(f"\nProcessing {relation}...")
        
        # Load relation data
        hidden_states_by_layer, labels = load_probe_data(
            probe_data_dir, relation, split
        )
        num_layers = len(hidden_states_by_layer)
        
        # Evaluate each layer's probe on this relation
        metrics_list = []
        for layer_idx in tqdm(range(num_layers), desc=f"{relation} layers"):
            # Load probe for this layer
            probe, scaler = load_probe_and_scaler(models_dir, layer_idx)
            
            # Evaluate
            metrics = evaluate_probe_on_relation(
                probe,
                scaler,
                hidden_states_by_layer[layer_idx],
                labels
            )
            metrics["layer"] = layer_idx
            metrics_list.append(metrics)
        
        relation_metrics[relation] = metrics_list
        
        # Print summary
        best_layer = max(metrics_list, key=lambda x: x["auc"])
        print(f"  Best layer for {relation}: {best_layer['layer']} "
              f"(AUC: {best_layer['auc']:.4f})")
    
    return relation_metrics


def plot_overall_heatmap(
    all_metrics: List[Dict],
    output_path: Path
):
    """
    Plot a simple heatmap showing train/dev AUC across layers (vertical layout).
    
    Args:
        all_metrics: List of metrics from train_linear_probe.py
        output_path: Where to save the plot
    """
    layers = [m["layer"] for m in all_metrics]
    num_layers = len(layers)
    
    # Create data matrix: num_layers rows x 2 columns (train, dev) - TRANSPOSED
    data = np.zeros((num_layers, 2))
    data[:, 0] = [m["train_auc"] for m in all_metrics]
    data[:, 1] = [m["dev_auc"] for m in all_metrics]
    
    # Create heatmap - vertical layout
    fig, ax = plt.subplots(figsize=(5, max(10, num_layers * 0.3)))
    
    sns.heatmap(
        data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0,
        xticklabels=['Train AUC', 'Dev AUC'],
        yticklabels=layers,
        cbar_kws={'label': 'ROC-AUC'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    # Highlight best dev layer
    best_dev_idx = np.argmax(data[:, 1])
    ax.add_patch(plt.Rectangle((1, best_dev_idx), 1, 1, 
                               fill=False, edgecolor='blue', lw=3))
    
    ax.set_title('Probe Performance Across Layers (Overall)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Layer Index', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved overall heatmap to: {output_path}")
    plt.close()


def plot_per_relation_heatmap(
    relation_metrics: Dict[str, List[Dict]],
    metric_name: str,
    output_path: Path,
    vmin: float = None,
    vmax: float = None
):
    """
    Plot heatmap showing metric across relations and layers (vertical layout).
    
    Args:
        relation_metrics: Dict mapping relation -> list of metrics
        metric_name: Which metric to plot ('auc' or 'accuracy')
        output_path: Where to save the plot
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
    """
    relations = sorted(relation_metrics.keys())
    num_layers = len(relation_metrics[relations[0]])
    
    # Create data matrix: layers x relations - TRANSPOSED
    data = np.zeros((num_layers, len(relations)))
    for i, relation in enumerate(relations):
        for layer_metrics in relation_metrics[relation]:
            layer_idx = layer_metrics["layer"]
            data[layer_idx, i] = layer_metrics[metric_name]
    
    # Set default vmin/vmax if not provided
    if vmin is None:
        vmin = max(0.5, data.min() - 0.05)
    if vmax is None:
        vmax = min(1.0, data.max() + 0.05)
    
    # Create heatmap - vertical layout
    fig, ax = plt.subplots(figsize=(len(relations) * 1.5 + 2, max(12, num_layers * 0.35)))
    
    sns.heatmap(
        data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=vmin,
        vmax=vmax,
        xticklabels=relations,
        yticklabels=range(num_layers),
        cbar_kws={'label': 'ROC-AUC' if metric_name == 'auc' else 'Accuracy'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    # Highlight best layer for each relation
    for i, relation in enumerate(relations):
        best_layer_idx = np.argmax(data[:, i])
        ax.add_patch(plt.Rectangle((i, best_layer_idx), 1, 1, 
                                   fill=False, edgecolor='blue', lw=2))
    
    metric_title = 'ROC-AUC' if metric_name == 'auc' else 'Accuracy'
    ax.set_title(f'Probe {metric_title} Across Layers and Relations', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {metric_name} heatmap to: {output_path}")
    plt.close()


def plot_combined_heatmaps(
    all_metrics: List[Dict],
    relation_metrics: Dict[str, List[Dict]],
    output_dir: Path
):
    """
    Create a comprehensive figure with multiple heatmaps (vertical layout).
    """
    relations = sorted(relation_metrics.keys())
    num_layers = len(all_metrics)
    
    # Create figure with subplots - vertical layout
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35, 
                          height_ratios=[1, 1.5],
                          width_ratios=[1, 1, 1])
    
    # 1. Overall Train/Dev AUC - vertical
    ax1 = fig.add_subplot(gs[0, 0])
    data_overall = np.zeros((num_layers, 2))
    data_overall[:, 0] = [m["train_auc"] for m in all_metrics]
    data_overall[:, 1] = [m["dev_auc"] for m in all_metrics]
    
    sns.heatmap(
        data_overall,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0,
        xticklabels=['Train', 'Dev'],
        yticklabels=range(num_layers),
        cbar_kws={'label': 'ROC-AUC'},
        ax=ax1,
        linewidths=0.5,
        linecolor='gray'
    )
    best_dev_idx = np.argmax(data_overall[:, 1])
    ax1.add_patch(plt.Rectangle((1, best_dev_idx), 1, 1, 
                                fill=False, edgecolor='blue', lw=3))
    ax1.set_title('Overall Probe Performance', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Layer', fontsize=9)
    
    # 2. Per-Relation AUC - vertical
    ax2 = fig.add_subplot(gs[1, :2])
    data_auc = np.zeros((num_layers, len(relations)))
    for i, relation in enumerate(relations):
        for layer_metrics in relation_metrics[relation]:
            layer_idx = layer_metrics["layer"]
            data_auc[layer_idx, i] = layer_metrics["auc"]
    
    sns.heatmap(
        data_auc,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=max(0.5, data_auc.min() - 0.05),
        vmax=min(1.0, data_auc.max() + 0.05),
        xticklabels=relations,
        yticklabels=range(num_layers),
        cbar_kws={'label': 'ROC-AUC'},
        ax=ax2,
        linewidths=0.5,
        linecolor='gray'
    )
    for i, relation in enumerate(relations):
        best_layer_idx = np.argmax(data_auc[:, i])
        ax2.add_patch(plt.Rectangle((i, best_layer_idx), 1, 1, 
                                    fill=False, edgecolor='blue', lw=2))
    ax2.set_title('Per-Relation ROC-AUC', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Relation', fontsize=9)
    ax2.set_ylabel('Layer', fontsize=9)
    
    # 3. Per-Relation Accuracy - vertical
    ax3 = fig.add_subplot(gs[1, 2])
    data_acc = np.zeros((num_layers, len(relations)))
    for i, relation in enumerate(relations):
        for layer_metrics in relation_metrics[relation]:
            layer_idx = layer_metrics["layer"]
            data_acc[layer_idx, i] = layer_metrics["accuracy"]
    
    sns.heatmap(
        data_acc,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=max(0.5, data_acc.min() - 0.05),
        vmax=min(1.0, data_acc.max() + 0.05),
        xticklabels=relations,
        yticklabels=range(num_layers),
        cbar_kws={'label': 'Accuracy'},
        ax=ax3,
        linewidths=0.5,
        linecolor='gray'
    )
    for i, relation in enumerate(relations):
        best_layer_idx = np.argmax(data_acc[:, i])
        ax3.add_patch(plt.Rectangle((i, best_layer_idx), 1, 1, 
                                    fill=False, edgecolor='blue', lw=2))
    ax3.set_title('Per-Relation Accuracy', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Relation', fontsize=9)
    ax3.set_ylabel('Layer', fontsize=9)
    
    fig.suptitle('Linear Probe Performance Analysis', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / "probe_performance_heatmaps_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined heatmaps to: {output_path}")
    plt.close()


def save_per_relation_metrics(
    relation_metrics: Dict[str, List[Dict]],
    output_path: Path
):
    """Save per-relation metrics to JSON."""
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(relation_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved per-relation metrics to: {output_path}")


def main():
    """
    Main function to plot probe performance heatmaps.
    """
    # Configuration
    models_dir = Path("./models/probes")
    probe_data_dir = Path("./data/probe_training_data")
    relations = ["P40", "P50", "P176", "P264"]
    output_dir = Path("./plots/probe_performance")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("Probe Performance Visualization")
    print("="*70)
    
    # Load overall metrics from training
    metrics_path = models_dir / "all_layers_metrics.json"
    if not metrics_path.exists():
        print(f"\nError: {metrics_path} not found!")
        print("Please run train_linear_probe.py first.")
        return
    
    print(f"\nLoading overall metrics from: {metrics_path}")
    overall_data = load_json(metrics_path)
    all_metrics = overall_data["all_metrics"]
    best_layer = overall_data["best_layer"]
    
    print(f"Best layer (overall): {best_layer}")
    print(f"Number of layers: {len(all_metrics)}")
    
    # Plot overall heatmap
    print("\n" + "="*70)
    print("Plotting Overall Performance")
    print("="*70)
    overall_heatmap_path = output_dir / "probe_performance_overall.png"
    plot_overall_heatmap(all_metrics, overall_heatmap_path)
    
    # Compute and plot per-relation metrics
    print("\n" + "="*70)
    print("Computing and Plotting Per-Relation Performance")
    print("="*70)
    
    relation_metrics = compute_per_relation_metrics(
        probe_data_dir,
        models_dir,
        relations,
        split="dev"
    )
    
    # Save per-relation metrics
    per_relation_metrics_path = models_dir / "per_relation_metrics.json"
    save_per_relation_metrics(relation_metrics, per_relation_metrics_path)
    
    # Plot individual heatmaps
    print("\n" + "="*70)
    print("Creating Heatmaps")
    print("="*70)
    
    # AUC heatmap
    auc_heatmap_path = output_dir / "probe_auc_by_relation_layer.png"
    plot_per_relation_heatmap(
        relation_metrics,
        "auc",
        auc_heatmap_path,
        vmin=0.5,
        vmax=1.0
    )
    
    # Accuracy heatmap
    acc_heatmap_path = output_dir / "probe_accuracy_by_relation_layer.png"
    plot_per_relation_heatmap(
        relation_metrics,
        "accuracy",
        acc_heatmap_path,
        vmin=0.5,
        vmax=1.0
    )
    
    # Combined heatmaps
    plot_combined_heatmaps(all_metrics, relation_metrics, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("Summary: Best Layers by Relation")
    print("="*70)
    print(f"{'Relation':<12} {'Best Layer':<12} {'AUC':<12} {'Accuracy':<12}")
    print("-"*70)
    for relation in sorted(relations):
        best = max(relation_metrics[relation], key=lambda x: x["auc"])
        print(f"{relation:<12} {best['layer']:<12} "
              f"{best['auc']:<12.4f} {best['accuracy']:<12.4f}")
    
    print("\n" + "="*70)
    print("Overall Best Layer (from training)")
    print("="*70)
    print(f"Layer: {best_layer}")
    print(f"Dev AUC: {overall_data['best_metrics']['dev_auc']:.4f}")
    print(f"Dev Accuracy: {overall_data['best_metrics']['dev_accuracy']:.4f}")
    
    print("\n" + "="*70)
    print("Visualization Complete!")
    print("="*70)
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()

