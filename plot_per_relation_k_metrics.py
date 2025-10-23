#!/usr/bin/env python3
"""
Plot per-relation K and K* metrics for the best layer.

This script loads the trained probe for the best layer and evaluates K and K* metrics
separately for each relation, then creates visualizations.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from train_linear_probe import (
    load_probe_training_data, 
    calculate_k_metrics_for_dev,
    save_json
)


def load_trained_probe(probe_path: Path, scaler_path: Path) -> Tuple:
    """
    Load trained probe and scaler from disk.
    
    Args:
        probe_path: Path to saved probe pickle file
        scaler_path: Path to saved scaler pickle file
        
    Returns:
        Tuple of (probe, scaler)
    """
    with open(probe_path, 'rb') as f:
        probe = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return probe, scaler


def calculate_per_relation_k_metrics(
    relations: List[str],
    layer_idx: int,
    probe,
    scaler,
    data_dir: Path
) -> Dict[str, Dict[str, float]]:
    """
    Calculate K and K* metrics for each relation separately.
    
    Args:
        relations: List of relation codes (e.g., ["P40", "P50"])
        layer_idx: Layer index to use
        probe: Trained probe
        scaler: Fitted scaler
        data_dir: Directory containing probe data files
        
    Returns:
        Dict mapping relation -> {"k": float, "k_star": float}
    """
    per_relation_metrics = {}
    
    for relation in relations:
        print(f"Processing relation {relation}...")
        
        # Load dev data for this relation only
        dev_hidden_states, dev_labels, dev_metadata = load_probe_training_data(
            data_dir, [relation], split="dev"
        )
        
        # Extract hidden states for the specified layer
        layer_hidden_states = dev_hidden_states[layer_idx]
        
        # Calculate K metrics for this relation
        k, k_star = calculate_k_metrics_for_dev(
            layer_hidden_states, dev_labels, dev_metadata, probe, scaler
        )
        
        per_relation_metrics[relation] = {
            "k": k,
            "k_star": k_star,
            "num_examples": len(dev_labels)
        }
        
        print(f"  {relation}: K={k:.4f}, K*={k_star:.4f} ({len(dev_labels)} examples)")
    
    return per_relation_metrics


def plot_per_relation_k_metrics(
    per_relation_metrics: Dict[str, Dict[str, float]],
    layer_idx: int,
    output_dir: Path
) -> None:
    """
    Create visualizations for per-relation K and K* metrics.
    
    Args:
        per_relation_metrics: Dict mapping relation -> metrics
        layer_idx: Layer index used
        output_dir: Directory to save plots
    """
    relations = list(per_relation_metrics.keys())
    k_values = [per_relation_metrics[rel]["k"] for rel in relations]
    k_star_values = [per_relation_metrics[rel]["k_star"] for rel in relations]
    num_examples = [per_relation_metrics[rel]["num_examples"] for rel in relations]
    
    # Set up the plotting style
    plt.style.use('default')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Standard matplotlib colors
    
    # Plot 1: Bar chart comparing K and K* per relation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(relations))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, k_values, width, label='K (Knowledge)', 
                    color=colors[0], alpha=0.8)
    bars2 = ax1.bar(x + width/2, k_star_values, width, label='K* (Perfect Knowledge)', 
                    color=colors[1], alpha=0.8)
    
    ax1.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Knowledge Metric Value', fontsize=12, fontweight='bold')
    ax1.set_title(f'K and K* Metrics by Relation (Layer {layer_idx})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(relations)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Number of examples per relation
    bars3 = ax2.bar(relations, num_examples, color=colors[2], alpha=0.8)
    ax2.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Dev Examples', fontsize=12, fontweight='bold')
    ax2.set_title('Dev Set Size by Relation', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    bar_chart_path = output_dir / f"k_metrics_per_relation_layer_{layer_idx}_bar.png"
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to: {bar_chart_path}")
    plt.close()
    
    # Plot 3: Scatter plot of K vs K* with relation labels
    plt.figure(figsize=(10, 8))
    
    for i, (rel, k_val, k_star_val) in enumerate(zip(relations, k_values, k_star_values)):
        plt.scatter(k_val, k_star_val, s=150, color=colors[i], alpha=0.8, 
                   label=f'{rel} (n={num_examples[i]})')
        
        # Add relation label next to point
        plt.annotate(rel, (k_val, k_star_val), xytext=(5, 5), 
                    textcoords='offset points', fontsize=11, fontweight='bold')
    
    # Add diagonal line (K = K*)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='K = K*')
    
    plt.xlabel('K (Knowledge)', fontsize=12, fontweight='bold')
    plt.ylabel('K* (Perfect Knowledge)', fontsize=12, fontweight='bold')
    plt.title(f'K vs K* by Relation (Layer {layer_idx})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add text box with interpretation
    textstr = '\n'.join([
        'Interpretation:',
        '• Points on diagonal: K = K*',
        '• Points below diagonal: K > K*',
        '• Higher values = better knowledge'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    scatter_path = output_dir / f"k_vs_k_star_per_relation_layer_{layer_idx}.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {scatter_path}")
    plt.close()
    
    # Plot 4: Horizontal bar chart for better readability with many relations
    plt.figure(figsize=(10, 6))
    
    y_pos = np.arange(len(relations))
    
    plt.barh(y_pos - 0.2, k_values, 0.4, label='K (Knowledge)', 
             color=colors[0], alpha=0.8)
    plt.barh(y_pos + 0.2, k_star_values, 0.4, label='K* (Perfect Knowledge)', 
             color=colors[1], alpha=0.8)
    
    plt.yticks(y_pos, relations)
    plt.xlabel('Knowledge Metric Value', fontsize=12, fontweight='bold')
    plt.ylabel('Relation', fontsize=12, fontweight='bold')
    plt.title(f'K and K* Metrics by Relation (Layer {layer_idx})', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, axis='x')
    plt.xlim(0, 1)
    
    # Add value labels
    for i, (k_val, k_star_val) in enumerate(zip(k_values, k_star_values)):
        plt.text(k_val + 0.01, i - 0.2, f'{k_val:.3f}', va='center', fontsize=9)
        plt.text(k_star_val + 0.01, i + 0.2, f'{k_star_val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    horizontal_bar_path = output_dir / f"k_metrics_per_relation_layer_{layer_idx}_horizontal.png"
    plt.savefig(horizontal_bar_path, dpi=300, bbox_inches='tight')
    print(f"Horizontal bar chart saved to: {horizontal_bar_path}")
    plt.close()


def main():
    """
    Main function to calculate and plot per-relation K metrics for the best layer.
    """
    # Configuration
    best_layer = 14  # The best layer as mentioned by the user
    probe_data_dir = Path("./data/probe_training_data")
    models_dir = Path("./models/probes")
    output_dir = Path("./models/probes/per_relation_plots")
    relations = ["P40", "P50", "P176", "P264"]
    
    print("=" * 70)
    print("Per-Relation K and K* Metrics Analysis")
    print("=" * 70)
    print(f"Best layer: {best_layer}")
    print(f"Relations: {relations}")
    print()
    
    # Load trained probe and scaler for the best layer
    probe_path = models_dir / f"probe_layer_{best_layer}.pkl"
    scaler_path = models_dir / f"scaler_layer_{best_layer}.pkl"
    
    if not probe_path.exists() or not scaler_path.exists():
        print(f"Error: Probe or scaler not found for layer {best_layer}")
        print(f"Expected files:")
        print(f"  {probe_path}")
        print(f"  {scaler_path}")
        print("Make sure you have run train_linear_probe.py first.")
        return
    
    print("Loading trained probe and scaler...")
    probe, scaler = load_trained_probe(probe_path, scaler_path)
    print(f"Loaded probe and scaler for layer {best_layer}")
    
    # Calculate per-relation K metrics
    print("\n" + "=" * 70)
    print("Calculating K and K* Metrics per Relation")
    print("=" * 70)
    
    per_relation_metrics = calculate_per_relation_k_metrics(
        relations, best_layer, probe, scaler, probe_data_dir
    )
    
    # Save metrics to JSON
    output_dir.mkdir(exist_ok=True, parents=True)
    metrics_file = output_dir / f"per_relation_k_metrics_layer_{best_layer}.json"
    
    # Format metrics for saving
    metrics_to_save = {
        "layer": best_layer,
        "relations": per_relation_metrics
    }
    
    save_json(metrics_to_save, metrics_file)
    print(f"\nPer-relation metrics saved to: {metrics_file}")
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)
    
    plot_per_relation_k_metrics(per_relation_metrics, best_layer, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"{'Relation':<10} {'K':<8} {'K*':<8} {'Examples':<10}")
    print("-" * 40)
    for rel in relations:
        metrics = per_relation_metrics[rel]
        print(f"{rel:<10} {metrics['k']:<8.4f} {metrics['k_star']:<8.4f} {metrics['num_examples']:<10}")
    
    # Calculate averages
    avg_k = np.mean([per_relation_metrics[rel]['k'] for rel in relations])
    avg_k_star = np.mean([per_relation_metrics[rel]['k_star'] for rel in relations])
    total_examples = sum([per_relation_metrics[rel]['num_examples'] for rel in relations])
    
    print("-" * 40)
    print(f"{'Average':<10} {avg_k:<8.4f} {avg_k_star:<8.4f} {total_examples:<10}")
    
    print(f"\nGenerated plots:")
    print(f"  - k_metrics_per_relation_layer_{best_layer}_bar.png")
    print(f"  - k_vs_k_star_per_relation_layer_{best_layer}.png")
    print(f"  - k_metrics_per_relation_layer_{best_layer}_horizontal.png")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
