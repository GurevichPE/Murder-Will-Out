"""
Visualize hidden knowledge: comparison between internal and external knowledge.

Creates plots showing:
1. K comparison (internal vs external)
2. K* comparison (internal vs external)
3. Hidden knowledge gap across relations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict

from generate_data_for_probe import load_json


def plot_knowledge_comparison(
    metrics: List[Dict],
    output_path: Path
):
    """
    Create comprehensive comparison plots.
    
    Args:
        metrics: List of metrics dicts for each relation
        output_path: Path to save the plot
    """
    relations = [m["relation"] for m in metrics]
    
    # Extract data
    ext_k = [m["external"]["mean_k"] for m in metrics]
    int_k = [m["internal"]["mean_k"] for m in metrics]
    ext_k_star = [m["external"]["mean_k_star"] for m in metrics]
    int_k_star = [m["internal"]["mean_k_star"] for m in metrics]
    k_gaps = [m["gaps"]["k_relative"] * 100 for m in metrics]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Relation name mapping
    relation_names = {
        "P40": "P40\n(child)",
        "P50": "P50\n(author)",
        "P176": "P176\n(manufacturer)",
        "P264": "P264\n(record label)"
    }
    
    x_labels = [relation_names.get(r, r) for r in relations]
    x = np.arange(len(relations))
    width = 0.35
    
    # Plot 1: K comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, ext_k, width, label='External (p_true)', 
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, int_k, width, label='Internal (probe)', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax1.set_ylabel('Mean K', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax1.set_title('Knowledge Score (K) Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: K* comparison
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, ext_k_star, width, label='External (p_true)', 
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars4 = ax2.bar(x + width/2, int_k_star, width, label='Internal (probe)', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax2.set_ylabel('Mean K*', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax2.set_title('Perfect Knowledge (K*) Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Hidden knowledge gap
    ax3 = axes[2]
    colors = ['#2ecc71' if gap > 0 else '#e67e22' for gap in k_gaps]
    bars5 = ax3.bar(x, k_gaps, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax3.set_ylabel('Hidden Knowledge Gap (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax3.set_title('Hidden Knowledge Gap\n(Relative Improvement)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, gap in zip(bars5, k_gaps):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{gap:.1f}%', ha='center', 
                va='bottom' if gap > 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_gap_distribution(
    metrics: List[Dict],
    output_path: Path
):
    """
    Create a simple bar plot showing just the gaps.
    
    Args:
        metrics: List of metrics dicts for each relation
        output_path: Path to save the plot
    """
    relations = [m["relation"] for m in metrics]
    k_gaps = [m["gaps"]["k_relative"] * 100 for m in metrics]
    
    # Calculate average
    avg_gap = np.mean(k_gaps)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Relation name mapping
    relation_names = {
        "P40": "P40 (child)",
        "P50": "P50 (author)",
        "P176": "P176 (manufacturer)",
        "P264": "P264 (record label)"
    }
    
    x_labels = [relation_names.get(r, r) for r in relations]
    x = np.arange(len(relations))
    
    # Plot bars
    colors = ['#2ecc71' if gap > 0 else '#e67e22' for gap in k_gaps]
    bars = ax.bar(x, k_gaps, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add average line
    ax.axhline(y=avg_gap, color='red', linestyle='--', linewidth=2, 
               label=f'Average: {avg_gap:.1f}%', alpha=0.7)
    
    # Styling
    ax.set_ylabel('Hidden Knowledge Gap (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Relation', fontsize=14, fontweight='bold')
    ax.set_title('Hidden Knowledge: Internal vs External Scoring\n'
                 '(Relative Improvement of Probe over P_true)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    # Add value labels
    for bar, gap in zip(bars, k_gaps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{gap:.1f}%', ha='center', 
                va='bottom' if gap > 0 else 'top', 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gap distribution plot saved to: {output_path}")
    plt.close()


def main():
    """
    Main function to create visualizations.
    """
    # Load metrics
    metrics_file = Path("./data/scored_with_probe/dev/knowledge_comparison_metrics.json")
    
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        print("Please run score_dev_with_probe.py first!")
        return
    
    print("Loading metrics...")
    data = load_json(metrics_file)
    metrics = data["relations"]
    summary = data["summary"]
    
    print(f"Loaded metrics for {len(metrics)} relations")
    print(f"Average hidden knowledge gap: {summary['avg_k_gap_relative']*100:.1f}%")
    
    # Create plots
    print("\nCreating plots...")
    
    # Comprehensive comparison plot
    plot_knowledge_comparison(
        metrics,
        Path("./hidden_knowledge_comparison.png")
    )
    
    # Gap distribution plot
    plot_gap_distribution(
        metrics,
        Path("./hidden_knowledge_gap.png")
    )
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

