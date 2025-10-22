"""
Calculate K and K* for all dev relations and create comparison bar plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from knowledge_calculation import load_json, calculate_knowledge_for_dataset


def calculate_metrics_for_all_relations(data_dir: Path) -> dict:
    """
    Calculate K and K* metrics for all relation files in the directory.
    
    Args:
        data_dir: Path to directory containing relation JSON files
        
    Returns:
        Dictionary with relation codes as keys and metrics as values
    """
    relations = ["P40", "P50", "P176", "P264"]
    results = {}
    
    for relation in relations:
        file_path = data_dir / f"{relation}.test.json"
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Processing {relation}...")
        data = load_json(file_path)
        metrics = calculate_knowledge_for_dataset(data, score_key="p_true")
        
        results[relation] = {
            "mean_k": metrics["mean_k"],
            "mean_k_star": metrics["mean_k_star"],
            "count": metrics["count"],
            "total": len(data)
        }
        
        print(f"  Mean K: {metrics['mean_k']:.4f}, Mean K*: {metrics['mean_k_star']:.4f}, Valid: {metrics['count']}/{len(data)}")
    
    return results


def plot_knowledge_comparison(results: dict, output_path: str = "test_knowledge_comparison.png"):
    """
    Create bar plots comparing K and K* across relations.
    
    Args:
        results: Dictionary with metrics for each relation
        output_path: Path to save the plot
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Prepare data
    relations = list(results.keys())
    k_values = [results[r]["mean_k"] for r in relations]
    k_star_values = [results[r]["mean_k_star"] for r in relations]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot K
    colors = sns.color_palette("husl", len(relations))
    bars1 = ax1.bar(relations, k_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Mean K', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax1.set_title('Knowledge per Question (K)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, k_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot K*
    bars2 = ax2.bar(relations, k_star_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Mean K*', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax2.set_title('Perfect Knowledge (K*)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, k_star_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add relation descriptions
    relation_names = {
        "P40": "P40 (child)",
        "P50": "P50 (author)",
        "P176": "P176 (manufacturer)",
        "P264": "P264 (record label)"
    }
    ax1.set_xticks(range(len(relations)))
    ax1.set_xticklabels([relation_names.get(r, r) for r in relations], rotation=15, ha='right')
    ax2.set_xticks(range(len(relations)))
    ax2.set_xticklabels([relation_names.get(r, r) for r in relations], rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def plot_combined_comparison(results: dict, output_path: str = "test_knowledge_combined.png"):
    """
    Create a grouped bar plot comparing K and K* side by side.
    
    Args:
        results: Dictionary with metrics for each relation
        output_path: Path to save the plot
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Prepare data
    relations = list(results.keys())
    k_values = [results[r]["mean_k"] for r in relations]
    k_star_values = [results[r]["mean_k_star"] for r in relations]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set positions for bars
    x = np.arange(len(relations))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, k_values, width, label='K (Knowledge)', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, k_star_values, width, label='K* (Perfect Knowledge)', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Customize plot
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax.set_title('Knowledge Metrics Comparison (Dev Set, p_true scoring)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    
    # Relation names
    relation_names = {
        "P40": "P40\n(child)",
        "P50": "P50\n(author)",
        "P176": "P176\n(manufacturer)",
        "P264": "P264\n(record label)"
    }
    ax.set_xticks(x)
    ax.set_xticklabels([relation_names.get(r, r) for r in relations])
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def print_summary_table(results: dict):
    """
    Print a formatted summary table of the results.
    
    Args:
        results: Dictionary with metrics for each relation
    """
    print("\n" + "="*70)
    print("Knowledge Metrics Summary (Dev Set, p_true scoring)")
    print("="*70)
    print(f"{'Relation':<20} {'Mean K':>10} {'Mean K*':>10} {'Valid':>10} {'Total':>10}")
    print("-"*70)
    
    for relation, metrics in results.items():
        print(f"{relation:<20} {metrics['mean_k']:>10.4f} {metrics['mean_k_star']:>10.4f} "
              f"{metrics['count']:>10} {metrics['total']:>10}")
    
    print("-"*70)
    
    # Calculate overall averages
    avg_k = np.mean([m["mean_k"] for m in results.values()])
    avg_k_star = np.mean([m["mean_k_star"] for m in results.values()])
    total_valid = sum([m["count"] for m in results.values()])
    total_items = sum([m["total"] for m in results.values()])
    
    print(f"{'Average':<20} {avg_k:>10.4f} {avg_k_star:>10.4f} "
          f"{total_valid:>10} {total_items:>10}")
    print("="*70)


def main():
    """
    Main function to calculate metrics and create plots.
    """
    data_dir = Path("./data/sampled_labeled_answers_1000_temp1/test")
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return
    
    print("Calculating knowledge metrics for all relations...\n")
    results = calculate_metrics_for_all_relations(data_dir)
    
    if not results:
        print("No results calculated. Please check your data files.")
        return
    
    # Print summary table
    print_summary_table(results)
    
    # Create plots
    print("\nCreating plots...")
    plot_knowledge_comparison(results, "dev_knowledge_comparison.png")
    plot_combined_comparison(results, "dev_knowledge_combined.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

