"""
Compare two models (p_true vs p_true_honest) across all relations and create comparison plots.

This script compares knowledge metrics between two models:
- Model 1: Uses "p_true" scoring 
- Model 2: Uses "p_true_honest" scoring

Usage:
    python plot_dev_knowledge.py [dev|test]
    
Examples:
    python plot_dev_knowledge.py dev   # Process dev split
    python plot_dev_knowledge.py test  # Process test split  
    python plot_dev_knowledge.py       # Default: test split

Features:
- Compares two models side by side for K and K* metrics
- Automatic answer deduplication for correct K/K* calculations
- Dynamic plot titles and filenames based on split
- Comprehensive summary statistics for both models
- Multiple plot formats showing model comparisons
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys
from knowledge_calculation import load_json, calculate_knowledge_for_dataset


def calculate_metrics_for_all_relations(data_dir: Path, split: str = "test") -> dict:
    """
    Calculate K and K* metrics for both models (p_true and p_true_honest) across all relations.
    
    Args:
        data_dir: Path to directory containing relation JSON files
        split: Data split to use ("dev" or "test")
        
    Returns:
        Dictionary with relation codes as keys and nested model metrics as values
    """
    relations = ["P40", "P50", "P176", "P264"]
    results = {}
    
    for relation in relations:
        file_path = data_dir / f"{relation}.{split}.json"
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Processing {relation}...")
        data = load_json(file_path)
        
        # Calculate metrics for both models
        metrics_p_true = calculate_knowledge_for_dataset(data, score_key="p_true")
        metrics_p_true_honest = calculate_knowledge_for_dataset(data, score_key="p_true_honest")
        
        results[relation] = {
            "p_true": {
                "mean_k": metrics_p_true["mean_k"],
                "mean_k_star": metrics_p_true["mean_k_star"],
                "count": metrics_p_true["count"]
            },
            "p_true_honest": {
                "mean_k": metrics_p_true_honest["mean_k"],
                "mean_k_star": metrics_p_true_honest["mean_k_star"],
                "count": metrics_p_true_honest["count"]
            },
            "total": len(data)
        }
        
        print(f"  p_true:        K={metrics_p_true['mean_k']:.4f}, K*={metrics_p_true['mean_k_star']:.4f}, Valid={metrics_p_true['count']}/{len(data)}")
        print(f"  p_true_honest: K={metrics_p_true_honest['mean_k']:.4f}, K*={metrics_p_true_honest['mean_k_star']:.4f}, Valid={metrics_p_true_honest['count']}/{len(data)}")
    
    return results


def plot_knowledge_comparison(results: dict, split: str = "test", output_path: str = None):
    """
    Create bar plots comparing p_true and p_true_honest models across relations.
    
    Args:
        results: Dictionary with metrics for each relation and model
        split: Data split being used ("dev" or "test")
        output_path: Path to save the plot (auto-generated if None)
    """
    if output_path is None:
        output_path = f"{split}_model_comparison.png"
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Prepare data
    relations = list(results.keys())
    p_true_k = [results[r]["p_true"]["mean_k"] for r in relations]
    p_true_honest_k = [results[r]["p_true_honest"]["mean_k"] for r in relations]
    p_true_k_star = [results[r]["p_true"]["mean_k_star"] for r in relations]
    p_true_honest_k_star = [results[r]["p_true_honest"]["mean_k_star"] for r in relations]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set positions for grouped bars
    x = np.arange(len(relations))
    width = 0.35
    
    # Plot K comparison
    bars1 = ax1.bar(x - width/2, p_true_k, width, label='p_true', 
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, p_true_honest_k, width, label='p_true_honest', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax1.set_ylabel('Mean K', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax1.set_title(f'Knowledge (K) Model Comparison - {split.upper()} Set', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # Add value labels on K bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot K* comparison
    bars3 = ax2.bar(x - width/2, p_true_k_star, width, label='p_true', 
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars4 = ax2.bar(x + width/2, p_true_honest_k_star, width, label='p_true_honest', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax2.set_ylabel('Mean K*', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax2.set_title(f'Perfect Knowledge (K*) Model Comparison - {split.upper()} Set', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    # Add value labels on K* bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add relation descriptions
    relation_names = {
        "P40": "P40\n(child)",
        "P50": "P50\n(author)",
        "P176": "P176\n(manufacturer)",
        "P264": "P264\n(record label)"
    }
    ax1.set_xticks(x)
    ax1.set_xticklabels([relation_names.get(r, r) for r in relations])
    ax2.set_xticks(x)
    ax2.set_xticklabels([relation_names.get(r, r) for r in relations])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def plot_combined_comparison(results: dict, split: str = "test", output_path: str = None):
    """
    Create a comprehensive grouped bar plot comparing both models' K and K* metrics.
    
    Args:
        results: Dictionary with metrics for each relation and model
        split: Data split being used ("dev" or "test")
        output_path: Path to save the plot (auto-generated if None)
    """
    if output_path is None:
        output_path = f"{split}_combined_model_comparison.png"
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Prepare data
    relations = list(results.keys())
    p_true_k = [results[r]["p_true"]["mean_k"] for r in relations]
    p_true_honest_k = [results[r]["p_true_honest"]["mean_k"] for r in relations]
    p_true_k_star = [results[r]["p_true"]["mean_k_star"] for r in relations]
    p_true_honest_k_star = [results[r]["p_true_honest"]["mean_k_star"] for r in relations]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set positions for bars (4 bars per relation)
    x = np.arange(len(relations))
    width = 0.2
    
    # Create bars for all combinations
    bars1 = ax.bar(x - 1.5*width, p_true_k, width, label='p_true (K)', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x - 0.5*width, p_true_k_star, width, label='p_true (K*)', 
                   color='#2980b9', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + 0.5*width, p_true_honest_k, width, label='p_true_honest (K)', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars4 = ax.bar(x + 1.5*width, p_true_honest_k_star, width, label='p_true_honest (K*)', 
                   color='#c0392b', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Customize plot
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax.set_title(f'Complete Model Comparison: K and K* Metrics ({split.upper()} Set)', 
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
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def print_summary_table(results: dict, split: str = "test"):
    """
    Print a formatted summary table comparing both models.
    
    Args:
        results: Dictionary with metrics for each relation and model
        split: Data split being used ("dev" or "test")
    """
    print("\n" + "="*95)
    print(f"Knowledge Metrics Summary - Model Comparison ({split.upper()} Set)")
    print("="*95)
    print(f"{'Relation':<12} {'p_true K':<10} {'p_true K*':<11} {'p_true_honest K':<16} {'p_true_honest K*':<17} {'Valid':<8} {'Total':<8}")
    print("-"*95)
    
    for relation, metrics in results.items():
        print(f"{relation:<12} "
              f"{metrics['p_true']['mean_k']:>9.4f} "
              f"{metrics['p_true']['mean_k_star']:>10.4f} "
              f"{metrics['p_true_honest']['mean_k']:>15.4f} "
              f"{metrics['p_true_honest']['mean_k_star']:>16.4f} "
              f"{metrics['p_true']['count']:>7} "
              f"{metrics['total']:>7}")
    
    print("-"*95)
    
    # Calculate overall averages
    avg_p_true_k = np.mean([m["p_true"]["mean_k"] for m in results.values()])
    avg_p_true_k_star = np.mean([m["p_true"]["mean_k_star"] for m in results.values()])
    avg_p_true_honest_k = np.mean([m["p_true_honest"]["mean_k"] for m in results.values()])
    avg_p_true_honest_k_star = np.mean([m["p_true_honest"]["mean_k_star"] for m in results.values()])
    total_valid = sum([m["p_true"]["count"] for m in results.values()])  # Should be same for both models
    total_items = sum([m["total"] for m in results.values()])
    
    print(f"{'Average':<12} "
          f"{avg_p_true_k:>9.4f} "
          f"{avg_p_true_k_star:>10.4f} "
          f"{avg_p_true_honest_k:>15.4f} "
          f"{avg_p_true_honest_k_star:>16.4f} "
          f"{total_valid:>7} "
          f"{total_items:>7}")
    print("="*95)
    
    # Print model comparison insights
    print(f"\nModel Comparison Insights:")
    print(f"- Average K improvement (p_true_honest vs p_true): {avg_p_true_honest_k - avg_p_true_k:+.4f}")
    print(f"- Average K* improvement (p_true_honest vs p_true): {avg_p_true_honest_k_star - avg_p_true_k_star:+.4f}")
    print("="*95)


def main(split: str = None):
    """
    Main function to calculate model comparison metrics and create comparison plots.
    
    Compares p_true and p_true_honest models across all relations for both K and K* metrics.
    
    Args:
        split: Data split to use ("dev" or "test"). If None, taken from command line args.
    """
    # Parse command-line arguments for split
    if split is None:
        if len(sys.argv) > 1:
            split = sys.argv[1].lower()
            if split not in ["dev", "test"]:
                print(f"Error: Invalid split '{split}'. Use 'dev' or 'test'.")
                print("Usage: python plot_dev_knowledge.py [dev|test]")
                return
        else:
            split = "test"  # Default to test
    
    print(f"Processing {split.upper()} split...")
    
    # Set up data directory 
    base_dir = Path("./data/sampled_labeled_answers_1000_temp1")
    data_dir = base_dir / split
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        print(f"Available directories in {base_dir}:")
        if base_dir.exists():
            for item in base_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item.name}")
        return
    
    print("Calculating knowledge metrics for both models (p_true vs p_true_honest) across all relations...\n")
    print("Note: Using answer deduplication for accurate K/K* calculations")
    results = calculate_metrics_for_all_relations(data_dir, split)
    
    if not results:
        print("No results calculated. Please check your data files.")
        return
    
    # Print summary table
    print_summary_table(results, split)
    
    # Create plots
    print("\nCreating model comparison plots...")
    plot_knowledge_comparison(results, split)
    plot_combined_comparison(results, split)
    
    print(f"\nDone! Model comparison plots saved with '{split}_' prefix.")
    print("Generated plots:")
    print(f"  - {split}_model_comparison.png (separate K and K* comparisons)")
    print(f"  - {split}_combined_model_comparison.png (comprehensive comparison)")


if __name__ == "__main__":
    main()

