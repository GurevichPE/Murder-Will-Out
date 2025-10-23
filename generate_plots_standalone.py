#!/usr/bin/env python3
"""
Standalone plot generation script for linear probe results.

This script allows you to regenerate all plots using previously saved plotting data,
without needing to retrain the probes.

Usage:
    python generate_plots_standalone.py [--plotting-data PATH] [--output-dir PATH]
"""

import argparse
from pathlib import Path
from train_linear_probe import generate_plots_from_saved_data


def main():
    """Generate plots from saved plotting data."""
    parser = argparse.ArgumentParser(
        description="Generate plots from saved linear probe training data"
    )
    parser.add_argument(
        "--plotting-data",
        type=Path,
        default=Path("./models/probes/plotting_data.json"),
        help="Path to saved plotting data JSON file (default: ./models/probes/plotting_data.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models/probes/plots"),
        help="Directory to save generated plots (default: ./models/probes/plots)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Standalone Plot Generation for Linear Probe Results")
    print("=" * 70)
    print(f"Plotting data: {args.plotting_data}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        generate_plots_from_saved_data(args.plotting_data, args.output_dir)
        print("\n" + "=" * 70)
        print("Plot generation completed successfully!")
        print("=" * 70)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run train_linear_probe.py first to generate the plotting data.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
