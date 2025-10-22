#!/usr/bin/env python3
"""
Test script for external_scoring.py to verify it works with sample data.
"""

import json
import sys
from pathlib import Path

# Add the current directory to the Python path so we can import external_scoring
sys.path.insert(0, str(Path(__file__).parent))

from external_scoring import (
    initialize_model,
    score_single_entry,
    score_single_entry_optimized,
    score_batch_entries,
    save_json_data,
    load_json_data
)


def create_sample_data():
    """Create a small sample dataset for testing."""
    sample_data = [
        {
            "question": "What is the capital of France?",
            "golden_answer": "Paris",
            "answer_labels": [
                ["Paris", "EXACT_MATCH"],
                ["London", "False"],
                ["Berlin", "False"]
            ]
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "golden_answer": "William Shakespeare",
            "answer_labels": [
                ["William Shakespeare", "EXACT_MATCH"],
                ["Charles Dickens", "False"],
                ["Jane Austen", "False"]
            ]
        }
    ]
    return sample_data


def test_single_entry_scoring():
    """Test scoring a single entry."""
    print("Testing single entry scoring...")

    # Initialize model (this will take some time on first run)
    try:
        initialize_model()
        print("Model initialized successfully")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        print("This is expected if running without proper setup")
        return False

    # Create sample data
    sample_entry = create_sample_data()[0]

    # Test scoring
    try:
        scored_entry = score_single_entry_optimized(sample_entry, batch_size=10)
        print("Single entry scoring successful!")
        print(f"Golden answer scores: {scored_entry['golden_external_scores']}")
        print(f"Number of scored answers: {len(scored_entry['scored_answers'])}")
        return True
    except Exception as e:
        print(f"Single entry scoring failed: {e}")
        return False


def test_batch_scoring():
    """Test batch scoring."""
    print("Testing batch scoring...")

    # Skip if model not available
    try:
        initialize_model()
    except Exception as e:
        print(f"Skipping batch test due to model initialization error: {e}")
        return True

    # Create sample data
    sample_data = create_sample_data()

    # Test batch scoring
    try:
        scored_data = score_batch_entries(sample_data, batch_size=2, inner_batch_size=10)
        print("Batch scoring successful!")
        print(f"Scored {len(scored_data)} entries")
        return True
    except Exception as e:
        print(f"Batch scoring failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running external scoring tests...")

    # Test single entry scoring
    single_test_passed = test_single_entry_scoring()

    # Test batch scoring
    batch_test_passed = test_batch_scoring()

    if single_test_passed and batch_test_passed:
        print("\nAll tests passed!")
        return True
    else:
        print("\nSome tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
