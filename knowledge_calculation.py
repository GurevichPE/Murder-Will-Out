"""
Knowledge calculation module based on the paper "Hidden Knowledge in LLMs".

This module implements the formulas for calculating knowledge metrics:
- K_q: Knowledge per question
- K: Overall knowledge degree
- K*: Perfect knowledge indicator

IMPORTANT: This module implements proper answer deduplication for K calculations.
For correct K/K* metrics, use calculate_k_q_with_answers() which:
1. Deduplicates answers (e.g., "Paris" appearing multiple times = 1 unique answer)
2. Aggregates scores for duplicates (default: max score)
3. Then calculates K on the unique answer sets

References:
    Paper: arxiv 2503.15299v4
    Equation (1): K_q formula
    Equation (2): K formula
    Equation (3): K* formula
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Union, Callable, Optional
import numpy as np


def deduplicate_answers_with_scores(
    answers: List[str],
    scores: Union[List[float], np.ndarray],
    labels: Union[List[str], List[bool], np.ndarray],
    aggregation_method: str = "max"
) -> Tuple[List[str], List[float], List]:
    """
    Deduplicate answers and aggregate scores for unique answers.
    
    This is crucial for proper K calculation - we need to work with unique answer sets,
    not individual answer instances. If "Paris" appears multiple times, we should 
    treat it as one unique answer with aggregated score.
    
    Args:
        answers: List of answer strings (may contain duplicates)
        scores: List/array of scores corresponding to each answer
        labels: List/array of labels corresponding to each answer 
        aggregation_method: How to aggregate scores for duplicate answers ("max", "mean")
        
    Returns:
        Tuple of (unique_answers, aggregated_scores, aggregated_labels)
    """
    if len(answers) != len(scores) or len(answers) != len(labels):
        raise ValueError("answers, scores, and labels must have the same length")
    
    # Convert to numpy for easier handling
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Group by unique answers
    answer_groups = {}
    for i, answer in enumerate(answers):
        # Normalize answer for comparison (strip whitespace, lowercase)
        normalized_answer = answer.strip().lower()
        
        if normalized_answer not in answer_groups:
            answer_groups[normalized_answer] = {
                "original_answer": answer,  # Keep original formatting
                "scores": [],
                "labels": []
            }
        
        answer_groups[normalized_answer]["scores"].append(scores[i])
        answer_groups[normalized_answer]["labels"].append(labels[i])
    
    # Aggregate scores and labels for each unique answer
    unique_answers = []
    aggregated_scores = []
    aggregated_labels = []
    
    for normalized_answer, data in answer_groups.items():
        unique_answers.append(data["original_answer"])
        
        # Aggregate scores
        if aggregation_method == "max":
            agg_score = np.max(data["scores"])
        elif aggregation_method == "mean":
            agg_score = np.mean(data["scores"])
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        aggregated_scores.append(agg_score)
        
        # For labels, if any instance is correct, consider the unique answer correct
        # This handles cases where the same answer might have inconsistent labels
        answer_labels = data["labels"]
        if any(label in ["EXACT_MATCH", "True", True] for label in answer_labels):
            aggregated_labels.append("EXACT_MATCH")  # Use consistent correct label
        else:
            aggregated_labels.append("False")  # Use consistent incorrect label
    
    return unique_answers, aggregated_scores, aggregated_labels


def calculate_k_q_with_answers(
    answers: List[str],
    scores: Union[List[float], np.ndarray],
    labels: Union[List[str], List[bool], np.ndarray],
    aggregation_method: str = "max"
) -> float:
    """
    Calculate K_q with proper answer deduplication.
    
    This is the correct way to calculate K_q - first deduplicate answers,
    then calculate K on the unique answer sets.
    
    Args:
        answers: List of answer strings (may contain duplicates)
        scores: List/array of scores for each answer
        labels: List/array of labels for each answer
        aggregation_method: How to aggregate scores for duplicate answers ("max", "mean")
        
    Returns:
        K_q value in [0, 1], representing the fraction of correctly ranked unique answer pairs.
        Returns np.nan if there are no correct-incorrect pairs after deduplication.
    """
    # Deduplicate answers and aggregate scores/labels
    unique_answers, agg_scores, agg_labels = deduplicate_answers_with_scores(
        answers, scores, labels, aggregation_method
    )
    
    # Now calculate K_q on the unique answer sets
    return calculate_k_q(agg_scores, agg_labels)


def calculate_k_q(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[str], List[bool], np.ndarray],
    correct_label: Union[str, bool, None] = None  # Ignored - kept for compatibility
) -> float:
    """
    Calculate K_q - knowledge per question.
    
    This metric quantifies the fraction of (correct, incorrect) answer pairs
    where the correct answer is ranked higher than the incorrect one.
    
    Formula from paper (Equation 1):
        K_q(s,r,o; S_M) = (1/|Ω(s,r,o)|) * Σ I(S_M(q,a) > S_M(q,ã))
    
    where:
        - Ω(s,r,o) is the set of all (correct, incorrect) answer pairs
        - S_M is a scoring function
        - I is the indicator function
    
    Args:
        scores: List/array of scores for each answer candidate
        labels: List/array of labels. Supports both:
                - Boolean: True (correct), False (incorrect) 
                - String: "EXACT_MATCH"/"True" (correct), "False" (incorrect)
        correct_label: Ignored (kept for compatibility) - auto-detects label format
        
    Returns:
        K_q value in [0, 1], representing the fraction of correctly ranked pairs.
        Returns np.nan if there are no correct-incorrect pairs (Ω is empty).
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    
    # Convert to numpy arrays for easier handling
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Handle both string labels and boolean labels
    # Correct: "EXACT_MATCH", "True", or boolean True
    # Incorrect: "False" or boolean False
    
    if len(labels) > 0 and isinstance(labels[0], (bool, np.bool_)):
        # Boolean labels: True = correct, False = incorrect
        correct_mask = labels == True
        incorrect_mask = labels == False
    else:
        # String labels: "EXACT_MATCH"/"True" = correct, "False" = incorrect  
        correct_mask = (labels == "EXACT_MATCH") | (labels == "True")
        incorrect_mask = (labels == "False")
    
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]
    
    # Handle edge cases - no (correct, incorrect) pairs means Ω is empty
    if len(correct_indices) == 0 or len(incorrect_indices) == 0:
        # Based on paper Section 4.2: "We discard ~8% of the questions 
        # where all sampled answers are correct"
        # Similarly, we should discard questions with no correct or no incorrect answers
        return np.nan
    
    # Count correctly ranked pairs
    # For each (correct, incorrect) pair, check if correct score > incorrect score
    correctly_ranked = 0
    total_pairs = 0
    
    for correct_idx in correct_indices:
        for incorrect_idx in incorrect_indices:
            total_pairs += 1
            if scores[correct_idx] > scores[incorrect_idx]:
                correctly_ranked += 1
    
    return correctly_ranked / total_pairs if total_pairs > 0 else np.nan


def calculate_k(
    questions_scores: List[List[float]],
    questions_labels: List[List[str]],
    correct_label: Union[str, bool, None] = None
) -> float:
    """
    Calculate K - overall knowledge degree for a fact.
    
    This averages K_q across all question paraphrases for a given fact.
    
    Formula from paper (Equation 2):
        K(s,r,o; S_M) = (1/|Q(s,r)|) * Σ K_q(s,r,o; S_M)
    
    Note: In the paper's main experiments, |Q(s,r)| = 1 (single question per fact),
    so K = K_q in those cases.
    
    Args:
        questions_scores: List of score lists, one per question paraphrase
        questions_labels: List of label lists, one per question paraphrase
        correct_label: The label that indicates a correct answer
        
    Returns:
        K value in [0, 1], representing average knowledge across questions
    """
    if len(questions_scores) != len(questions_labels):
        raise ValueError("questions_scores and questions_labels must have the same length")
    
    k_q_values = []
    for scores, labels in zip(questions_scores, questions_labels):
        k_q = calculate_k_q(scores, labels, correct_label)
        if not np.isnan(k_q):
            k_q_values.append(k_q)
    
    if len(k_q_values) == 0:
        return np.nan
    
    return np.mean(k_q_values)


def calculate_k_star(
    questions_scores: List[List[float]],
    questions_labels: List[List[str]],
    correct_label: Union[str, bool, None] = None
) -> int:
    """
    Calculate K* - perfect knowledge indicator.
    
    This is 1 if the model has perfect knowledge (K = 1), and 0 otherwise.
    
    Formula from paper (Equation 3):
        K*(s,r,o; S_M) = I(K(s,r,o; S_M) = 1)
    
    Args:
        questions_scores: List of score lists, one per question paraphrase
        questions_labels: List of label lists, one per question paraphrase
        correct_label: The label that indicates a correct answer
        
    Returns:
        1 if K = 1 (perfect knowledge), 0 otherwise
    """
    k = calculate_k(questions_scores, questions_labels, correct_label)
    
    if np.isnan(k):
        return 0
    
    # Use epsilon comparison for floating point
    return 1 if np.isclose(k, 1.0) else 0


def _extract_labels_from_answer_labels(
    answer_labels: List[List[str]]
) -> List[str]:
    """
    Extract labels from answer_labels structure.
    
    Args:
        answer_labels: List of [answer_text, label] pairs
        
    Returns:
        List of labels only
    """
    return [pair[1] for pair in answer_labels]


def calculate_knowledge_for_entry(
    entry: Dict,
    score_key: str = "p_true",
    correct_label: Union[str, bool, None] = None
) -> Dict[str, float]:
    """
    Calculate all knowledge metrics for a single data entry with proper answer deduplication.
    
    Args:
        entry: Dictionary containing question, answer_labels, and scores
        score_key: Key in entry dict containing the scores (e.g., "p_true")
        correct_label: The label that indicates a correct answer
        
    Returns:
        Dictionary with keys: "k_q", "k", "k_star"
    """
    if score_key not in entry:
        raise ValueError(f"Entry does not contain '{score_key}' field")
    
    scores = entry[score_key]
    answer_labels = entry["answer_labels"]
    
    # Extract answers and labels from answer_labels pairs
    answers = [pair[0] for pair in answer_labels]
    labels = [pair[1] for pair in answer_labels]
    
    # Use the new K_q calculation with answer deduplication
    k_q = calculate_k_q_with_answers(answers, scores, labels, aggregation_method="max")
    k = k_q  # In the dataset, |Q(s,r)| = 1
    k_star = 1 if (not np.isnan(k) and np.isclose(k, 1.0)) else 0
    
    return {
        "k_q": k_q,
        "k": k,
        "k_star": k_star
    }


def calculate_knowledge_for_dataset(
    data: List[Dict],
    score_key: str = "p_true",
    correct_label: Union[str, bool, None] = None
) -> Dict[str, float]:
    """
    Calculate average knowledge metrics across a dataset.
    
    Args:
        data: List of entries, each containing question, answer_labels, and scores
        score_key: Key in entry dict containing the scores (e.g., "p_true")
        correct_label: The label that indicates a correct answer
        
    Returns:
        Dictionary with keys: "mean_k_q", "mean_k", "mean_k_star", "count"
    """
    k_q_values = []
    k_values = []
    k_star_values = []
    
    for entry in data:
        metrics = calculate_knowledge_for_entry(entry, score_key, correct_label)
        
        if not np.isnan(metrics["k_q"]):
            k_q_values.append(metrics["k_q"])
            k_values.append(metrics["k"])
            k_star_values.append(metrics["k_star"])
    
    return {
        "mean_k_q": np.mean(k_q_values) if k_q_values else np.nan,
        "mean_k": np.mean(k_values) if k_values else np.nan,
        "mean_k_star": np.mean(k_star_values) if k_star_values else np.nan,
        "count": len(k_q_values)
    }


def load_json(path: Union[str, Path]) -> List[Dict]:
    """
    Load JSON data from file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded data as list of dictionaries
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data: Union[List, Dict], path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: Path to output file
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def test_label_detection():
    """Test both boolean and string label formats."""
    print("\n" + "="*50)
    print("Testing: Boolean AND String Label Formats")
    print("="*50)
    
    # Test with boolean labels (used in train_linear_probe.py)
    scores1 = [0.8, 0.3, 0.9, 0.1]
    labels1 = [True, False, True, False]
    k_q1 = calculate_k_q(scores1, labels1)
    print(f"Boolean [True, False]: K_q = {k_q1:.3f}")
    
    # Test with string "True"/"False" labels
    scores2 = [0.8, 0.3, 0.9, 0.1]
    labels2 = ["True", "False", "True", "False"]
    k_q2 = calculate_k_q(scores2, labels2)
    print(f"String ['True', 'False']: K_q = {k_q2:.3f}")
    
    # Test with "EXACT_MATCH"/"False" labels
    scores3 = [0.8, 0.3, 0.9, 0.1]
    labels3 = ["EXACT_MATCH", "False", "EXACT_MATCH", "False"]
    k_q3 = calculate_k_q(scores3, labels3)
    print(f"String ['EXACT_MATCH', 'False']: K_q = {k_q3:.3f}")
    
    # Test with MIXED string labels - both "True" and "EXACT_MATCH" as correct
    scores4 = [0.8, 0.3, 0.9, 0.1, 0.7]
    labels4 = ["True", "False", "EXACT_MATCH", "False", "True"]
    k_q4 = calculate_k_q(scores4, labels4)
    print(f"Mixed string ['True'+'EXACT_MATCH']: K_q = {k_q4:.3f}")
    
    # All should be 1.0 because correct scores are always higher than incorrect
    print(f"All should be 1.000: {k_q1:.3f} = {k_q2:.3f} = {k_q3:.3f} = {k_q4:.3f}")
    print("="*50)


def test_answer_deduplication():
    """Test the new answer deduplication functionality."""
    print("\n" + "="*60)
    print("Testing: Answer Deduplication Impact on K Calculation")
    print("="*60)
    
    # Example: Same answers appearing multiple times with different scores
    answers = ["Paris", "London", "Paris", "Berlin", "Paris", "London"]
    scores = [0.9, 0.2, 0.7, 0.3, 0.8, 0.1]  # Paris: 0.9, 0.7, 0.8; London: 0.2, 0.1; Berlin: 0.3
    labels = ["EXACT_MATCH", "False", "EXACT_MATCH", "False", "EXACT_MATCH", "False"]
    
    print("Original data (with duplicates):")
    print("Answers:", answers)
    print("Scores: ", scores)
    print("Labels: ", labels)
    
    # Calculate K without deduplication (old way)
    k_q_old = calculate_k_q(scores, labels)
    print(f"\nK_q (without deduplication): {k_q_old:.4f}")
    
    # Calculate K with deduplication (new way)  
    k_q_new = calculate_k_q_with_answers(answers, scores, labels, aggregation_method="max")
    print(f"K_q (with deduplication):    {k_q_new:.4f}")
    
    # Show what deduplication did
    unique_answers, agg_scores, agg_labels = deduplicate_answers_with_scores(answers, scores, labels)
    print(f"\nAfter deduplication:")
    print("Unique answers:", unique_answers)
    print("Aggregated scores:", [f"{s:.1f}" for s in agg_scores])
    print("Aggregated labels:", agg_labels)
    
    print(f"\nDifference: {abs(k_q_new - k_q_old):.4f}")
    print("="*60)


def main():
    """
    Example usage: Calculate knowledge metrics for the development set.
    """
    # Test the label detection first
    test_label_detection()
    
    # Test the new deduplication functionality
    test_answer_deduplication()
    
    data_path = Path("./data/sampled_labeled_answers_1000_temp1/dev")
    
    # Process one file as example
    file_path = data_path / "P264.dev.json"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"Loading data from {file_path}...")
    data = load_json(file_path)
    
    print(f"Loaded {len(data)} entries")
    print("\nCalculating knowledge metrics...")
    
    # Calculate metrics for the entire dataset
    metrics = calculate_knowledge_for_dataset(data, score_key="p_true")
    
    print("\n" + "="*50)
    print("Knowledge Metrics (using p_true scores):")
    print("="*50)
    print(f"Mean K_q:    {metrics['mean_k_q']:.4f}")
    print(f"Mean K:      {metrics['mean_k']:.4f}")
    print(f"Mean K*:     {metrics['mean_k_star']:.4f}")
    print(f"Valid items: {metrics['count']}")
    print("="*50)
    
    # Show example for first entry
    print("\n" + "="*50)
    print("Example: First entry metrics")
    print("="*50)
    entry_metrics = calculate_knowledge_for_entry(data[0], score_key="p_true")
    print(f"Question: {data[0]['question']}")
    print(f"Golden answer: {data[0]['golden_answer']}")
    print(f"K_q: {entry_metrics['k_q']:.4f}")
    print(f"K:   {entry_metrics['k']:.4f}")
    print(f"K*:  {entry_metrics['k_star']}")
    print("="*50)


if __name__ == "__main__":
    main()

