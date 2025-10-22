# Linear Probe Training Guide

This guide explains how to implement the paper's "knowledge-aware probing" methodology for measuring internal knowledge.

## Overview

Based on the paper (Section 3.2 "Approximating the Quantities of Interest"):

**The paper's methodology:**
1. Train a linear classifier (probe) on the model's hidden states
2. Use **knowledge-aware probing**: only train on questions where the model likely knows the answer
3. For each such question, create one positive and one negative example
4. Train probes for all layers and select the best on the dev set

**Our improved approach:**
- We process **ALL training questions**, not just ones where greedy is correct
- This makes the probe more robust and able to work even when the model doesn't know the answer well

## Step 1: Generate Probe Training Data

**Script:** `generate_data_for_probe.py`

### What it does:

**UPDATED APPROACH** - Better than the paper's original methodology:

The script processes **ALL training questions** and for each one:

1. **Gets a correct answer** (priority order):
   - If greedy answer is correct → use greedy answer
   - If any sampled answer is correct → use first correct sampled answer
   - If all sampled answers are wrong → use golden answer

2. **Gets an incorrect answer** from the sampled answers

3. **Creates training pair**:
   - Positive example: (question, correct_answer) → label=1
   - Negative example: (question, incorrect_answer) → label=0

4. **Extracts hidden states** from **all layers** for both examples

5. **Saves the data** for probe training

### Key insight:

By using ALL questions (not just ones where greedy is correct), the probe learns to distinguish correct from incorrect answers across different knowledge levels. This makes it more robust and useful for measuring hidden knowledge, even when the model's external performance is poor.

### Data protection:

The paper ensures:
> "there are no subject and object overlaps between the training and test splits, preventing the probe to learn information about test entities"

Your dataset already has this property (train/dev/test splits use different entities).

### Run it:

```bash
uv run python generate_data_for_probe.py
```

**Note:** This is computationally expensive! It runs the model forward pass for each (question, answer) pair to extract hidden states.

**Expected output:**
- Creates `data/probe_training_data/` directory
- Generates files like `P264.train.probe_data.json` with hidden states for each layer
- Statistics showing:
  - Total questions processed (should be ~500 per relation)
  - How many used greedy answer (greedy was correct)
  - How many used sampled answer (greedy wrong, but some sample correct)
  - How many used golden answer (all samples were wrong)
- Each file contains examples with:
  - `question`: The question text
  - `answer`: The answer (correct or incorrect)
  - `label`: 1 for correct, 0 for incorrect
  - `hidden_states`: Dictionary mapping layer_idx → hidden state vector
  - `golden_answer`: The ground truth answer
  - `source`: Where the answer came from ("greedy", "sampled", or "golden")

## Step 2: Train Linear Probes

**What you need to do next:**

1. **For each layer**, train a logistic regression classifier:
   - Input: hidden state at that layer
   - Output: probability that answer is correct
   - Loss: Binary cross-entropy (logistic regression objective)

2. **Implementation options:**

### Option A: Using scikit-learn (Simple)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load probe training data
data = load_json("data/probe_training_data/P264.train.probe_data.json")

# For each layer
for layer_idx in range(num_layers):
    # Extract features and labels
    X = np.array([ex["hidden_states"][str(layer_idx)] for ex in data])
    y = np.array([ex["label"] for ex in data])
    
    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)
    
    # Save the classifier
    # ... (use joblib or pickle)
```

### Option B: Using PyTorch (More control)

```python
import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Train with binary cross-entropy loss
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
```

3. **Combine data from all relations:**

From the paper, they train on all relations together:
```python
# Combine all relations
all_data = []
for relation in ["P40", "P50", "P176", "P264"]:
    data = load_json(f"data/probe_training_data/{relation}.train.probe_data.json")
    all_data.extend(data)
```

4. **Select the best layer using dev set:**

From the paper:
> "We train probes for all layers and choose the best layer based on a development set."

- Train a probe for each layer (0 to num_layers-1)
- Evaluate each on the dev set
- Select the layer with the highest performance

## Step 3: Extract Probe Scores for Test Data

Once you have trained probes:

1. **For each test question:**
   - For each answer candidate
   - Extract hidden state at the best layer
   - Pass through the trained probe
   - Get probability score (this is your internal scoring function)

2. **Add probe scores to test data:**

```python
# For each test entry
for entry in test_data:
    probe_scores = []
    for answer, _ in entry["answer_labels"]:
        hidden_state = extract_hidden_states(model, tokenizer, entry["question"], answer)
        score = probe.predict_proba(hidden_state[best_layer])[0][1]
        probe_scores.append(score)
    
    entry["probe_scores"] = probe_scores
```

3. **Calculate knowledge metrics:**

```python
from knowledge_calculation import calculate_knowledge_for_dataset

# Using external scoring (p_true)
metrics_external = calculate_knowledge_for_dataset(test_data, score_key="p_true")

# Using internal scoring (probe)
metrics_internal = calculate_knowledge_for_dataset(test_data, score_key="probe_scores")

# Calculate hidden knowledge gap
gap = (metrics_internal["mean_k"] - metrics_external["mean_k"]) / metrics_external["mean_k"]
print(f"Hidden knowledge gap: {gap*100:.1f}%")
```

## Expected Results

From the paper:
- **Llama-3-8B**: ~14% relative gap
- **Mistral-7B**: ~48% relative gap  
- **Gemma-2-9B**: ~57% relative gap

Your results should show that internal knowledge (probe) > external knowledge (p_true).

## Important Considerations

1. **Computational cost:** 
   - Extracting hidden states is expensive
   - Consider processing in batches
   - You may want to cache hidden states

2. **Layer selection:**
   - The paper trains probes for all layers
   - Best layer often varies by model (usually middle-to-late layers)

3. **Standardization:**
   - Always standardize hidden states before training logistic regression
   - Save the scaler for use at test time

4. **Balanced training:**
   - Each question contributes 1 positive + 1 negative example
   - Perfectly balanced dataset

## Files Created

After running `generate_data_for_probe.py`:

```
data/probe_training_data/
├── P40.train.probe_data.json          # Training data for P40
├── P40.train.probe_data.stats.json    # Statistics
├── P50.train.probe_data.json
├── P50.train.probe_data.stats.json
├── P176.train.probe_data.json
├── P176.train.probe_data.stats.json
├── P264.train.probe_data.json
└── P264.train.probe_data.stats.json
```

Each `.probe_data.json` file contains:
- One entry per (question, answer) pair
- Hidden states from all layers
- Binary label (1=correct, 0=incorrect)

## Next Steps

1. Run `generate_data_for_probe.py` (computationally expensive, be patient!)
2. Implement probe training (Step 2 above)
3. Select best layer on dev set
4. Extract probe scores for test set
5. Compare with external scores using `knowledge_calculation.py`
6. Plot results showing hidden knowledge gap

## Questions?

If you need help with any step, especially the probe training implementation, let me know!

