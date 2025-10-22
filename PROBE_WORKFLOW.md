# Linear Probe Training Workflow

Complete step-by-step guide to train and use linear probes for internal knowledge scoring.

## Current Status

✅ Training probe data generated for all 4 relations (P40, P50, P176, P264)
- Total: ~2000 training examples
- Statistics show good distribution across greedy/sampled/golden sources

## Step 1: Generate Dev Probe Data ⚠️ REQUIRED

Before training, you need to generate dev probe data for model selection.

### Option A: Modify generate_data_for_probe.py

Change the `main()` function to process both train and dev:

```python
def main():
    # ... existing code ...
    
    # Process train (already done)
    modes_to_process = ["train", "dev"]  # Add "dev" here
    
    for mode in modes_to_process:
        greedy_path = Path(f"./data/greedy_answers/{mode}")
        labeled_path = Path(f"./data/sampled_labeled_answers_1000_temp1/{mode}")
        
        # ... rest of processing ...
```

Then run:
```bash
# This will generate dev probe data
uv run python generate_data_for_probe.py
```

### Option B: Simple Python Script

Create a quick script to generate dev data:

```python
# quick_gen_dev.py
import torch
from pathlib import Path
from generate_data_for_probe import (
    load_json, save_json, process_one_relation, 
    load_model_and_tokenizer
)

model, tokenizer = load_model_and_tokenizer()
device = "cuda"

relations = ["P40", "P50", "P176", "P264"]
for rel in relations:
    greedy = load_json(f"data/greedy_answers/dev/{rel}.dev.json")
    labeled = load_json(f"data/sampled_labeled_answers_1000_temp1/dev/{rel}.dev.json")
    
    examples, stats = process_one_relation(greedy, labeled, model, tokenizer, device)
    
    save_json(examples, f"data/probe_training_data/{rel}.dev.probe_data.json")
    save_json(stats, f"data/probe_training_data/{rel}.dev.probe_data.stats.json")
    print(f"{rel}: {len(examples)} examples")
```

## Step 2: Train Linear Probes

Once you have both train and dev probe data:

```bash
uv run python train_linear_probe.py
```

This script will:
1. Load training data from all 4 relations
2. Load dev data from all 4 relations  
3. Train a logistic regression probe for EACH layer
4. Evaluate each probe on dev set
5. Select the best layer based on dev AUC
6. Save the best probe and scaler
7. Generate a plot showing performance across layers

**Expected output:**
```
models/probes/
├── probe_layer_{best}.pkl          # Best probe
├── scaler_layer_{best}.pkl         # Best scaler
├── all_layers_metrics.json         # Metrics for all layers
├── layer_performance.png           # Performance plot
└── all_layers/                     # All probes (for analysis)
    ├── probe_layer_0.pkl
    ├── scaler_layer_0.pkl
    ...
```

**Expected results:**
- Best layer is usually in the middle-to-late layers (e.g., layer 15-25 for 32-layer model)
- Dev AUC should be high (>0.85) if the model encodes knowledge well
- Training AUC should be higher than dev AUC (normal overfitting)

## Step 3: Extract Probe Scores for Test Data

After training, you need to score test data with the trained probe.

Create `score_with_probe.py`:

```python
"""Extract probe scores for test data."""
import pickle
import torch
import numpy as np
from pathlib import Path
from generate_data_for_probe import (
    load_json, save_json, extract_hidden_states,
    load_model_and_tokenizer
)

def load_probe(probe_dir, layer_idx):
    """Load trained probe and scaler."""
    with open(probe_dir / f"probe_layer_{layer_idx}.pkl", 'rb') as f:
        probe = pickle.load(f)
    with open(probe_dir / f"scaler_layer_{layer_idx}.pkl", 'rb') as f:
        scaler = pickle.load(f)
    return probe, scaler

def score_entry(entry, model, tokenizer, probe, scaler, layer_idx, device):
    """Score all answer candidates for an entry."""
    question = entry["question"]
    probe_scores = []
    
    for answer, _ in entry["answer_labels"]:
        # Extract hidden state
        hidden_states = extract_hidden_states(model, tokenizer, question, answer, device)
        hidden_state = hidden_states[layer_idx].reshape(1, -1)
        
        # Scale and score
        hidden_state_scaled = scaler.transform(hidden_state)
        score = probe.predict_proba(hidden_state_scaled)[0][1]
        probe_scores.append(float(score))
    
    return probe_scores

def main():
    # Load best probe
    probe_dir = Path("models/probes")
    metrics = load_json(probe_dir / "all_layers_metrics.json")
    best_layer = metrics["best_layer"]
    
    print(f"Using best layer: {best_layer}")
    
    probe, scaler = load_probe(probe_dir, best_layer)
    model, tokenizer = load_model_and_tokenizer()
    device = "cuda"
    
    # Score test data
    relations = ["P40", "P50", "P176", "P264"]
    for rel in relations:
        print(f"Processing {rel}...")
        
        test_file = Path(f"data/sampled_labeled_answers_1000_temp1/dev/{rel}.dev.json")
        data = load_json(test_file)
        
        for entry in tqdm(data):
            entry["probe_scores"] = score_entry(
                entry, model, tokenizer, probe, scaler, best_layer, device
            )
        
        # Save with probe scores
        output_file = Path(f"data/scored_with_probe/dev/{rel}.dev.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(data, output_file)
        
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    from tqdm import tqdm
    main()
```

## Step 4: Calculate Hidden Knowledge

Compare internal vs external knowledge:

```python
from knowledge_calculation import calculate_knowledge_for_dataset
from pathlib import Path

relations = ["P40", "P50", "P176", "P264"]

for rel in relations:
    # Load data with both p_true and probe_scores
    data = load_json(f"data/scored_with_probe/dev/{rel}.dev.json")
    
    # Calculate with external scoring (p_true)
    metrics_external = calculate_knowledge_for_dataset(data, score_key="p_true")
    
    # Calculate with internal scoring (probe)
    metrics_internal = calculate_knowledge_for_dataset(data, score_key="probe_scores")
    
    # Calculate gap
    k_gap = (metrics_internal["mean_k"] - metrics_external["mean_k"]) / metrics_external["mean_k"]
    
    print(f"{rel}:")
    print(f"  External K: {metrics_external['mean_k']:.4f}")
    print(f"  Internal K: {metrics_internal['mean_k']:.4f}")
    print(f"  Hidden knowledge gap: {k_gap*100:.1f}%")
```

## Expected Timeline

1. Generate dev probe data: ~30-60 minutes (GPU)
2. Train probes: ~5-10 minutes (CPU is fine, scikit-learn)
3. Score test data with probe: ~30-60 minutes (GPU)
4. Calculate knowledge metrics: ~1 minute

## Troubleshooting

**Q: Dev data generation fails with "file not found"**
A: Make sure you have greedy answers for dev set. Check `data/greedy_answers/dev/`

**Q: Training fails with "dev probe data not found"**
A: You must generate dev probe data first (Step 1)

**Q: All layers have similar performance**
A: This can happen if:
- The model doesn't encode much knowledge differently across layers
- Try looking at the actual AUC values - even small differences matter

**Q: Dev AUC is very low (<0.6)**
A: This suggests:
- The model doesn't encode knowledge well internally
- Or the probe can't extract it
- Check if training data was generated correctly

## Next Steps After This Workflow

Once you have probe scores:
- Compare hidden knowledge across different models
- Analyze which types of questions have more hidden knowledge
- Try different probe architectures (MLP instead of linear)
- Experiment with different layers or layer combinations

