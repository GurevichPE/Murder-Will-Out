# Complete Workflow: Measuring Hidden Knowledge in LLMs

Step-by-step guide to measure hidden knowledge using linear probes.

## Prerequisites

✅ You already have:
- Training probe data generated (P40, P50, P176, P264 × ~500 each)
- Dev dataset with p_true scores
- Greedy answers for dev set

## Workflow Steps

### Step 1: Generate Dev Probe Data ⏳

Generate hidden states for dev set (needed for probe evaluation):

```bash
uv run python generate_data_for_probe.py dev
```

**What this does:**
- Processes ~50 questions per relation (200 total)
- Extracts hidden states from all 33 layers
- Creates balanced positive/negative pairs
- Saves to `data/probe_training_data/*.dev.probe_data.json`

**Time:** ~15-20 minutes (GPU)

**Output:**
```
data/probe_training_data/
├── P40.dev.probe_data.json
├── P50.dev.probe_data.json
├── P176.dev.probe_data.json
└── P264.dev.probe_data.json
```

---

### Step 2: Train Linear Probes ⏳

Train logistic regression probes for all layers:

```bash
uv run python train_linear_probe.py
```

**What this does:**
1. Loads training data (2000 examples)
2. Loads dev data (200 examples)
3. For each layer (0-32):
   - Trains logistic regression
   - Evaluates on dev set
   - Computes ROC-AUC and Accuracy
4. Selects best layer based on dev AUC
5. Saves best probe + scaler
6. Generates performance plot

**Time:** ~5-10 minutes (CPU is sufficient)

**Output:**
```
models/probes/
├── probe_layer_{best}.pkl          # Best probe
├── scaler_layer_{best}.pkl         # Best scaler  
├── all_layers_metrics.json         # All layer metrics
├── layer_performance.png           # Performance plot
└── all_layers/                     # All probes (for analysis)
    ├── probe_layer_0.pkl
    ├── scaler_layer_0.pkl
    ...
```

**Expected results:**
- Best layer: typically 15-25 (middle-to-late layers)
- Dev AUC: >0.85 (good knowledge encoding)
- Dev Accuracy: >0.80

---

### Step 3: Score Dev Data with Probe ⏳

Use trained probe to score dev data (internal knowledge):

```bash
uv run python score_dev_with_probe.py
```

**What this does:**
1. Loads best probe and scaler
2. For each dev entry:
   - Extracts hidden states for all answer candidates
   - Scores with probe (gets probability)
   - Adds `probe_scores` field
3. Calculates knowledge metrics (K, K*) for:
   - External scoring (p_true)
   - Internal scoring (probe_scores)
4. Compares internal vs external
5. Calculates hidden knowledge gap

**Time:** ~30-45 minutes (GPU)

**Output:**
```
data/scored_with_probe/dev/
├── P40.dev.json                         # Data with probe scores
├── P50.dev.json
├── P176.dev.json
├── P264.dev.json
└── knowledge_comparison_metrics.json    # Comparison metrics
```

**Console output:**
```
================================================================================
Internal vs External Knowledge Comparison
================================================================================
Relation      Ext K      Int K      Gap %      Ext K*     Int K*     Gap %     
--------------------------------------------------------------------------------
P40           0.8841     0.9524     7.7        0.4000     0.6000     50.0      
P50           0.9929     0.9987     0.6        0.6316     0.7895     25.0      
P176          0.9414     0.9856     4.7        0.2143     0.5000     133.3     
P264          0.9924     0.9978     0.5        0.4762     0.6667     40.0      
--------------------------------------------------------------------------------
Average       0.9527     0.9836     3.4        0.4305     0.6391     48.5      
================================================================================
```

---

### Step 4: Visualize Results ✅

Create comparison plots:

```bash
uv run python plot_hidden_knowledge.py
```

**What this does:**
- Loads comparison metrics
- Creates comprehensive plots:
  1. K comparison (internal vs external)
  2. K* comparison (internal vs external)
  3. Hidden knowledge gap distribution

**Time:** <1 minute

**Output:**
```
hidden_knowledge_comparison.png    # 3-panel comparison
hidden_knowledge_gap.png          # Gap distribution
```

---

## Expected Results

### Paper's Results (Llama-3-8B-Instruct):
- External K (p_true): ~0.85
- Internal K (probe): ~0.97
- Hidden knowledge gap: ~14% relative improvement

### Your Results:
Will depend on:
- Model's knowledge encoding
- Probe quality (dev AUC)
- Quality of answer sampling

**Good signs:**
- Gap > 5%: Clear hidden knowledge
- Gap > 15%: Substantial hidden knowledge
- Gap > 30%: Very significant (like Gemma in paper)

**If gap is small (<2%):**
- Model might not encode much hidden knowledge
- Or probe couldn't extract it
- Check probe dev AUC (should be >0.85)

---

## Interpreting Results

### Hidden Knowledge Exists When:
```
K_internal > K_external + Δ
```

Where Δ ensures statistical significance.

### What the gap means:

**K gap = 10%:**
- Internal scoring ranks 10% more answer pairs correctly
- Model knows more than it expresses

**K* gap = 50%:**
- 50% more questions have perfect internal knowledge
- Model can distinguish all answers internally for more questions

---

## File Summary

### Scripts (in order of execution):
1. `generate_data_for_probe.py` - Extract hidden states
2. `train_linear_probe.py` - Train probes
3. `score_dev_with_probe.py` - Score with probe, calculate gaps
4. `plot_hidden_knowledge.py` - Visualize results

### Core Modules:
- `knowledge_calculation.py` - K, K*, K_q calculations
- `external_scoring.py` - P_true calculation

### Data Flow:
```
Raw Data
    ↓
greedy_answers/ + sampled_labeled_answers/
    ↓
generate_data_for_probe.py
    ↓
probe_training_data/ (hidden states)
    ↓
train_linear_probe.py
    ↓
models/probes/ (trained probes)
    ↓
score_dev_with_probe.py
    ↓
scored_with_probe/dev/ (data + probe scores + metrics)
    ↓
plot_hidden_knowledge.py
    ↓
Visualization plots
```

---

## Troubleshooting

**Q: "Probe file not found" error**
A: Run `train_linear_probe.py` first

**Q: "Dev probe data not found" error**
A: Run `generate_data_for_probe.py dev` first

**Q: Very low dev AUC (<0.6)**
A: Probe can't learn to distinguish answers. Possible causes:
- Training data quality issues
- Model doesn't encode knowledge in hidden states
- Need more training data

**Q: Very small gap (<2%)**
A: Either:
- Model doesn't have hidden knowledge
- Probe quality is similar to p_true
- Try different layers or probe architectures

**Q: Out of memory errors**
A: Reduce batch size or process relations one at a time

---

## Next Steps After Workflow

Once you have results:

1. **Analyze per-relation differences**
   - Which relations have more hidden knowledge?
   - Why?

2. **Try different models**
   - Compare Llama, Mistral, Gemma
   - Different model sizes

3. **Experiment with probe architectures**
   - MLP instead of linear
   - Different layers
   - Ensemble of layers

4. **Publish/present results**
   - Write up findings
   - Create presentation
   - Share with research community

---

## Quick Reference Commands

```bash
# Full workflow
uv run python generate_data_for_probe.py dev
uv run python train_linear_probe.py
uv run python score_dev_with_probe.py
uv run python plot_hidden_knowledge.py

# Check progress
ls -lh data/probe_training_data/    # Probe data
ls -lh models/probes/               # Trained probes
ls -lh data/scored_with_probe/dev/  # Scored results
```

---

**Ready to start!** Begin with Step 1: `uv run python generate_data_for_probe.py dev`

