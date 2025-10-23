#!/usr/bin/env python3
"""
Quick setup verification script for test evaluation.

This script checks if all required files and dependencies are available 
before running the full test evaluation with score_test_with_probe.py.
"""

import sys
from pathlib import Path
import json

def check_file(path: Path, description: str, required: bool = True) -> bool:
    """Check if a file exists and report status."""
    exists = path.exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {description}: {path}")
    if not exists and required:
        print(f"   → REQUIRED: This file is needed for the evaluation")
    elif not exists and not required:
        print(f"   → OPTIONAL: Will fallback to slower on-the-fly computation")
    return exists

def check_import(module_name: str, description: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False

def main():
    """Run all setup checks."""
    print("="*70)
    print("TEST EVALUATION SETUP VERIFICATION")
    print("="*70)
    
    all_good = True
    
    # 1. Check trained probe files
    print("\n1. Checking trained probe files...")
    models_dir = Path("./models/probes")
    
    metrics_file = models_dir / "all_layers_metrics.json"
    has_metrics = check_file(metrics_file, "Probe training metrics", required=True)
    
    if has_metrics:
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            best_layer = metrics["best_layer"]
            print(f"   → Best layer found: {best_layer}")
            
            # Check if probe and scaler files exist
            probe_file = models_dir / f"probe_layer_{best_layer}.pkl"
            scaler_file = models_dir / f"scaler_layer_{best_layer}.pkl"
            
            has_probe = check_file(probe_file, f"Trained probe (layer {best_layer})", required=True)
            has_scaler = check_file(scaler_file, f"Trained scaler (layer {best_layer})", required=True)
            
            if not (has_probe and has_scaler):
                all_good = False
                
        except Exception as e:
            print(f"❌ Error reading metrics file: {e}")
            all_good = False
    else:
        print("   → Run 'uv run python train_linear_probe.py' first")
        all_good = False
    
    # 2. Check test data availability
    print("\n2. Checking test data availability...")
    
    relations = ["P40", "P50", "P176", "P264"]
    probe_data_dir = Path("./data/probe_training_data")
    labeled_data_dir = Path("./data/sampled_labeled_answers_1000_temp1")
    
    # Check pre-computed test data (preferred)
    print("\n   Pre-computed test data (faster evaluation):")
    precomputed_files = []
    for relation in relations:
        test_file = probe_data_dir / f"{relation}.test.probe_data.pkl"
        has_file = check_file(test_file, f"Pre-computed {relation} test data", required=False)
        precomputed_files.append(has_file)
    
    has_precomputed = any(precomputed_files)
    
    # Check original test data (fallback)
    print("\n   Original test data (slower evaluation with external baselines):")
    original_files = []
    for relation in relations:
        test_file = labeled_data_dir / "test" / f"{relation}.test.json"
        has_file = check_file(test_file, f"Original {relation} test data", required=False)
        original_files.append(has_file)
    
    has_original = any(original_files)
    
    if not (has_precomputed or has_original):
        print("❌ No test data found!")
        print("   → Generate test data with: uv run python generate_data_for_probe.py test")
        all_good = False
    
    # 3. Check Python dependencies
    print("\n3. Checking Python dependencies...")
    
    deps = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sklearn", "Scikit-learn"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
    ]
    
    for module, desc in deps:
        if not check_import(module, desc):
            all_good = False
    
    # 4. Check local modules
    print("\n4. Checking local modules...")
    
    local_modules = [
        ("knowledge_calculation", "Knowledge calculation utilities"),
        ("external_scoring", "External scoring methods")
    ]
    
    for module, desc in local_modules:
        if not check_import(module, desc):
            all_good = False
    
    # 5. Check HuggingFace token (if using on-the-fly)
    print("\n5. Checking HuggingFace access...")
    
    key_file = Path("key.py")
    has_key = check_file(key_file, "HuggingFace token file", required=False)
    
    if not has_key:
        try:
            import os
            token = os.getenv("HUGGINGFACE_HUB_TOKEN")
            if token:
                print("✅ HuggingFace token found in environment")
            else:
                print("⚠️ No HuggingFace token found")
                print("   → Create key.py with KEY='your_token' or set HUGGINGFACE_HUB_TOKEN")
                print("   → Only needed for on-the-fly evaluation (if no pre-computed data)")
        except:
            print("⚠️ Could not check environment variables")
    
    # 6. Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*70)
    
    if all_good:
        print("✅ All required components are available!")
        
        if has_precomputed:
            print("\n📈 FAST MODE READY:")
            print("   → Run: uv run python score_test_with_probe.py")
            print("   → Uses pre-computed hidden states (5-10 minutes)")
            print("   → Internal scoring only")
        
        if has_original:
            print("\n🔬 FULL MODE READY:")
            print("   → Delete pre-computed files for full evaluation")
            print("   → Includes external scoring baselines (2-4 hours)")
            print("   → Complete hidden knowledge analysis")
            
    else:
        print("❌ Some required components are missing!")
        print("\nTo fix:")
        
        if not has_metrics:
            print("   1. Train probes first: uv run python train_linear_probe.py")
        
        if not (has_precomputed or has_original):
            print("   2. Generate test data: uv run python generate_data_for_probe.py test")
        
        print("   3. Install missing dependencies with: pip install -r requirements.txt")
    
    print("\n" + "="*70)
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
