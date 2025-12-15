"""
Quick verification that training completed successfully
"""
from pathlib import Path
import joblib

print("\n" + "="*70)
print("TRAINING VERIFICATION")
print("="*70 + "\n")

# Check for model file
model_path = Path("models/angle_grinder_classifier.pkl")
viz_path = Path("models/model_evaluation.png")

print("Checking output files...")
print("-"*70)

if model_path.exists():
    size_kb = model_path.stat().st_size / 1024
    print(f"✓ Model file found: {model_path}")
    print(f"  Size: {size_kb:.2f} KB")
    
    # Load and inspect model
    model_package = joblib.load(model_path)
    print(f"\nModel Details:")
    print(f"  Algorithm: {model_package['model_name']}")
    print(f"  F1-Score: {model_package['f1_score']:.4f}")
    print(f"  Minority Class Recall: {model_package['minority_recall']:.4f}")
    print(f"  Classes: {list(model_package['label_encoder'].classes_)}")
else:
    print("✗ Model file NOT found!")
    print("  Training may have failed.")

print()

if viz_path.exists():
    size_kb = viz_path.stat().st_size / 1024
    print(f"✓ Visualization found: {viz_path}")
    print(f"  Size: {size_kb:.2f} KB")
else:
    print("✗ Visualization NOT found!")

print("\n" + "="*70 + "\n")
