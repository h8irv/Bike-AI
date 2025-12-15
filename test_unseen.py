"""
Test model on completely unseen data (real validation)
"""

import numpy as np
from pathlib import Path
from test_model import AngleGrinderDetector


def test_unseen_data():
    """
    Test model on data it has never seen before
    """
    print("\n" + "="*70)
    print("TESTING ON COMPLETELY UNSEEN DATA")
    print("="*70 + "\n")
    
    # Load model
    model_path = Path("models/angle_grinder_classifier.pkl")
    detector = AngleGrinderDetector(model_path)
    
    test_dir = Path("data/test")
    
    if not test_dir.exists():
        print("No test data found. Please create data/test/ directory")
        print("and add new audio files that weren't used in training.")
        return
    
    # Test each category
    categories = {
        'angle_grinder': 'Angle Grinder (Should DETECT)',
        'background_noise': 'Background Noise (Should be CLEAR)',
        'challenging_cases': 'Challenging Cases (Edge cases)'
    }
    
    all_results = {}
    
    for category, description in categories.items():
        cat_dir = test_dir / category
        
        if not cat_dir.exists():
            print(f"⊘ {description}: Directory not found\n")
            continue
        
        audio_files = list(cat_dir.glob("*.wav")) + list(cat_dir.glob("*.mp3"))
        
        if len(audio_files) == 0:
            print(f"⊘ {description}: No audio files found\n")
            continue
        
        print(f"\n{description}:")
        print("-"*70)
        
        results = detector.predict_batch(audio_files)
        all_results[category] = results
        
        # Calculate accuracy if we know expected label
        if category == 'angle_grinder':
            correct = sum(1 for r in results if r['prediction'] == 'angle_grinder')
            accuracy = correct / len(results) if results else 0
            print(f"Detection Rate: {accuracy:.1%} ({correct}/{len(results)})")
        elif category == 'background_noise':
            correct = sum(1 for r in results if r['prediction'] == 'background_noise')
            accuracy = correct / len(results) if results else 0
            print(f"Correct Classification: {accuracy:.1%} ({correct}/{len(results)})")
        
        print()
    
    # Summary
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if 'angle_grinder' in all_results:
        ag_results = all_results['angle_grinder']
        detected = sum(1 for r in ag_results if r['prediction'] == 'angle_grinder')
        print(f"✓ Angle Grinder Detection: {detected}/{len(ag_results)} ({detected/len(ag_results)*100:.1f}%)")
    
    if 'background_noise' in all_results:
        bg_results = all_results['background_noise']
        correct = sum(1 for r in bg_results if r['prediction'] == 'background_noise')
        print(f"✓ Background Classification: {correct}/{len(bg_results)} ({correct/len(bg_results)*100:.1f}%)")
    
    if 'challenging_cases' in all_results:
        ch_results = all_results['challenging_cases']
        print(f"✓ Challenging Cases: {len(ch_results)} tested (review manually)")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    test_unseen_data()
