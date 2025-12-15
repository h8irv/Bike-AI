"""
Test model on completely unseen data (real validation)
Save this as: test_unseen_data.py
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
    
    if not model_path.exists():
        print("ERROR: Model not found. Run train_model.py first!")
        return
    
    detector = AngleGrinderDetector(model_path)
    
    test_dir = Path("data/test")
    
    if not test_dir.exists():
        print("ERROR: Test directory not found!")
        print("\nCreate it with:")
        print("  mkdir -p data/test/angle_grinder")
        print("  mkdir -p data/test/background_noise")
        print("  mkdir -p data/test/challenging_cases")
        print("\nThen add your test audio files to these folders.")
        return
    
    # Test each category
    categories = {
        'angle_grinder': {
            'description': 'Angle Grinder Sounds (Should DETECT)',
            'expected': 'angle_grinder',
            'symbol': '🚨'
        },
        'background_noise': {
            'description': 'Background Noise (Should be CLEAR)',
            'expected': 'background_noise',
            'symbol': '✓'
        },
        'challenging_cases': {
            'description': 'Challenging Cases (Review Manually)',
            'expected': None,
            'symbol': '❓'
        }
    }
    
    all_results = {}
    total_tested = 0
    
    for category, info in categories.items():
        cat_dir = test_dir / category
        
        if not cat_dir.exists():
            print(f"⊘ {info['description']}: Directory not found")
            print(f"   Create with: mkdir -p {cat_dir}\n")
            continue
        
        # Find all audio files (multiple formats)
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']:
            audio_files.extend(list(cat_dir.glob(ext)))
        
        if len(audio_files) == 0:
            print(f"⊘ {info['description']}: No audio files found")
            print(f"   Add files to: {cat_dir}\n")
            continue
        
        print(f"\n{info['symbol']} {info['description']}")
        print("-"*70)
        print(f"Testing {len(audio_files)} files from: {cat_dir}")
        print()
        
        results = detector.predict_batch(audio_files)
        all_results[category] = results
        total_tested += len(results)
        
        # Calculate accuracy if we know expected label
        if info['expected']:
            correct = sum(1 for r in results if r['prediction'] == info['expected'])
            accuracy = correct / len(results) if results else 0
            
            if category == 'angle_grinder':
                print(f"\n📊 Detection Rate: {accuracy:.1%} ({correct}/{len(results)})")
                if accuracy >= 0.8:
                    print("   ✓ EXCELLENT - High detection rate!")
                elif accuracy >= 0.6:
                    print("   ⚠️  GOOD - Could be improved with more training data")
                else:
                    print("   ⚠️  LOW - Model may need more diverse training samples")
            
            elif category == 'background_noise':
                print(f"\n📊 Correct Classification: {accuracy:.1%} ({correct}/{len(results)})")
                false_alarms = len(results) - correct
                print(f"   False Alarms: {false_alarms}/{len(results)}")
                if accuracy >= 0.95:
                    print("   ✓ EXCELLENT - Very few false alarms!")
                elif accuracy >= 0.85:
                    print("   ⚠️  GOOD - Some false alarms, acceptable")
                else:
                    print("   ⚠️  HIGH FALSE ALARM RATE - May need retraining")
        
        else:
            # Challenging cases - show distribution
            ag_detected = sum(1 for r in results if r['prediction'] == 'angle_grinder')
            bg_detected = len(results) - ag_detected
            
            print(f"\n📊 Results Distribution:")
            print(f"   Classified as Angle Grinder: {ag_detected}/{len(results)}")
            print(f"   Classified as Background:    {bg_detected}/{len(results)}")
            print(f"   (Review these manually to understand model behavior)")
        
        print()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if total_tested == 0:
        print("\n⚠️  No test files found!")
        print("\nTo test your model:")
        print("  1. Add audio files to data/test/angle_grinder/ or data/test/background_noise/")
        print("  2. Run this script again: python test_unseen_data.py")
        print()
        return
    
    print(f"\nTotal files tested: {total_tested}")
    
    if 'angle_grinder' in all_results:
        ag_results = all_results['angle_grinder']
        detected = sum(1 for r in ag_results if r['prediction'] == 'angle_grinder')
        print(f"\n🎯 Angle Grinder Detection:")
        print(f"   Detected: {detected}/{len(ag_results)} ({detected/len(ag_results)*100:.1f}%)")
        print(f"   Missed:   {len(ag_results)-detected}/{len(ag_results)}")
    
    if 'background_noise' in all_results:
        bg_results = all_results['background_noise']
        correct = sum(1 for r in bg_results if r['prediction'] == 'background_noise')
        false_alarms = len(bg_results) - correct
        print(f"\n🎯 Background Noise Detection:")
        print(f"   Correct: {correct}/{len(bg_results)} ({correct/len(bg_results)*100:.1f}%)")
        print(f"   False Alarms: {false_alarms}/{len(bg_results)}")
    
    if 'challenging_cases' in all_results:
        ch_results = all_results['challenging_cases']
        ag_count = sum(1 for r in ch_results if r['prediction'] == 'angle_grinder')
        print(f"\n🎯 Challenging Cases:")
        print(f"   Total tested: {len(ch_results)}")
        print(f"   Flagged as angle grinder: {ag_count}")
        print(f"   (Review individual predictions above)")
    
    # Overall assessment
    print("\n" + "─"*70)
    print("ASSESSMENT:")
    print("─"*70)
    
    if 'angle_grinder' in all_results and 'background_noise' in all_results:
        ag_results = all_results['angle_grinder']
        bg_results = all_results['background_noise']
        
        detection_rate = sum(1 for r in ag_results if r['prediction'] == 'angle_grinder') / len(ag_results)
        false_alarm_rate = (len(bg_results) - sum(1 for r in bg_results if r['prediction'] == 'background_noise')) / len(bg_results)
        
        print(f"\nThreat Detection Rate:  {detection_rate:.1%}")
        print(f"False Alarm Rate:       {false_alarm_rate:.1%}")
        
        if detection_rate >= 0.85 and false_alarm_rate <= 0.15:
            print("\n✓✓✓ EXCELLENT PERFORMANCE - Ready for deployment!")
        elif detection_rate >= 0.7 and false_alarm_rate <= 0.25:
            print("\n✓✓ GOOD PERFORMANCE - Acceptable for testing deployment")
        elif detection_rate >= 0.5:
            print("\n⚠️  MODERATE PERFORMANCE - Consider collecting more training data")
        else:
            print("\n⚠️  LOW PERFORMANCE - Model needs more diverse training samples")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_unseen_data()
