"""
Test trained model on new audio files
This simulates real-world deployment
"""

import numpy as np
import librosa
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AngleGrinderDetector:
    """
    Load trained model and make predictions on new audio files
    """
    
    def __init__(self, model_path):
        """
        Load the trained model package
        
        Args:
            model_path: Path to saved model (.pkl file)
        """
        print(f"Loading model from: {model_path}")
        
        model_package = joblib.load(model_path)
        
        self.model = model_package['model']
        self.scaler = model_package['scaler']
        self.label_encoder = model_package['label_encoder']
        self.model_name = model_package['model_name']
        
        print(f"✓ Model loaded successfully")
        print(f"  Algorithm: {self.model_name}")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
        print()
    
    def extract_features(self, audio_path, sample_rate=22050, duration=5.0, n_mfcc=40):
        """
        Extract features from audio file (same as training)
        
        Args:
            audio_path: Path to audio file
            sample_rate: Sample rate (must match training)
            duration: Duration in seconds (must match training)
            n_mfcc: Number of MFCCs (must match training)
            
        Returns:
            Feature array ready for prediction
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            # Pad/truncate to fixed length
            target_length = int(sample_rate * duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=2048,
                hop_length=512
            )
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Extract additional features
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
            rms = np.mean(librosa.feature.rms(y=audio)[0])
            
            # Combine features
            features = np.concatenate([mfcc_mean, [zcr, spectral_centroid, spectral_rolloff, rms]])
            
            return features.reshape(1, -1)
            
        except Exception as e:
            print(f"ERROR extracting features from {audio_path}: {e}")
            return None
    
    def predict(self, audio_path):
        """
        Predict class for a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get class name
        class_name = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))
        
        # Build result dictionary
        result = {
            'file': Path(audio_path).name,
            'prediction': class_name,
            'confidence': confidence,
            'probabilities': {
                self.label_encoder.classes_[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        }
        
        return result
    
    def predict_batch(self, audio_files):
        """
        Predict on multiple audio files
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(audio_files)} audio files...")
        print("-"*70)
        
        for audio_file in audio_files:
            result = self.predict(audio_file)
            if result:
                results.append(result)
                
                # Display result
                pred = result['prediction']
                conf = result['confidence']
                
                # Color code output
                if pred == 'angle_grinder':
                    symbol = "🚨"
                    status = "DETECTED"
                else:
                    symbol = "✓"
                    status = "CLEAR"
                
                print(f"{symbol} {result['file']:<40} | {status:<10} | Confidence: {conf:.1%}")
        
        print("-"*70)
        print(f"Processed {len(results)} files\n")
        
        return results


def test_on_training_samples(detector, data_dir, n_samples=5):
    """
    Test model on some samples from training data
    
    Args:
        detector: AngleGrinderDetector instance
        data_dir: Path to data/raw directory
        n_samples: Number of samples per class to test
    """
    print("\n" + "="*70)
    print("TESTING ON TRAINING DATA SAMPLES")
    print("="*70 + "\n")
    
    data_dir = Path(data_dir)
    
    # Test angle grinder samples
    print("Testing Angle Grinder Samples:")
    print("-"*70)
    grinder_dir = data_dir / "angle_grinder"
    if grinder_dir.exists():
        grinder_files = list(grinder_dir.glob("*.wav"))[:n_samples]
        if len(grinder_files) == 0:
            grinder_files = list(grinder_dir.glob("*.*"))[:n_samples]
        
        detector.predict_batch(grinder_files)
    else:
        print(f"Directory not found: {grinder_dir}\n")
    
    # Test background samples
    print("\nTesting Background Noise Samples:")
    print("-"*70)
    background_dir = data_dir / "background_noise"
    if background_dir.exists():
        background_files = list(background_dir.glob("*.wav"))[:n_samples]
        if len(background_files) == 0:
            background_files = list(background_dir.glob("*.*"))[:n_samples]
        
        detector.predict_batch(background_files)
    else:
        print(f"Directory not found: {background_dir}\n")


def test_on_new_audio(detector, test_file):
    """
    Test model on a single new audio file
    
    Args:
        detector: AngleGrinderDetector instance
        test_file: Path to audio file
    """
    print("\n" + "="*70)
    print("TESTING ON NEW AUDIO FILE")
    print("="*70 + "\n")
    
    result = detector.predict(test_file)
    
    if result:
        print(f"File: {result['file']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print("\nProbability Breakdown:")
        for class_name, prob in result['probabilities'].items():
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {class_name:20} {bar} {prob:.1%}")
        
        # Alarm simulation
        if result['prediction'] == 'angle_grinder' and result['confidence'] > 0.7:
            print("\n" + "="*70)
            print("🚨 ALERT: ANGLE GRINDER DETECTED! 🚨")
            print("="*70)
            print("Actions to take:")
            print("  • Send notification to security")
            print("  • Trigger alarm system")
            print("  • Log event with timestamp")
    else:
        print("Failed to process audio file")


def calculate_performance_metrics(detector, data_dir):
    """
    Calculate overall performance on training data
    """
    print("\n" + "="*70)
    print("PERFORMANCE METRICS ON FULL DATASET")
    print("="*70 + "\n")
    
    data_dir = Path(data_dir)
    
    # Collect all files
    all_predictions = []
    all_labels = []
    
    for class_name in ['angle_grinder', 'background_noise']:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        
        audio_files = list(class_dir.glob("*.wav"))
        if len(audio_files) == 0:
            audio_files = list(class_dir.glob("*.*"))
        
        print(f"Processing {class_name}: {len(audio_files)} files...")
        
        correct = 0
        for audio_file in audio_files:
            result = detector.predict(audio_file)
            if result:
                all_predictions.append(result['prediction'])
                all_labels.append(class_name)
                if result['prediction'] == class_name:
                    correct += 1
        
        accuracy = correct / len(audio_files) if len(audio_files) > 0 else 0
        print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(audio_files)})")
    
    # Overall metrics
    if len(all_predictions) > 0:
        overall_accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_predictions)
        print(f"\nOverall Accuracy: {overall_accuracy:.1%}")
        
        # Per-class accuracy
        for class_name in set(all_labels):
            class_predictions = [p for p, l in zip(all_predictions, all_labels) if l == class_name]
            class_labels = [l for l in all_labels if l == class_name]
            class_correct = sum(p == l for p, l in zip(class_predictions, class_labels))
            class_accuracy = class_correct / len(class_labels) if len(class_labels) > 0 else 0
            print(f"  {class_name}: {class_accuracy:.1%}")


def main():
    """
    Main testing function
    """
    print("\n" + "="*70)
    print("ANGLE GRINDER DETECTOR - MODEL TESTING")
    print("="*70 + "\n")
    
    # Load trained model
    model_path = Path("models/angle_grinder_classifier.pkl")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first with train_model.py")
        return
    
    detector = AngleGrinderDetector(model_path)
    
    # Test on sample from training data
    test_on_training_samples(detector, "data/raw", n_samples=5)
    
    # Calculate full performance
    calculate_performance_metrics(detector, "data/raw")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("\nTo test on a specific file, use:")
    print("  python test_model.py path/to/your/audio.wav")
    print("\nTo integrate into your system:")
    print("  1. Import AngleGrinderDetector class")
    print("  2. Load model once at startup")
    print("  3. Call detector.predict() on new audio")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check if specific file provided
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        if Path(test_file).exists():
            model_path = Path("models/angle_grinder_classifier.pkl")
            detector = AngleGrinderDetector(model_path)
            test_on_new_audio(detector, test_file)
        else:
            print(f"File not found: {test_file}")
    else:
        # Run full test suite
        main()
