"""
Feature Extraction Script for Angle Grinder Detection
Run this first to process all your audio files into ML-ready features
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """Extract MFCC and additional features from audio files"""
    
    def __init__(self, sample_rate=22050, duration=5.0, n_mfcc=40):
        """
        Initialize the feature extractor
        
        Args:
            sample_rate: Target sampling rate (22050 Hz is standard)
            duration: Fixed length for all clips in seconds
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.target_length = int(sample_rate * duration)
        
        print(f"Feature Extractor Initialized:")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  Duration: {duration} seconds")
        print(f"  MFCC Coefficients: {n_mfcc}")
        print(f"  Target Length: {self.target_length} samples")
    
    def load_and_preprocess_audio(self, file_path):
        """
        Load audio file and preprocess to fixed length
        
        Args:
            file_path: Path to audio file
            
        Returns:
            numpy array of audio samples, or None if error
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Normalize audio to [-1, 1] range
            audio = librosa.util.normalize(audio)
            
            # Pad or truncate to fixed length
            if len(audio) < self.target_length:
                # Pad with zeros if too short
                padding = self.target_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            else:
                # Truncate if too long
                audio = audio[:self.target_length]
            
            return audio
            
        except Exception as e:
            print(f"  ERROR loading {file_path.name}: {e}")
            return None
    
    def extract_mfcc_features(self, audio):
        """
        Extract MFCC features from audio
        
        Args:
            audio: Audio time series
            
        Returns:
            Mean MFCC features (n_mfcc,) or None if error
        """
        try:
            # Compute MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=2048,
                hop_length=512
            )
            
            # Take mean across time to get fixed-length features
            mfcc_mean = np.mean(mfccs, axis=1)
            
            return mfcc_mean
            
        except Exception as e:
            print(f"  ERROR extracting MFCC: {e}")
            return None
    
    def extract_additional_features(self, audio):
        """
        Extract additional spectral and temporal features
        
        Args:
            audio: Audio time series
            
        Returns:
            Array of additional features [4 values]
        """
        try:
            features = []
            
            # Zero Crossing Rate (temporal feature)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
            features.append(zcr)
            
            # Spectral Centroid (brightness)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate)[0])
            features.append(spectral_centroid)
            
            # Spectral Rolloff (frequency cutoff)
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate)[0])
            features.append(spectral_rolloff)
            
            # RMS Energy (loudness)
            rms = np.mean(librosa.feature.rms(y=audio)[0])
            features.append(rms)
            
            return np.array(features)
            
        except Exception as e:
            print(f"  ERROR extracting additional features: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0])
    
    def process_single_file(self, file_path):
        """
        Process a single audio file and extract all features
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with features and metadata, or None if error
        """
        # Load audio
        audio = self.load_and_preprocess_audio(file_path)
        if audio is None:
            return None
        
        # Extract MFCC features
        mfcc_features = self.extract_mfcc_features(audio)
        if mfcc_features is None:
            return None
        
        # Extract additional features
        additional_features = self.extract_additional_features(audio)
        
        # Combine all features
        all_features = np.concatenate([mfcc_features, additional_features])
        
        # Return as dictionary
        return {
            'features': all_features,
            'file_path': str(file_path),
            'file_name': file_path.name
        }
    
    def process_dataset(self, data_dir):
        """
        Process entire dataset organized in class folders
        
        Expected structure:
            data_dir/
                angle_grinder/
                    sample_001.wav
                    sample_002.wav
                background_noise/
                    ambient_001.wav
                    traffic_001.wav
        
        Args:
            data_dir: Path to directory containing class subdirectories
            
        Returns:
            pandas DataFrame with features and labels
        """
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise ValueError(f"Directory does not exist: {data_dir}")
        
        all_features = []
        all_labels = []
        all_file_paths = []
        
        # Get class directories
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {data_dir}")
        
        print(f"\nFound {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
        print("="*70)
        
        # Process each class
        for class_dir in class_dirs:
            label = class_dir.name
            print(f"\nProcessing class: '{label}'")
            print("-"*70)
            
            # Find all audio files
            audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(list(class_dir.glob(ext)))
            
            if len(audio_files) == 0:
                print(f"  WARNING: No audio files found in {class_dir}")
                continue
            
            print(f"  Found {len(audio_files)} audio files")
            
            # Process each file with progress bar
            successful = 0
            for audio_file in tqdm(audio_files, desc=f"  Extracting features"):
                result = self.process_single_file(audio_file)
                
                if result is not None:
                    all_features.append(result['features'])
                    all_labels.append(label)
                    all_file_paths.append(result['file_path'])
                    successful += 1
            
            print(f"  Successfully processed: {successful}/{len(audio_files)} files")
        
        # Create DataFrame
        print("\n" + "="*70)
        print("Creating feature DataFrame...")
        
        # Feature column names
        feature_names = [f'mfcc_{i}' for i in range(self.n_mfcc)]
        feature_names += ['zcr', 'spectral_centroid', 'spectral_rolloff', 'rms_energy']
        
        # Create DataFrame
        features_df = pd.DataFrame(all_features, columns=feature_names)
        features_df['label'] = all_labels
        features_df['file_path'] = all_file_paths
        
        # Print summary statistics
        print(f"\nDataset Summary:")
        print(f"  Total samples: {len(features_df)}")
        print(f"  Feature dimensions: {len(feature_names)}")
        print(f"\nClass Distribution:")
        print(features_df['label'].value_counts())
        print("="*70)
        
        return features_df


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("ANGLE GRINDER DETECTION - FEATURE EXTRACTION")
    print("="*70)
    
    # Configure paths
    DATA_DIR = Path("data/raw")  # Change this to your actual path
    OUTPUT_DIR = Path("data/processed")
    OUTPUT_FILE = OUTPUT_DIR / "features.csv"
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature extractor
    print("\n[Step 1] Initializing Feature Extractor...")
    extractor = AudioFeatureExtractor(
        sample_rate=22050,
        duration=5.0,
        n_mfcc=40
    )
    
    # Process all audio files
    print("\n[Step 2] Processing Audio Files...")
    try:
        features_df = extractor.process_dataset(DATA_DIR)
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Please check that your data directory is correct and contains class subdirectories.")
        return
    
    # Save features to CSV
    print(f"\n[Step 3] Saving Features to {OUTPUT_FILE}...")
    features_df.to_csv(OUTPUT_FILE, index=False)
    
    # Calculate and display file size
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # Display sample of features
    print(f"\n[Step 4] Feature Preview:")
    print("-"*70)
    print(features_df.head())
    
    print("\n" + "="*70)
    print("✓ FEATURE EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nNext step: Run 'python train_model.py' to train your classifier")
    print(f"Features saved to: {OUTPUT_FILE}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
