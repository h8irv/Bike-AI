#!/usr/bin/env python3
"""
Angle Grinder Detector Project Setup Script
============================================
This script initializes the entire project structure, creates necessary directories,
generates configuration files, validates data structure, and prepares the environment.

Usage:
    python setup_project.py
"""

import os
import sys
from pathlib import Path
import json

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(message):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(message):
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def create_directory_structure():
    """Create all necessary directories for the project."""
    print_header("Creating Directory Structure")
    
    directories = [
        # Config
        "config",
        
        # Data directories (excluding raw - already exists)
        "data/processed/universal",
        "data/processed/classical",
        "data/processed/neural",
        "data/splits",
        
        # Notebooks
        "notebooks",
        
        # Source code
        "src/preprocessing",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/optimization",
        "src/inference",
        
        # Scripts
        "scripts",
        
        # Deployment
        "deployment/esp32s3/arduino_sketch",
        "deployment/esp32s3/models",
        "deployment/rp2040/arduino_sketch",
        "deployment/rp2040/models",
        "deployment/pc_simulator",
        
        # Models
        "models/checkpoints",
        "models/trained/classical",
        "models/trained/neural",
        "models/optimized/tflite",
        "models/optimized/c_code",
        
        # Results
        "results/evaluations",
        "results/plots",
        "results/benchmarks",
        "results/logs",
        
        # Tests
        "tests",
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {directory}")

def validate_data_structure():
    """Validate that raw data directory exists and has correct structure."""
    print_header("Validating Data Structure")
    
    required_dirs = [
        "data/raw/angle_grinder",
        "data/raw/background_noise",
        "data/raw/power_tools"
    ]
    
    all_valid = True
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            # Count audio files
            audio_files = list(path.glob("*.wav")) + list(path.glob("*.mp3")) + list(path.glob("*.m4a"))
            print_success(f"Found: {directory} ({len(audio_files)} audio files)")
        else:
            print_error(f"Missing: {directory}")
            all_valid = False
    
    if not all_valid:
        print_warning("\nSome data directories are missing. Please ensure you have:")
        print_warning("  - data/raw/angle_grinder/")
        print_warning("  - data/raw/background_noise/")
        print_warning("  - data/raw/power_tools/")
        return False
    
    return True

def create_config_yaml():
    """Create the main configuration YAML file."""
    print_header("Creating Configuration Files")
    
    config_content = """# Angle Grinder Detector Configuration
# =====================================


  raw_dir: "data/raw"
  processed_dir: "data/processed"
  sample_rate: 16000
  duration: 1.0  # seconds
  overlap: 0.5   # for training data augmentation
  
  # Train/Val/Test split ratios
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  
  # Class names (must match folder names in data/raw/)
  classes:
    - angle_grinder
    - background_noise
    - power_tools

preprocessing:
  universal:
    normalize_method: "peak"  # Options: "peak", "rms"
    highpass_cutoff: 80  # Hz - remove low-frequency rumble
    
  classical:
    # MFCC features
    n_mfcc: 13
    n_fft: 512
    hop_length: 160  # 10ms at 16kHz
    win_length: 400  # 25ms at 16kHz
    
    # Spectral features
    include_spectral: true
    include_temporal: true
    include_chroma: true
    
  neural:
    # Spectrogram settings
    n_mels: 40
    n_mfcc: 13
    n_fft: 512
    hop_length: 160
    win_length: 400
    
    # Delta features for neural networks
    include_deltas: true
    include_delta_deltas: false
    
  augmentation:
    enabled: true
    
    # Augmentation probabilities
    gaussian_noise:
      enabled: true
      prob: 0.5
      min_amplitude: 0.001
      max_amplitude: 0.015
      
    time_stretch:
      enabled: true
      prob: 0.3
      min_rate: 0.8
      max_rate: 1.2
      
    pitch_shift:
      enabled: true
      prob: 0.3
      min_semitones: -2
      max_semitones: 2
      
    time_shift:
      enabled: true
      prob: 0.5
      min_fraction: -0.3
      max_fraction: 0.3
      
    gain:
      enabled: true
      prob: 0.3
      min_gain_db: -6
      max_gain_db: 6

training:
  # General training parameters
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15
  reduce_lr_patience: 10
  reduce_lr_factor: 0.5
  
  # Classical ML
  classical:
    svm:
      kernel: "rbf"
      C: [0.1, 1, 10]
      gamma: ["scale", "auto"]
      
    random_forest:
      n_estimators: [50, 100, 200]
      max_depth: [10, 20, 30, null]
      min_samples_split: [2, 5, 10]
      
    xgboost:
      n_estimators: [50, 100, 200]
      max_depth: [3, 6, 9]
      learning_rate: [0.01, 0.1, 0.3]
  
  # Neural networks
  neural:
    teacher_cnn:
      conv_blocks: 4
      filters: [32, 64, 128, 256]
      kernel_size: [3, 3, 3, 3]
      pool_size: [2, 2, 2, 2]
      dropout: 0.5
      dense_units: [128, 64]
      
    student_cnn:
      conv_blocks: 2
      filters: [16, 32]
      kernel_size: [3, 3]
      pool_size: [2, 2]
      dropout: 0.3
      dense_units: [64]
      
  # Distillation
  distillation:
    enabled: true
    alpha: 0.1  # Weight for student loss
    temperature: 3  # Softening temperature
    
  # Transfer learning
  transfer_learning:
    mobilenet:
      freeze_layers: 100
      fine_tune_layers: 20
      
    yamnet:
      freeze_layers: "all"  # Start with all frozen
      fine_tune_layers: 10

optimization:
  quantization:
    # TFLite quantization settings
    method: "int8"  # Options: "dynamic", "int8", "float16"
    representative_dataset_size: 100
    
  pruning:
    enabled: false  # Enable if needed
    target_sparsity: 0.5
    
  micromlgen:
    # C code export settings
    use_sklearn_optimized: true

evaluation:
  # Metrics to compute
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - roc_auc
    - confusion_matrix
    
  # Cross-validation
  cv_folds: 5
  
  # Visualization
  plot_confusion_matrix: true
  plot_roc_curves: true
  plot_training_history: true
  plot_feature_importance: true

inference:
  # Live inference settings
  input_sources:
    - file
    - microphone
    - simulation
    
  # Prediction smoothing
  temporal_smoothing: true
  smoothing_window: 5  # predictions
  confidence_threshold: 0.7
  
  # Logging
  log_predictions: true
  log_latency: true
  log_confidence: true
  
  # Vibration sensor (placeholder)
  vibration:
    enabled: false
    threshold: 0.5
    require_both: false  # Require audio + vibration for detection

hardware:
  # Will be overridden by hardware_configs.yaml
  esp32s3:
    sram_kb: 520
    psram_mb: 8
    cpu_mhz: 240
    
  rp2040:
    sram_kb: 264
    cpu_mhz: 133
"""
    
    with open("config/config.yaml", "w") as f:
        f.write(config_content)
    print_success("Created: config/config.yaml")

def create_hardware_config_yaml():
    """Create hardware-specific configuration file."""
    
    hardware_config_content = """# Hardware-Specific Configuration
# =================================

esp32s3:
  # Memory constraints
  sram_kb: 520
  psram_mb: 8
  flash_mb: 8
  
  # CPU
  cpu_mhz: 240
  cores: 2
  
  # TFLite Micro settings
  tensor_arena_kb: 150
  
  # Audio capture
  i2s_sample_rate: 16000
  i2s_bits_per_sample: 16
  i2s_dma_buf_count: 8
  i2s_dma_buf_len: 512
  
  # Feature extraction on-device
  feature_type: "log_mel"  # or "mfcc"
  n_features: 40
  frame_length_ms: 25
  frame_step_ms: 10
  
  # Model constraints
  max_model_size_kb: 800
  max_inference_time_ms: 100

rp2040:
  # Memory constraints
  sram_kb: 264
  flash_mb: 2
  
  # CPU
  cpu_mhz: 133
  cores: 2
  
  # TFLite Micro settings
  tensor_arena_kb: 100
  
  # Audio capture
  i2s_sample_rate: 16000
  i2s_bits_per_sample: 16
  
  # Feature extraction on-device
  feature_type: "log_mel"
  n_features: 32
  frame_length_ms: 25
  frame_step_ms: 10
  
  # Model constraints
  max_model_size_kb: 400
  max_inference_time_ms: 150

pc_simulator:
  # Simulate device constraints on PC
  simulate_memory_limit: true
  simulate_cpu_limit: true
  simulate_latency: true
  
  # Target device to simulate
  target_device: "esp32s3"  # or "rp2040"
"""
    
    with open("config/hardware_configs.yaml", "w") as f:
        f.write(hardware_config_content)
    print_success("Created: config/hardware_configs.yaml")

def create_requirements_txt():
    """Create requirements.txt with all dependencies."""
    print_header("Creating Requirements File")
    
    requirements_content = """# Core ML and Scientific Computing
numpy==1.24.3
scipy==1.11.3
pandas==2.1.1

# Audio Processing
librosa==0.10.1
soundfile==0.12.1
audiomentations==0.35.0

# Classical Machine Learning
scikit-learn==1.3.1
xgboost==2.0.0
joblib==1.3.2

# Deep Learning
tensorflow==2.15.0
keras==2.15.0
tensorflow-hub==0.15.0

# Model Export
micromlgen==1.1.26

# Evaluation and Visualization
matplotlib==3.8.0
seaborn==0.13.0

# Audio Capture (for live inference)
pyaudio==0.2.13

# System Monitoring
psutil==5.9.5

# Configuration and Utilities
pyyaml==6.0.1
tqdm==4.66.1

# Jupyter
jupyter==1.0.0
ipykernel==6.25.2
ipywidgets==8.1.1

# Testing
pytest==7.4.2
pytest-cov==4.1.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    print_success("Created: requirements.txt")

def create_readme():
    """Create project README."""
    print_header("Creating README")
    
    readme_content = """# Angle Grinder Audio Detection System

AI-powered bike theft prevention system using audio and vibration detection.

## Project Overview

This project uses machine learning to detect angle grinder sounds (commonly used in bike theft) using a bike-mounted device. The system combines audio detection with vibration sensing for reliable theft detection.

**Target Hardware:**
- ESP32S3 Sense board
- Arduino Nano RP2040 Connect

## Setup

### 1. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 2. Run Setup Script
```bash
python setup_project.py
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Data Structure
Ensure your data is organized as:
```
data/raw/
├── angle_grinder/    # Angle grinder sound samples
├── background_noise/ # Background noise samples
└── power_tools/      # Other power tool samples
```

## Project Structure

```
angle-grinder-detector/
├── config/              # Configuration files
├── data/                # Data storage
├── notebooks/           # Jupyter notebooks for training
├── src/                 # Source code modules
├── scripts/             # Automation scripts
├── deployment/          # Device deployment code
├── models/              # Trained models
└── results/             # Evaluation results
```

## Workflow

### Phase 1: Data Preparation
```bash
python scripts/01_prepare_data.py
python scripts/02_preprocess_all.py
```

### Phase 2: Model Training
Use Jupyter notebooks for interactive training:
```bash
jupyter notebook
```

Run in order:
1. `01_data_exploration.ipynb` - Explore dataset
2. `03_classical_training.ipynb` - Train SVM, RF, XGBoost
3. `04_neural_training.ipynb` - Train CNNs
4. `05_transfer_learning.ipynb` - Fine-tune pre-trained models
5. `06_distillation.ipynb` - Distill to student models

### Phase 3: Evaluation
```bash
# Run evaluation notebook
jupyter notebook notebooks/07_model_evaluation.ipynb

# Or use script
python scripts/06_benchmark_models.py
```

### Phase 4: Optimization
```bash
python scripts/05_optimize_models.py
```

### Phase 5: Deployment Testing
```bash
# Test with simulation
python src/inference/live_inference.py --model-type tflite --input-source simulation

# Test with microphone
python src/inference/live_inference.py --model-type tflite --input-source microphone
```

## Configuration

Edit `config/config.yaml` to adjust:
- Audio preprocessing parameters
- Model hyperparameters
- Training settings
- Data augmentation options

## Results

All results are saved in `results/`:
- `evaluations/` - Model performance metrics
- `plots/` - Visualization plots
- `benchmarks/` - Inference time and model size
- `logs/` - Training and inference logs

## Hardware Deployment

See `deployment/` directory for ESP32S3 and RP2040 Arduino sketches.

## License

[Your License Here]

## Authors

[Your Name]
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    print_success("Created: README.md")

def create_init_files():
    """Create __init__.py files in all Python package directories."""
    print_header("Creating Python Package Files")
    
    init_files = [
        "src/__init__.py",
        "src/preprocessing/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        "src/optimization/__init__.py",
        "src/inference/__init__.py",
    ]
    
    for init_file in init_files:
        with open(init_file, "w") as f:
            f.write(f'"""Package: {Path(init_file).parent.name}"""\n')
        print_success(f"Created: {init_file}")

def create_gitignore():
    """Create .gitignore file."""
    print_header("Creating .gitignore")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv
pip-log.txt
pip-delete-this-directory.txt

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/checkpoints/*
models/trained/*
models/optimized/*
!models/.gitkeep

# Results
results/evaluations/*
results/plots/*
results/benchmarks/*
results/logs/*
!results/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/

# TensorBoard
logs/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print_success("Created: .gitignore")

def print_next_steps():
    """Print next steps for the user."""
    print_header("Setup Complete!")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.END}\n")
    print("1. Install dependencies:")
    print(f"   {Colors.BLUE}pip install -r requirements.txt{Colors.END}\n")
    
    print("2. Verify your installation:")
    print(f"   {Colors.BLUE}python -c 'import tensorflow; import librosa; print(\"All good!\")'{Colors.END}\n")
    
    print("3. Start with data preparation:")
    print(f"   {Colors.BLUE}python scripts/01_prepare_data.py{Colors.END}\n")
    
    print("4. Explore your ")
    print(f"   {Colors.BLUE}jupyter notebook notebooks/01_data_exploration.ipynb{Colors.END}\n")
    
    print(f"{Colors.GREEN}Your project is ready to go! 🚀{Colors.END}\n")

def main():
    """Main setup function."""
    print_header("Angle Grinder Detector - Project Setup")
    
    try:
        # Check if data/raw exists
        if not Path("data/raw").exists():
            print_error("data/raw directory not found!")
            print_warning("Please create data/raw/ and add your audio data before running setup.")
            sys.exit(1)
        
        # Create directory structure
        create_directory_structure()
        
        # Validate data
        if not validate_data_structure():
            print_warning("\nSetup will continue, but please add missing data directories.")
        
        # Create configuration files
        create_config_yaml()
        create_hardware_config_yaml()
        
        # Create requirements and README
        create_requirements_txt()
        create_readme()
        
        # Create Python package structure
        create_init_files()
        
        # Create .gitignore
        create_gitignore()
        
        # Print next steps
        print_next_steps()
        
    except Exception as e:
        print_error(f"Setup failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
