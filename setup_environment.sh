#!/bin/bash

# BikeAIv4 - Environment Setup Script
# This script creates the directory structure, virtual environment, and installs dependencies
# for the angle grinder detection ML project

echo "=========================================="
echo "BikeAIv4 Environment Setup"
echo "=========================================="
echo ""

# Get the script's directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/raw/angle_grinder
mkdir -p data/raw/background
mkdir -p data/raw/tools
mkdir -p data/processed/universal
mkdir -p data/processed/classical_features
mkdir -p data/processed/neural_features
mkdir -p data/augmented
mkdir -p models/classical
mkdir -p models/neural
mkdir -p models/optimized
mkdir -p models/deployment
mkdir -p notebooks
mkdir -p results/figures
mkdir -p results/metrics
mkdir -p results/reports
mkdir -p logs

echo "✓ Directory structure created"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠ Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Create requirements.txt
echo ""
echo "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core ML Libraries
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0

# Audio Processing
librosa==0.10.1
soundfile==0.12.1
audioread==3.0.0

# Deep Learning
tensorflow==2.13.0
keras==2.13.1

# Data Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Model Optimization
optuna==3.3.0

# Utilities
joblib==1.3.2
tqdm==4.66.1
pyyaml==6.0.1

# Jupyter
jupyter==1.0.0
ipykernel==6.25.1
notebook==7.0.2

# Additional Audio Processing
pydub==0.25.1
scipy==1.11.2

# Model Analysis
shap==0.42.1

EOF

echo "✓ requirements.txt created"
echo ""

# Install requirements
echo "Installing Python packages (this may take several minutes)..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All packages installed successfully"
else
    echo ""
    echo "⚠ Some packages failed to install. Please check the error messages above."
fi

# Setup Jupyter kernel
echo ""
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=bikeai-env --display-name="BikeAI Environment"
echo "✓ Jupyter kernel created"

# Create .gitignore
echo ""
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Virtual Environment
venv/
env/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data (too large for git)
data/raw/*
data/processed/*
data/augmented/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/augmented/.gitkeep

# Models (too large for git)
models/*
!models/.gitkeep

# Logs
logs/*.log
*.log

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Results (can be regenerated)
results/figures/*
results/metrics/*
!results/figures/.gitkeep
!results/metrics/.gitkeep
EOF

echo "✓ .gitignore created"
echo ""

# Create .gitkeep files
echo "Creating .gitkeep files..."
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/augmented/.gitkeep
touch models/.gitkeep
touch results/figures/.gitkeep
touch results/metrics/.gitkeep
echo "✓ .gitkeep files created"
echo ""

# Create README
echo "Creating README.md..."
cat > README.md << 'EOF'
# BikeAIv4 - Angle Grinder Detection System

AI-powered bike security system that uses sound and vibration analysis to detect angle grinder attacks.

## Project Structure

```
BikeAIv4/
├── data/
│   ├── raw/                    # Original audio recordings
│   ├── processed/              # Preprocessed and feature-extracted data
│   └── augmented/              # Augmented training data
├── models/
│   ├── classical/              # SVM, Random Forest, XGBoost models
│   ├── neural/                 # CNN and YAMNet models
│   ├── optimized/              # Ensemble and distilled models
│   └── deployment/             # Quantized models for deployment
├── notebooks/                  # Jupyter notebooks for experiments
├── results/
│   ├── figures/                # Plots and visualizations
│   ├── metrics/                # Performance metrics
│   └── reports/                # Analysis reports
└── logs/                       # Training logs

```

## Setup

1. Run the setup script:
   ```bash
   bash setup_environment.sh
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

## Workflow

The project follows a 10-notebook workflow:

1. `01_data_exploration.ipynb` - Dataset analysis
2. `02_universal_preprocessing.ipynb` - Audio preprocessing
3. `03_classical_feature_extraction.ipynb` - MFCC and spectral features
4. `04_neural_feature_extraction.ipynb` - Mel spectrograms
5. `05_classical_training_and_tuning.ipynb` - Classical ML models
6. `06_classical_optimization.ipynb` - Feature selection and ensembles
7. `07_custom_cnn_training.ipynb` - Custom CNN architecture
8. `08_yamnet_transfer_learning.ipynb` - Transfer learning with YAMNet
9. `09_deployment_optimization.ipynb` - Model compression and optimization
10. `10_evaluation_and_reporting.ipynb` - Final evaluation

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Data Format

Place raw audio files in:
- `data/raw/angle_grinder/` - Angle grinder sounds
- `data/raw/background/` - Background/ambient noise
- `data/raw/tools/` - Other tool sounds

Supported formats: WAV, MP3, FLAC
EOF

echo "✓ README.md created"
echo ""

# Create a simple config file
echo "Creating config.yaml..."
cat > config.yaml << 'EOF'
# BikeAIv4 Configuration

# Data paths

  raw_dir: "data/raw"
  processed_dir: "data/processed"
  augmented_dir: "data/augmented"

# Audio settings
audio:
  sample_rate: 16000
  duration: 1.0
  overlap: 0.5
  
# Preprocessing
preprocessing:
  normalize: "peak"
  target_db: -1.0
  highpass_cutoff: 80
  
# Feature extraction
features:
  classical:
    n_mfcc: 13
    n_fft: 2048
    hop_length: 512
  neural:
    n_mels: 40
    n_fft: 2048
    hop_length: 512

# Training
training:
  test_size: 0.2
  val_size: 0.1
  random_state: 42
  
# Model settings
models:
  classical_dir: "models/classical"
  neural_dir: "models/neural"
  optimized_dir: "models/optimized"
  deployment_dir: "models/deployment"

# Results
results:
  figures_dir: "results/figures"
  metrics_dir: "results/metrics"
  reports_dir: "results/reports"
EOF

echo "✓ config.yaml created"
echo ""

# Summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Place your audio data in the appropriate folders:"
echo "   - data/raw/angle_grinder/"
echo "   - data/raw/background/"
echo "   - data/raw/tools/"
echo ""
echo "3. Launch Jupyter Notebook:"
echo "   jupyter notebook"
echo ""
echo "4. Start with: notebooks/01_data_exploration.ipynb"
echo ""
echo "Happy coding!"
echo ""
