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
