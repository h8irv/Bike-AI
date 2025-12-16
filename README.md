# Angle Grinder Audio Detection System

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
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
