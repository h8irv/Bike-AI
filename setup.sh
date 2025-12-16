#!/bin/bash

# Angle Grinder Detection - Robust Setup Script
# Uses staged installation to avoid dependency resolution issues

set -e  # Exit on error

echo "=========================================="
echo "BikeAI v5 - Angle Grinder Detection Setup"
echo "Staged Installation Method"
echo "=========================================="
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"
echo ""

# ============================================================================
# 1. CREATE DIRECTORY STRUCTURE (safe - won't overwrite existing)
# ============================================================================
echo "Step 1/7: Creating directory structure..."
mkdir -p data/raw/{grinder,background,tools}
mkdir -p data/processed/universal
mkdir -p data/features/{classical,neural}
mkdir -p models/{classical,neural,quantized}
mkdir -p config
mkdir -p results/{figures,tables}
mkdir -p notebooks
mkdir -p logs
mkdir -p .dvc
echo "✓ Directory structure ready"
echo ""

# ============================================================================
# 2. VIRTUAL ENVIRONMENT
# ============================================================================
echo "Step 2/7: Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# ============================================================================
# 3. UPGRADE CORE TOOLS
# ============================================================================
echo "Step 3/7: Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel
echo "✓ Core tools upgraded"
echo ""

# ============================================================================
# 4. STAGED PACKAGE INSTALLATION (avoids resolution-too-deep)
# ============================================================================
echo "Step 4/7: Installing packages in stages..."
echo "This approach installs compatible package groups sequentially"
echo "to avoid overwhelming pip's dependency resolver."
echo ""

# STAGE 1: Core scientific computing
echo "  [Stage 1/5] Core scientific stack..."
pip install \
    "numpy>=1.24.0,<2.0.0" \
    "scipy>=1.10.0,<1.14.0" \
    "pandas>=2.0.0,<2.3.0" \
    "scikit-learn>=1.3.0,<1.6.0"
echo "  ✓ Scientific computing installed"
echo ""

# STAGE 2: Audio processing
echo "  [Stage 2/5] Audio processing libraries..."
pip install \
    "librosa>=0.10.0,<0.11.0" \
    "soundfile>=0.12.0" \
    "audioread>=3.0.0" \
    "resampy>=0.4.0"
echo "  ✓ Audio libraries installed"
echo ""

# STAGE 3: Machine learning frameworks
echo "  [Stage 3/5] ML frameworks (TensorFlow, XGBoost)..."
pip install \
    "tensorflow>=2.15.0,<2.18.0" \
    "tensorflow-hub>=0.15.0" \
    "xgboost>=2.0.0,<2.2.0"
echo "  ✓ ML frameworks installed"
echo ""

# STAGE 4: MLOps tools
echo "  [Stage 4/5] MLOps (DVC, MLflow)..."
pip install \
    "dvc>=3.0.0" \
    "mlflow>=2.9.0,<2.20.0"
echo "  ✓ MLOps tools installed"
echo ""

# STAGE 5: Additional dependencies
echo "  [Stage 5/5] Supporting packages..."
pip install \
    "imbalanced-learn>=0.11.0" \
    "spafe>=0.3.2" \
    "shap>=0.44.0" \
    "matplotlib>=3.7.0,<3.10.0" \
    "seaborn>=0.12.0,<0.14.0" \
    "pyyaml>=6.0" \
    "joblib>=1.3.0" \
    "tqdm>=4.65.0"
echo "  ✓ Supporting packages installed"
echo ""

# STAGE 6: Jupyter environment
echo "  [Stage 6/6] Jupyter notebook environment..."
pip install \
    "jupyter>=1.0.0" \
    "ipykernel>=6.25.0" \
    "ipywidgets>=8.0.0"
echo "  ✓ Jupyter environment installed"
echo ""

echo "✓ All packages installed successfully!"
echo ""

# ============================================================================
# 5. INITIALIZE DVC
# ============================================================================
echo "Step 5/7: Initializing DVC..."
if [ ! -f ".dvc/config" ]; then
    dvc init
    echo "✓ DVC initialized"
    
    # Configure local remote storage
    DVC_REMOTE="../BikeAI_dvc_storage"
    mkdir -p "$DVC_REMOTE"
    dvc remote add -d local_storage "$DVC_REMOTE"
    echo "✓ DVC remote configured: $DVC_REMOTE"
else
    echo "✓ DVC already initialized"
fi
echo ""

# ============================================================================
# 6. SETUP MLFLOW
# ============================================================================
echo "Step 6/7: Setting up MLflow..."
mkdir -p logs/mlruns

# Create MLflow configuration note
cat > logs/MLFLOW_INFO.txt << 'EOF'
MLflow Tracking Setup
=====================

To view your experiments:
  1. Open terminal in BikeAIv5 directory
  2. Activate environment: source venv/bin/activate
  3. Run: mlflow ui --backend-store-uri logs/mlruns
  4. Open browser: http://localhost:5000

The experiment 'angle_grinder_pipeline' will be created 
automatically when you run Notebook 00.
EOF

echo "✓ MLflow directories created"
echo "  View experiments: mlflow ui --backend-store-uri logs/mlruns"
echo ""

# ============================================================================
# 7. VERIFY INSTALLATION
# ============================================================================
echo "Step 7/7: Verifying installation..."
echo ""

python3 << 'VERIFY_PYTHON'

import sys
print(f"Python version: {sys.version.split()[0]}")

# Check key packages
packages = [
    "numpy", "scipy", "pandas", "sklearn",
    "librosa", "tensorflow", "xgboost",
    "dvc", "mlflow", "imblearn", "spafe", "shap",
    "matplotlib", "seaborn", "yaml", "joblib"
]

print("\nInstalled packages:")
for pkg in packages:
    try:
        if pkg == "sklearn":
            import sklearn
            version = sklearn.__version__
        elif pkg == "yaml":
            import yaml
            version = yaml.__version__
        else:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "✓")
        print(f"  ✓ {pkg:20s} {version}")
    except ImportError:
        print(f"  ✗ {pkg:20s} MISSING")
        sys.exit(1)
print("\n✓ All packages verified!")
VERIFY_PYTHON

echo ""

# ============================================================================
# 8. CREATE/UPDATE .gitignore
# ============================================================================
cat > .gitignore << 'EOF'
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info/
dist/
build/

# Data (managed by DVC)
data/raw/*
data/processed/*
data/features/*
!data/.gitkeep
!data/*/.gitkeep

# Models (managed by DVC)
models/*.pkl
models/*.h5
models/*.tflite
models/*.keras
!models/.gitkeep

# Logs
logs/mlruns/
*.log

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# MLflow
mlruns/

# DVC
.dvc/cache
.dvc/tmp

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
EOF

echo "✓ .gitignore updated"
echo ""

# ============================================================================
# SETUP COMPLETE
# ============================================================================
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Environment Info:"
echo "  Python: $(python3 --version)"
echo "  Pip: $(pip --version | cut -d' ' -f2)"
echo "  Virtual env: $PROJECT_ROOT/venv"
echo ""
echo "Next Steps:"
echo ""
echo "1. Add your audio files:"
echo "   - Angle grinder recordings → data/raw/grinder/"
echo "   - Background/ambient sounds → data/raw/background/"
echo "   - Other power tools → data/raw/tools/"
echo ""
echo "2. Launch Jupyter Notebook:"
echo "   jupyter notebook notebooks/00_dvc_mlflow_setup.ipynb"
echo ""
echo "3. Run through the notebooks in order:"
echo "   00_dvc_mlflow_setup.ipynb     (Setup & verification)"
echo "   01_data_exploration.ipynb     (Explore your data)"
echo "   02_universal_preprocessing... (Process audio)"
echo "   ... and so on"
echo ""
echo "4. View MLflow experiments anytime:"
echo "   mlflow ui --backend-store-uri logs/mlruns"
echo "   Then visit: http://localhost:5000"
echo ""
echo "Important: Always activate the environment first:"
echo "   source venv/bin/activate"
echo ""
echo "=========================================="
echo "Ready to detect angle grinders! 🔧🚴"
echo "=========================================="
