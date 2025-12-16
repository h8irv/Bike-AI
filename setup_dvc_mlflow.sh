#!/bin/bash
# Setup script for DVC and MLflow integration

set -e  # Exit on error

echo "🚀 Setting up DVC and MLflow for BikeAIv4..."
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "   Activating venv..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "❌ Error: venv not found. Run setup_environment.sh first."
        exit 1
    fi
fi

echo "📦 Installing DVC and MLflow..."
pip install -q dvc dvc-gs mlflow nbdime

echo ""
echo "🔧 Configuring Git for Jupyter notebooks..."
nbdime config-git --enable

echo ""
echo "📊 Initializing DVC..."
if [ ! -d ".dvc" ]; then
    dvc init
    echo "   ✓ DVC initialized"
else
    echo "   ✓ DVC already initialized"
fi

echo ""
echo "📁 Setting up DVC remote storage..."
echo "   Choose storage backend:"
echo "   1) Local storage (recommended for testing)"
echo "   2) Google Drive"
echo "   3) Manual setup later"
read -p "   Enter choice [1-3]: " choice

case $choice in
    1)
        REMOTE_DIR="$HOME/Development/dvc-storage/BikeAIv4"
        mkdir -p "$REMOTE_DIR"
        dvc remote add -d storage "$REMOTE_DIR"
        echo "   ✓ Local storage configured at: $REMOTE_DIR"
        ;;
    2)
        echo "   For Google Drive:"
        echo "   1. Create folder in Google Drive named 'BikeAIv4-DVC'"
        echo "   2. Get folder ID from URL: drive.google.com/drive/folders/<FOLDER_ID>"
        read -p "   Enter Google Drive folder ID: " gdrive_id
        dvc remote add -d storage gdrive://$gdrive_id
        echo "   ✓ Google Drive storage configured"
        echo "   Run 'dvc push' to authenticate and upload data"
        ;;
    3)
        echo "   ✓ Skipping remote setup - configure manually later with:"
        echo "      dvc remote add -d storage <url>"
        ;;
esac

echo ""
echo "📋 Adding DVC-tracked directories..."

# Track large data directories with DVC
if [ -d "data/raw" ] && [ ! -f "data/raw.dvc" ]; then
    dvc add data/raw
    echo "   ✓ data/raw tracked by DVC"
fi

if [ -d "data/processed" ] && [ ! -f "data/processed.dvc" ]; then
    dvc add data/processed
    echo "   ✓ data/processed tracked by DVC"
fi

if [ -d "data/augmented" ] && [ ! -f "data/augmented.dvc" ]; then
    dvc add data/augmented
    echo "   ✓ data/augmented tracked by DVC"
fi

if [ -d "models" ] && [ ! -f "models.dvc" ]; then
    dvc add models
    echo "   ✓ models tracked by DVC"
fi

echo ""
echo "🔄 Creating DVC pipeline..."
cat > dvc.yaml << 'EOF'
# DVC Pipeline for BikeAIv4
# Run with: dvc repro

stages:
  data_exploration:
    cmd: jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb --output 01_data_exploration.ipynb
    deps:
      - notebooks/01_data_exploration.ipynb
      - data/raw
    params:
      - config.yaml:
          - audio
    outs:
      - results/metrics/dataset_summary.csv

  universal_preprocessing:
    cmd: jupyter nbconvert --to notebook --execute notebooks/02_universal_preprocessing.ipynb --output 02_universal_preprocessing.ipynb
    deps:
      - notebooks/02_universal_preprocessing.ipynb
      - data/raw
    params:
      - config.yaml:
          - audio
          - augmentation
    outs:
      - data/processed/universal

  classical_features:
    cmd: jupyter nbconvert --to notebook --execute notebooks/03_classical_feature_extraction.ipynb --output 03_classical_feature_extraction.ipynb
    deps:
      - notebooks/03_classical_feature_extraction.ipynb
      - data/processed/universal
    params:
      - config.yaml:
          - features.classical
    outs:
      - data/processed/classical_features

  neural_features:
    cmd: jupyter nbconvert --to notebook --execute notebooks/04_neural_feature_extraction.ipynb --output 04_neural_feature_extraction.ipynb
    deps:
      - notebooks/04_neural_feature_extraction.ipynb
      - data/processed/universal
    params:
      - config.yaml:
          - features.neural
    outs:
      - data/processed/neural_features

  classical_training:
    cmd: jupyter nbconvert --to notebook --execute notebooks/05_classical_training_and_tuning.ipynb --output 05_classical_training_and_tuning.ipynb
    deps:
      - notebooks/05_classical_training_and_tuning.ipynb
      - data/processed/classical_features
    params:
      - config.yaml:
          - training.classical
    outs:
      - models/classical
    metrics:
      - results/metrics/classical_models_comparison.csv:
          cache: false
EOF

echo "   ✓ dvc.yaml created"

echo ""
echo "🧪 Setting up MLflow..."

# Create MLflow tracking directory
mkdir -p mlruns

echo "   ✓ MLflow directory created"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo ""
echo "1. Initialize Git (if not done):"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial commit with DVC and MLflow'"
echo ""
echo "2. Track existing data with DVC:"
echo "   git add data/*.dvc .dvc/config .dvcignore"
echo "   git commit -m 'Track data with DVC'"
echo ""
echo "3. Push data to DVC remote:"
echo "   dvc push"
echo ""
echo "4. Run pipeline:"
echo "   dvc repro"
echo ""
echo "5. View MLflow UI:"
echo "   mlflow ui"
echo "   Open http://localhost:5000"
echo ""
echo "📖 See git_workflow.md and dvc_mlflow_workflow.md for detailed usage"
echo ""
