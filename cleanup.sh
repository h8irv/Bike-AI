#!/bin/bash
cd ~/Development/projects/ai-ml/BikeAIv5

echo "Cleaning processed data and derivatives..."

# Delete processed audio (will be regenerated)
rm -rf data/processed/*
echo "✓ Cleared processed audio"

# Delete features (will be re-extracted)
rm -rf data/features/classical/*
rm -rf data/features/neural/*
echo "✓ Cleared features"

# Delete old models
rm -rf models/classical/*
rm -rf models/neural/*
echo "✓ Cleared models"

# Delete old results
rm -rf results/*.csv
rm -rf results/figures/*.png
echo "✓ Cleared results"

# Keep MLflow logs for comparison
echo "✓ Kept MLflow logs (for experiment comparison)"

# Keep DVC cache
echo "✓ Kept DVC cache"

# Keep raw data
echo "✓ Kept raw data"

echo ""
echo "Clean! Ready to rerun pipeline:"
echo "  1. Notebook 02 (preprocessing with new params)"
echo "  2. Notebook 03 (feature extraction)"
echo "  3. Notebook 05 (training)"
