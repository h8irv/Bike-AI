# MLflow Integration Code for 05_classical_training_and_tuning.ipynb
# Copy these sections into your notebook cells

# ====================
# CELL 1: Imports and Setup (REPLACE EXISTING CELL 1)
# ====================

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# XGBoost is optional — handle missing package gracefully
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception as e:
    print('XGBoost not available, skipping XGBoost models:', e)
    XGB_AVAILABLE = False

PROJECT_ROOT = Path.cwd().resolve().parents[0] if Path.cwd().name == 'notebooks' else Path.cwd()
print(f"Project root: {PROJECT_ROOT}")

# MLflow configuration
mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")
mlflow.set_experiment("BikeAI_Classical_Models")

print("✓ MLflow experiment set: BikeAI_Classical_Models")
print(f"✓ Tracking URI: {PROJECT_ROOT}/mlruns")

# Load config
CFG_PATH = PROJECT_ROOT / 'config.yaml'
with open(CFG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

# Paths
FEATURES_DIR = PROJECT_ROOT / 'data' / 'processed' / 'classical_features'
MODELS_DIR = PROJECT_ROOT / 'models' / 'classical'
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ====================
# CELL 2: Load Data (KEEP YOUR EXISTING CELL, NO CHANGES NEEDED)
# ====================
# Your existing data loading code stays the same

# ====================
# CELL 3: Helper Function for MLflow Logging (NEW CELL - ADD AFTER DATA LOADING)
# ====================

def log_model_with_mlflow(model, model_name, X_train, X_test, y_train, y_test, params, cv_scores=None):
    """
    Train model and log everything to MLflow.
    
    Args:
        model: Sklearn model instance
        model_name: String name for the model
        X_train, X_test, y_train, y_test: Train/test data
        params: Dictionary of model parameters
        cv_scores: Optional cross-validation scores
    
    Returns:
        Trained model, predictions, metrics dictionary
    """
    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        
        # Train model
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log cross-validation scores if provided
        if cv_scores is not None:
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Create and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = FIGURES_DIR / f'{model_name}_confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log confusion matrix as artifact
        mlflow.log_artifact(str(cm_path))
        
        # Create classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        report_path = METRICS_DIR / f'{model_name}_classification_report.json'
        with open(report_path, 'w') as f:
            json.dump(class_report, f, indent=2)
        mlflow.log_artifact(str(report_path))
        
        # Log model with signature
        signature = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            signature=signature,
            registered_model_name=f"BikeAI_{model_name}"
        )
        
        # Save model locally as well
        model_path = MODELS_DIR / f'{model_name}_model.pkl'
        joblib.dump(model, model_path)
        
        print(f"\n{'='*60}")
        print(f"{model_name} Results:")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"✓ Model logged to MLflow")
        print(f"✓ Model saved to {model_path}")
        print(f"{'='*60}\n")
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
        
        return model, y_pred, metrics

# ====================
# CELL 4: Train SVM (REPLACE YOUR EXISTING SVM TRAINING CELL)
# ====================

print("\n" + "="*60)
print("TRAINING SVM")
print("="*60)

# SVM parameters
svm_params = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'random_state': 42,
    'test_size': 0.2,
}

# Create model
svm_model = SVC(
    kernel=svm_params['kernel'],
    C=svm_params['C'],
    gamma=svm_params['gamma'],
    random_state=svm_params['random_state']
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_cv_scores = cross_val_score(svm_model, X_train, y_train, cv=cv, scoring='accuracy')

# Train and log with MLflow
svm_model, svm_pred, svm_metrics = log_model_with_mlflow(
    model=svm_model,
    model_name="SVM",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    params=svm_params,
    cv_scores=svm_cv_scores
)

# ====================
# CELL 5: Train Random Forest (REPLACE YOUR EXISTING RF TRAINING CELL)
# ====================

print("\n" + "="*60)
print("TRAINING RANDOM FOREST")
print("="*60)

# Random Forest parameters
rf_params = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'test_size': 0.2,
}

# Create model
rf_model = RandomForestClassifier(
    n_estimators=rf_params['n_estimators'],
    max_depth=rf_params['max_depth'],
    min_samples_split=rf_params['min_samples_split'],
    min_samples_leaf=rf_params['min_samples_leaf'],
    random_state=rf_params['random_state'],
    n_jobs=-1
)

# Cross-validation
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')

# Train and log with MLflow
rf_model, rf_pred, rf_metrics = log_model_with_mlflow(
    model=rf_model,
    model_name="RandomForest",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    params=rf_params,
    cv_scores=rf_cv_scores
)

# Log feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])],
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
feature_importance.head(20).plot(x='feature', y='importance', kind='barh', ax=ax)
plt.title('Top 20 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()

fi_path = FIGURES_DIR / 'RandomForest_feature_importance.png'
plt.savefig(fi_path, dpi=150, bbox_inches='tight')
plt.close()

# Log feature importance to last MLflow run
with mlflow.start_run(run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None):
    mlflow.log_artifact(str(fi_path))

# ====================
# CELL 6: Train K-NN (REPLACE YOUR EXISTING KNN TRAINING CELL)
# ====================

print("\n" + "="*60)
print("TRAINING K-NEAREST NEIGHBORS")
print("="*60)

# KNN parameters
knn_params = {
    'n_neighbors': 5,
    'weights': 'distance',
    'metric': 'minkowski',
    'test_size': 0.2,
}

# Create model
knn_model = KNeighborsClassifier(
    n_neighbors=knn_params['n_neighbors'],
    weights=knn_params['weights'],
    metric=knn_params['metric'],
    n_jobs=-1
)

# Cross-validation
knn_cv_scores = cross_val_score(knn_model, X_train, y_train, cv=cv, scoring='accuracy')

# Train and log with MLflow
knn_model, knn_pred, knn_metrics = log_model_with_mlflow(
    model=knn_model,
    model_name="KNN",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    params=knn_params,
    cv_scores=knn_cv_scores
)

# ====================
# CELL 7: Train XGBoost (REPLACE YOUR EXISTING XGB TRAINING CELL)
# ====================

if XGB_AVAILABLE:
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    # XGBoost parameters
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'test_size': 0.2,
    }
    
    # Create model
    xgb_model = XGBClassifier(
        n_estimators=xgb_params['n_estimators'],
        max_depth=xgb_params['max_depth'],
        learning_rate=xgb_params['learning_rate'],
        subsample=xgb_params['subsample'],
        colsample_bytree=xgb_params['colsample_bytree'],
        random_state=xgb_params['random_state'],
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    # Cross-validation
    xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Train and log with MLflow
    xgb_model, xgb_pred, xgb_metrics = log_model_with_mlflow(
        model=xgb_model,
        model_name="XGBoost",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        params=xgb_params,
        cv_scores=xgb_cv_scores
    )
else:
    print("\n⚠️  XGBoost not available, skipping...")

# ====================
# CELL 8: Model Comparison (NEW CELL - ADD AT END)
# ====================

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

# Compile results
results = {
    'SVM': svm_metrics,
    'RandomForest': rf_metrics,
    'KNN': knn_metrics,
}

if XGB_AVAILABLE:
    results['XGBoost'] = xgb_metrics

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.sort_values('accuracy', ascending=False)

print("\n", comparison_df)

# Save comparison
comparison_path = METRICS_DIR / 'model_comparison.csv'
comparison_df.to_csv(comparison_path)

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Classical ML Models Comparison', fontsize=16, fontweight='bold')

metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    comparison_df[metric].plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title(metric.replace('_', ' ').title())
    ax.set_xlabel('Score')
    ax.set_xlim([0, 1])
    
    # Add value labels
    for i, v in enumerate(comparison_df[metric]):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')

plt.tight_layout()
comparison_plot_path = FIGURES_DIR / 'models_comparison.png'
plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Comparison saved to {comparison_path}")
print(f"✓ Comparison plot saved to {comparison_plot_path}")

# Log comparison to MLflow
with mlflow.start_run(run_name=f"Comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    mlflow.log_artifact(str(comparison_path))
    mlflow.log_artifact(str(comparison_plot_path))
    
    # Log best model info
    best_model = comparison_df.index[0]
    best_accuracy = comparison_df.iloc[0]['accuracy']
    mlflow.log_param("best_model", best_model)
    mlflow.log_metric("best_accuracy", best_accuracy)
    
    print(f"\n🏆 Best Model: {best_model} with accuracy: {best_accuracy:.4f}")

print("\n" + "="*60)
print("✓ All models trained and logged to MLflow!")
print(f"✓ View results: mlflow ui --backend-store-uri {PROJECT_ROOT}/mlruns")
print("="*60)
