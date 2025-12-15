"""
Model Training Script for Angle Grinder Detection
Handles imbalanced datasets and trains multiple classifiers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)

# Import classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


class ImbalancedClassifier:
    """
    Train and evaluate classifiers with special handling for imbalanced data
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_f1_score = 0.0
        self.scaler = None
        self.label_encoder = None
        
    def initialize_models(self, class_ratio):
        """
        Initialize models with class_weight='balanced' for imbalanced data
        
        Args:
            class_ratio: Ratio of majority to minority class
        """
        print(f"\nInitializing models with balanced class weights...")
        print(f"  (Addressing {class_ratio:.1f}:1 class imbalance)")
        
        # All models use class_weight='balanced' to handle imbalance
        self.models = {
            'SVM_RBF': SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                class_weight='balanced',  # Critical for imbalanced data
                probability=True,
                random_state=42
            ),
            
            'SVM_Linear': SVC(
                kernel='linear',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',  # Handles imbalance
                random_state=42,
                n_jobs=-1
            ),
            
            'Logistic_Regression': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            
            'MLP': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15
            )
        }
        
        print(f"  Initialized {len(self.models)} classifiers")
        print(f"  All models configured to handle class imbalance\n")
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, class_names):
        """
        Train all models and evaluate performance
        
        Args:
            X_train, X_test: Feature arrays
            y_train, y_test: Labels
            class_names: List of class names
        """
        print("="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        minority_class_idx = 0 if sum(y_train == 0) < sum(y_train == 1) else 1
        minority_class_name = class_names[minority_class_idx]
        
        for name, model in self.models.items():
            print(f"\n{'─'*70}")
            print(f"Training: {name}")
            print(f"{'─'*70}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            # Get per-class metrics for minority class
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                y_test, y_pred, average=None
            )
            
            minority_recall = recall_per_class[minority_class_idx]
            minority_precision = precision_per_class[minority_class_idx]
            minority_f1 = f1_per_class[minority_class_idx]
            
            # Calculate ROC-AUC if possible
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_proba)
                else:
                    roc_auc = None
            except:
                roc_auc = None
            
            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'minority_recall': minority_recall,
                'minority_precision': minority_precision,
                'minority_f1': minority_f1,
                'roc_auc': roc_auc
            }
            
            # Print metrics
            print(f"  Overall Accuracy:  {accuracy:.4f}")
            print(f"  Weighted F1-Score: {f1:.4f}")
            print(f"  ─────────────────────────────────────────")
            print(f"  {minority_class_name} Detection:")
            print(f"    Precision: {minority_precision:.4f}")
            print(f"    Recall:    {minority_recall:.4f} ← CRITICAL (must detect threats!)")
            print(f"    F1-Score:  {minority_f1:.4f}")
            if roc_auc:
                print(f"  ROC-AUC:           {roc_auc:.4f}")
            
            # Track best model based on F1 score (balanced metric)
            if f1 > self.best_f1_score:
                self.best_f1_score = f1
                self.best_model = model
                self.best_model_name = name
        
        print("\n" + "="*70)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"  F1-Score: {self.best_f1_score:.4f}")
        print(f"  {minority_class_name} Recall: {self.results[self.best_model_name]['minority_recall']:.4f}")
        print("="*70 + "\n")
    
    def display_detailed_evaluation(self, X_test, y_test, class_names):
        """
        Display detailed evaluation of best model
        """
        if self.best_model is None:
            print("No model trained yet!")
            return
        
        print("\n" + "="*70)
        print(f"DETAILED EVALUATION: {self.best_model_name}")
        print("="*70)
        
        y_pred = self.best_model.predict(X_test)
        
        # Classification report
        print("\nClassification Report:")
        print("─"*70)
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            digits=4
        )
        print(report)
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print("─"*70)
        cm = confusion_matrix(y_test, y_pred)
        
        # Create formatted confusion matrix display
        print(f"\n{'':20} Predicted:")
        print(f"{'':20} {class_names[0]:>15} {class_names[1]:>15}")
        print(f"Actual:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name:15}   {cm[i,0]:>15} {cm[i,1]:>15}")
        
        # Interpret for imbalanced data
        print("\n" + "─"*70)
        print("INTERPRETATION (Critical for Imbalanced Data):")
        print("─"*70)
        
        minority_idx = 0 if sum(y_test == 0) < sum(y_test == 1) else 1
        majority_idx = 1 - minority_idx
        
        tp = cm[minority_idx, minority_idx]
        fn = cm[minority_idx, majority_idx]
        fp = cm[majority_idx, minority_idx]
        tn = cm[majority_idx, majority_idx]
        
        total_minority = tp + fn
        total_majority = tn + fp
        
        print(f"\n{class_names[minority_idx]} Detection (MINORITY CLASS):")
        print(f"  ✓ Correctly detected: {tp}/{total_minority} ({tp/total_minority*100:.1f}%)")
        print(f"  ✗ Missed (False Neg): {fn}/{total_minority} ({fn/total_minority*100:.1f}%)")
        print(f"  False alarms:         {fp} times")
        
        print(f"\n{class_names[majority_idx]} Detection (MAJORITY CLASS):")
        print(f"  ✓ Correctly identified: {tn}/{total_majority} ({tn/total_majority*100:.1f}%)")
        print(f"  ✗ Misclassified:        {fp}/{total_majority} ({fp/total_majority*100:.1f}%)")
        
        # Critical metrics for security application
        print("\n" + "─"*70)
        print("SECURITY APPLICATION METRICS:")
        print("─"*70)
        
        sensitivity = tp / total_minority  # Recall for minority class
        specificity = tn / total_majority
        false_alarm_rate = fp / total_majority
        miss_rate = fn / total_minority
        
        print(f"  Sensitivity (Threat Detection):  {sensitivity:.2%}")
        print(f"  Miss Rate (Threats Missed):      {miss_rate:.2%} ← Should be LOW")
        print(f"  False Alarm Rate:                {false_alarm_rate:.2%}")
        print(f"  Specificity:                     {specificity:.2%}")
        
        print("\n" + "="*70 + "\n")
    
    def visualize_results(self, X_test, y_test, class_names, output_dir):
        """
        Create visualization plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Analysis - Imbalanced Dataset', 
                     fontsize=16, fontweight='bold')
        
        # 1. Model Comparison
        model_names = list(self.results.keys())
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        minority_recalls = [self.results[name]['minority_recall'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, f1_scores, width, label='F1-Score', color='skyblue')
        axes[0, 0].bar(x + width/2, minority_recalls, width, 
                      label=f'{class_names[0]} Recall', color='coral')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Confusion Matrix
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=axes[0, 1],
                   cbar_kws={'label': 'Count'})
        axes[0, 1].set_title(f'Confusion Matrix - {self.best_model_name}')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')
        
        # 3. ROC Curve (if available)
        if self.results[self.best_model_name]['roc_auc'] is not None:
            y_proba = self.best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_auc = self.results[self.best_model_name]['roc_auc']
            
            axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2,
                           label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                           label='Random Classifier')
            axes[1, 0].set_xlim([0.0, 1.0])
            axes[1, 0].set_ylim([0.0, 1.05])
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate (Recall)')
            axes[1, 0].set_title('ROC Curve')
            axes[1, 0].legend(loc="lower right")
            axes[1, 0].grid(alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'ROC curve not available\nfor this model',
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('ROC Curve')
        
        # 4. Per-Class Metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        x = np.arange(len(class_names))
        width = 0.25
        
        axes[1, 1].bar(x - width, precision_per_class, width, label='Precision', color='lightgreen')
        axes[1, 1].bar(x, recall_per_class, width, label='Recall', color='lightcoral')
        axes[1, 1].bar(x + width, f1_per_class, width, label='F1-Score', color='lightblue')
        
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Per-Class Performance Metrics')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(class_names)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add sample counts as text
        for i, (class_name, count) in enumerate(zip(class_names, support)):
            axes[1, 1].text(i, 1.05, f'n={count}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / 'model_evaluation.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
        plt.close()
    
    def save_model(self, output_path, scaler, label_encoder):
        """
        Save the best model with preprocessing objects
        """
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'f1_score': self.best_f1_score,
            'minority_recall': self.results[self.best_model_name]['minority_recall'],
            'metrics': self.results[self.best_model_name]
        }
        
        joblib.dump(model_package, output_path)
        
        file_size_kb = output_path.stat().st_size / 1024
        print(f"\n✓ Model saved to: {output_path}")
        print(f"  Model size: {file_size_kb:.2f} KB")


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*70)
    print("ANGLE GRINDER DETECTION - MODEL TRAINING")
    print("Specialized for Imbalanced Datasets")
    print("="*70)
    
    # Paths
    FEATURES_FILE = Path("data/processed/features.csv")
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    
    # ========================
    # STEP 1: LOAD DATA
    # ========================
    print("\n[STEP 1] Loading Feature Data")
    print("─"*70)
    
    if not FEATURES_FILE.exists():
        print(f"ERROR: Features file not found: {FEATURES_FILE}")
        print("Please run feature_extraction.py first!")
        return
    
    features_df = pd.read_csv(FEATURES_FILE)
    print(f"  Loaded {len(features_df)} samples")
    print(f"  Features: {len(features_df.columns) - 2} dimensions")
    
    # Check class distribution
    class_dist = features_df['label'].value_counts()
    print(f"\n  Class Distribution:")
    for class_name, count in class_dist.items():
        percentage = count / len(features_df) * 100
        print(f"    {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Calculate imbalance ratio
    imbalance_ratio = class_dist.max() / class_dist.min()
    print(f"\n  ⚠️  Imbalance Ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        print(f"  This is a HIGHLY IMBALANCED dataset!")
        print(f"  Using specialized techniques to handle class imbalance...")
    
    # ========================
    # STEP 2: PREPARE DATA
    # ========================
    print("\n[STEP 2] Preparing Data for Training")
    print("─"*70)
    
    # Separate features and labels
    feature_columns = [col for col in features_df.columns 
                      if col not in ['label', 'file_path']]
    X = features_df[feature_columns].values
    y = features_df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    print(f"  Class encoding:")
    for idx, class_name in enumerate(class_names):
        print(f"    {idx} = {class_name}")
    
    # Stratified train-test split (maintains class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded  # Maintains class distribution
    )
    
    print(f"\n  Training set:")
    train_dist = pd.Series(y_train).value_counts()
    for idx, count in train_dist.items():
        print(f"    {class_names[idx]}: {count} samples")
    
    print(f"\n  Test set:")
    test_dist = pd.Series(y_test).value_counts()
    for idx, count in test_dist.items():
        print(f"    {class_names[idx]}: {count} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n  ✓ Features scaled using StandardScaler")
    
    # ========================
    # STEP 3: TRAIN MODELS
    # ========================
    print("\n[STEP 3] Training Classification Models")
    print("─"*70)
    
    classifier = ImbalancedClassifier()
    classifier.initialize_models(imbalance_ratio)
    classifier.scaler = scaler
    classifier.label_encoder = label_encoder
    
    classifier.train_and_evaluate(
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        class_names
    )
    
    # ========================
    # STEP 4: DETAILED EVALUATION
    # ========================
    print("[STEP 4] Detailed Model Evaluation")
    classifier.display_detailed_evaluation(X_test_scaled, y_test, class_names)
    
    # ========================
    # STEP 5: VISUALIZATION
    # ========================
    print("[STEP 5] Creating Visualizations")
    print("─"*70)
    classifier.visualize_results(X_test_scaled, y_test, class_names, MODEL_DIR)
    
    # ========================
    # STEP 6: SAVE MODEL
    # ========================
    print("\n[STEP 6] Saving Best Model")
    print("─"*70)
    
    model_path = MODEL_DIR / "angle_grinder_classifier.pkl"
    classifier.save_model(model_path, scaler, label_encoder)
    
    # ========================
    # FINAL SUMMARY
    # ========================
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Model: {classifier.best_model_name}")
    print(f"  Overall F1-Score: {classifier.best_f1_score:.4f}")
    print(f"  {class_names[0]} Recall: {classifier.results[classifier.best_model_name]['minority_recall']:.4f}")
    print(f"\nModel Location: {model_path}")
    print(f"Visualization: {MODEL_DIR / 'model_evaluation.png'}")
    
    # Recommendations
    minority_recall = classifier.results[classifier.best_model_name]['minority_recall']
    print("\n" + "─"*70)
    print("RECOMMENDATIONS:")
    print("─"*70)
    
    if minority_recall < 0.7:
        print("⚠️  Low recall for angle grinder detection (<70%)")
        print("   Suggestions:")
        print("   1. Collect more angle grinder samples (aim for 100+)")
        print("   2. Try data augmentation (pitch shift, time stretch)")
        print("   3. Consider using SMOTE oversampling")
    elif minority_recall < 0.85:
        print("⚡ Moderate recall for angle grinder detection (70-85%)")
        print("   Suggestions:")
        print("   1. Add more diverse angle grinder samples")
        print("   2. Verify audio quality and labeling accuracy")
    else:
        print("✓ Excellent recall for angle grinder detection (>85%)")
        print("   Model is ready for deployment testing!")
    
    print("\nNext step: Test the model with inference.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
