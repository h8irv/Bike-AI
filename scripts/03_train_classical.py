#!/usr/bin/env python3
"""
03_train_classical.py — Train SVM, Random Forest, XGBoost on classical features
===============================================================================

- Loads processed classical feature .npy files and labels
- Uses data/splits/{train,val,test}.txt to build split-specific datasets by filename stem matching
- Runs GridSearchCV for SVM, RF, and optionally XGBoost, using hyperparameters from config/config.yaml
- Evaluates on validation and test sets, saves metrics, confusion matrices, ROC curves
- Saves best models to models/trained/classical/

Usage:
  python scripts/03_train_classical.py [--skip-xgb] [--n-jobs -1]
"""

from __future__ import annotations
import os
import json
import yaml
import joblib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, auc)
from sklearn.model_selection import GridSearchCV

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'config' / 'config.yaml'
SPLITS_DIR = ROOT / 'data' / 'splits'
CLASSICAL_DIR = ROOT / 'data' / 'processed' / 'classical'
PLOTS_DIR = ROOT / 'results' / 'plots'
EVALS_DIR = ROOT / 'results' / 'evaluations'
MODELS_DIR = ROOT / 'models' / 'trained' / 'classical'

for d in [PLOTS_DIR, EVALS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_cfg() -> Dict:
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def read_manifest(path: Path) -> List[str]:
    stems = set()
    if not path.exists():
        return list(stems)
    for line in open(path, 'r'):
        line = line.strip()
        if not line:
            continue
        rel, _label = line.split(',', 1)
        base = Path(rel).stem  # raw file stem
        stems.add(base)
    return list(stems)


def build_split_index() -> Dict[str, set]:
    idx = {}
    for split in ['train', 'val', 'test']:
        stems = read_manifest(SPLITS_DIR / f'{split}.txt')
        # Match any feature file that starts with one of these stems + "_seg"
        idx[split] = set(stems)
    return idx


def collect_features_for_split(split_stems: set) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    paths: List[str] = []

    # Label mapping from folder names (stable ordering)
    class_dirs = [p for p in CLASSICAL_DIR.iterdir() if p.is_dir()]
    class_names = sorted([p.name for p in class_dirs])
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    # Walk all .npy feature files
    for cdir in class_dirs:
        label = label_to_idx[cdir.name]
        for f in cdir.glob('*.npy'):
            name = f.stem  # e.g., originalstem_seg0001
            base_stem = name.split('_seg')[0]
            if base_stem in split_stems:
                try:
                    arr = np.load(f)
                except Exception:
                    continue
                X_list.append(arr.astype(np.float32))
                y_list.append(label)
                paths.append(str(f))

    if not X_list:
        return np.empty((0,)), np.empty((0,), dtype=int), []

    # Ensure 2D features
    X = np.vstack([x.reshape(1, -1) if x.ndim > 1 else x[np.newaxis, ...] for x in X_list])
    y = np.array(y_list, dtype=int)
    return X, y, paths


def plot_confusion(y_true, y_pred, class_names: List[str], title: str, save_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_and_save(name: str, model, X_val, y_val, class_names: List[str]):
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
    try:
        # OvR ROC-AUC using decision function or probabilities
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_val)
        elif hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_val)
        else:
            y_score = None
        rocauc = float(roc_auc_score(y_val, y_score, multi_class='ovr')) if y_score is not None else None
    except Exception:
        rocauc = None

    # Save report JSON
    out_json = EVALS_DIR / f'{name}_val_report.json'
    with open(out_json, 'w') as f:
        json.dump({'classification_report': report, 'roc_auc_ovr': rocauc}, f, indent=2)

    # Confusion matrix plot
    out_png = PLOTS_DIR / f'{name}_val_confusion.png'
    plot_confusion(y_val, y_pred, class_names, f'{name} — Validation', out_png)

    return report, rocauc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-xgb', action='store_true', help='Skip XGBoost if unavailable')
    ap.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs for GridSearchCV')
    args = ap.parse_args()

    cfg = load_cfg()
    class_names = cfg.get('config', {}).get('classes', [])

    split_idx = build_split_index()
    X_train, y_train, _ = collect_features_for_split(split_idx['train'])
    X_val, y_val, _ = collect_features_for_split(split_idx['val'])
    X_test, y_test, _ = collect_features_for_split(split_idx['test'])

    if X_train.size == 0 or X_val.size == 0:
        print('[ERROR] No classical features found. Run scripts/02_preprocess_all.py with --only classical or default to generate.')
        return

    # Pipelines
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True))
    ])
    rf_pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),  # RF not sensitive, but keep consistent dims
        ('rf', RandomForestClassifier(random_state=42))
    ])

    grids = cfg.get('training', {}).get('classical', {})
    svm_grid = {
        'svc__kernel': grids.get('svm', {}).get('kernel', ['rbf']),
        'svc__C': grids.get('svm', {}).get('C', [1.0]),
        'svc__gamma': grids.get('svm', {}).get('gamma', ['scale'])
    }
    rf_grid = {
        'rf__n_estimators': grids.get('random_forest', {}).get('n_estimators', [100]),
        'rf__max_depth': grids.get('random_forest', {}).get('max_depth', [None]),
        'rf__min_samples_split': grids.get('random_forest', {}).get('min_samples_split', [2])
    }

    print('Training SVM (GridSearchCV)...')
    svm_cv = GridSearchCV(svm_pipe, svm_grid, cv=3, n_jobs=args.n_jobs, scoring='f1_weighted', verbose=1)
    svm_cv.fit(X_train, y_train)
    joblib.dump(svm_cv.best_estimator_, MODELS_DIR / 'svm_model.pkl')
    print('Best SVM:', svm_cv.best_params_)
    evaluate_and_save('svm', svm_cv.best_estimator_, X_val, y_val, class_names)

    print('Training Random Forest (GridSearchCV)...')
    rf_cv = GridSearchCV(rf_pipe, rf_grid, cv=3, n_jobs=args.n_jobs, scoring='f1_weighted', verbose=1)
    rf_cv.fit(X_train, y_train)
    joblib.dump(rf_cv.best_estimator_, MODELS_DIR / 'rf_model.pkl')
    print('Best RF:', rf_cv.best_params_)
    evaluate_and_save('rf', rf_cv.best_estimator_, X_val, y_val, class_names)

    # XGBoost (optional)
    if HAS_XGB and not args.skip_xgb:
        xgb_grid_raw = grids.get('xgboost', {})
        xgb_pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('xgb', XGBClassifier(objective='multi:softprob', num_class=len(class_names),
                                  eval_metric='mlogloss', tree_method='hist', random_state=42))
        ])
        xgb_grid = {
            'xgb__n_estimators': xgb_grid_raw.get('n_estimators', [100]),
            'xgb__max_depth': xgb_grid_raw.get('max_depth', [6]),
            'xgb__learning_rate': xgb_grid_raw.get('learning_rate', [0.1])
        }
        print('Training XGBoost (GridSearchCV)...')
        xgb_cv = GridSearchCV(xgb_pipe, xgb_grid, cv=3, n_jobs=args.n_jobs, scoring='f1_weighted', verbose=1)
        xgb_cv.fit(X_train, y_train)
        joblib.dump(xgb_cv.best_estimator_, MODELS_DIR / 'xgboost_model.pkl')
        print('Best XGB:', xgb_cv.best_params_)
        evaluate_and_save('xgboost', xgb_cv.best_estimator_, X_val, y_val, class_names)
    else:
        print('Skipping XGBoost (unavailable or --skip-xgb)')

    # Final test evaluation for SVM and RF (and XGB if available)
    models = {'svm': svm_cv.best_estimator_, 'rf': rf_cv.best_estimator_}
    if HAS_XGB and not args.skip_xgb:
        models['xgboost'] = xgb_cv.best_estimator_

    test_summary = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        test_summary[name] = rep
        plot_confusion(y_test, y_pred, class_names, f'{name} — Test', PLOTS_DIR / f'{name}_test_confusion.png')

    with open(EVALS_DIR / 'classical_test_summary.json', 'w') as f:
        json.dump(test_summary, f, indent=2)

    print('Training complete. Models saved to', MODELS_DIR)


if __name__ == '__main__':
    main()
