#!/usr/bin/env python3
"""
04_train_neural.py — Train custom CNNs on log-mel features
==========================================================

- Loads neural features from data/processed/neural/<class>/*.npy
- Uses data/splits/{train,val,test}.txt to map segments to splits by stem
- Builds Teacher/Student CNNs from config training.neural.* blocks
- Trains with EarlyStopping + ReduceLROnPlateau, saves best .h5
- Evaluates on test set and writes reports/plots

Usage:
  python scripts/04_train_neural.py --model teacher        # or student
  python scripts/04_train_neural.py --epochs 50 --batch-size 64
"""
from __future__ import annotations
import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'config' / 'config.yaml'
SPLITS_DIR = ROOT / 'data' / 'splits'
NEURAL_DIR = ROOT / 'data' / 'processed' / 'neural'
PLOTS_DIR = ROOT / 'results' / 'plots'
EVALS_DIR = ROOT / 'results' / 'evaluations'
MODELS_DIR = ROOT / 'models' / 'trained' / 'neural'

for d in [PLOTS_DIR, EVALS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_cfg() -> Dict:
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def read_manifest_stems(path: Path) -> set:
    stems = set()
    if not path.exists():
        return stems
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel, _lbl = line.split(',', 1)
            stems.add(Path(rel).stem)
    return stems


def build_split_index() -> Dict[str, set]:
    return {
        'train': read_manifest_stems(SPLITS_DIR / 'train.txt'),
        'val':   read_manifest_stems(SPLITS_DIR / 'val.txt'),
        'test':  read_manifest_stems(SPLITS_DIR / 'test.txt'),
    }


def collect_neural_for_split(split_stems: set, class_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_list, y_list, paths = [], [], []
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    for c in class_names:
        cdir = NEURAL_DIR / c
        if not cdir.exists():
            continue
        for f in cdir.glob('*.npy'):
            base = f.stem.split('_seg')[0]
            if base in split_stems:
                try:
                    arr = np.load(f)
                except Exception:
                    continue
                # Saved as [C, n_mels, T] → channels-last (n_mels, T, C)
                if arr.ndim == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                elif arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]
                else:
                    continue
                X_list.append(arr.astype(np.float32))
                y_list.append(name_to_idx[c])
                paths.append(str(f))
    if not X_list:
        return np.empty((0,)), np.empty((0,), dtype=int), []
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    return X, y, paths


def compute_class_weights(y: np.ndarray, num_classes: int) -> Dict[int, float]:
    # Inverse frequency
    counts = np.bincount(y, minlength=num_classes)
    total = counts.sum()
    weights = {i: (total / (counts[i] if counts[i] > 0 else 1)) for i in range(num_classes)}
    # normalize
    mean_w = np.mean(list(weights.values()))
    weights = {k: v / mean_w for k, v in weights.items()}
    return weights


def build_teacher_cnn(input_shape: Tuple[int, int, int], num_classes: int, cfg: Dict) -> keras.Model:
    filters = cfg.get('filters', [32, 64, 128, 256])
    kernel = cfg.get('kernel_size', [3, 3, 3, 3])
    pool = cfg.get('pool_size', [2, 2, 2, 2])
    dropout = float(cfg.get('dropout', 0.5))
    dense_units = cfg.get('dense_units', [128, 64])

    x_in = layers.Input(shape=input_shape)
    x = x_in
    for i, f in enumerate(filters):
        k = kernel[i] if i < len(kernel) else 3
        p = pool[i] if i < len(pool) else 2
        x = layers.Conv2D(f, (k, k), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(pool_size=(p, p))(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    for u in dense_units:
        x = layers.Dense(u, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(x_in, out, name='teacher_cnn')


def build_student_cnn(input_shape: Tuple[int, int, int], num_classes: int, cfg: Dict) -> keras.Model:
    filters = cfg.get('filters', [16, 32])
    kernel = cfg.get('kernel_size', [3, 3])
    pool = cfg.get('pool_size', [2, 2])
    dropout = float(cfg.get('dropout', 0.3))
    dense_units = cfg.get('dense_units', [64])

    x_in = layers.Input(shape=input_shape)
    x = x_in
    for i, f in enumerate(filters):
        k = kernel[i] if i < len(kernel) else 3
        p = pool[i] if i < len(pool) else 2
        x = layers.Conv2D(f, (k, k), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(pool_size=(p, p))(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    for u in dense_units:
        x = layers.Dense(u, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(x_in, out, name='student_cnn')


def plot_history(hist: keras.callbacks.History, name: str):
    h = hist.history
    plt.figure(figsize=(8,4))
    plt.plot(h.get('loss', []), label='train loss')
    plt.plot(h.get('val_loss', []), label='val loss')
    plt.title(f'{name} — Loss')
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{name}_loss.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(h.get('accuracy', []), label='train acc')
    plt.plot(h.get('val_accuracy', []), label='val acc')
    plt.title(f'{name} — Accuracy')
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{name}_accuracy.png', dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['teacher', 'student'], default='teacher')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    dcfg = cfg.get('config', {})
    tcfg = cfg.get('training', {})
    ncfg = tcfg.get('neural', {})

    classes = dcfg.get('classes', [])
    num_classes = len(classes)

    # Load splits
    idx = build_split_index()
    X_train, y_train, _ = collect_neural_for_split(idx['train'], classes)
    X_val,   y_val,   _ = collect_neural_for_split(idx['val'], classes)
    X_test,  y_test,  _ = collect_neural_for_split(idx['test'], classes)

    if X_train.size == 0 or X_val.size == 0:
        print('[ERROR] No neural features found. Run scripts/02_preprocess_all.py with --only neural or default to generate.')
        return

    # Input shape detection (n_mels, T, C)
    input_shape = X_train.shape[1:]

    # One-hot labels
    y_train_oh = keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = keras.utils.to_categorical(y_val,   num_classes)
    y_test_oh  = keras.utils.to_categorical(y_test,  num_classes)

    # Hyperparams
    batch_size = args.batch_size or int(tcfg.get('batch_size', 32))
    epochs     = args.epochs or int(tcfg.get('epochs', 100))
    lr         = args.lr or float(tcfg.get('learning_rate', 1e-3))

    # Build model
    if args.model == 'teacher':
        arch_cfg = ncfg.get('teacher_cnn', {})
        model = build_teacher_cnn(input_shape, num_classes, arch_cfg)
        model_name = 'teacher_cnn'
    else:
        arch_cfg = ncfg.get('student_cnn', {})
        model = build_student_cnn(input_shape, num_classes, arch_cfg)
        model_name = 'student_cnn'

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])

    # Callbacks
    early_pat = int(tcfg.get('early_stopping_patience', 15))
    rlrop_pat = int(tcfg.get('reduce_lr_patience', 10))
    rlrop_fac = float(tcfg.get('reduce_lr_factor', 0.5))

    ckpt_path = MODELS_DIR / f'{model_name}.h5'
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_pat, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=rlrop_fac, patience=rlrop_pat, verbose=1),
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor='val_loss', save_best_only=True, save_weights_only=False)
    ]

    # Class weights to handle imbalance
    class_weights = compute_class_weights(y_train, num_classes)

    hist = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    plot_history(hist, model_name)

    # Save final best model (already via checkpoint)
    model.save(ckpt_path)

    # Evaluate on test
    y_pred = np.argmax(model.predict(X_test, batch_size=batch_size), axis=1)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    with open(EVALS_DIR / f'{model_name}_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} — Test Confusion')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f'{model_name}_test_confusion.png', dpi=150); plt.close()

    print('Training complete. Saved:', ckpt_path)


if __name__ == '__main__':
    main()
