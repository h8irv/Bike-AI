#!/usr/bin/env python3
"""
05_distill_student.py — Knowledge Distillation from Teacher → Student
=====================================================================

- Loads teacher model (models/trained/neural/teacher_cnn.h5 by default)
- Builds student CNN per config.training.neural.student_cnn
- Trains with distillation: alpha * CE(y_true, student)
                           + (1 - alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T))
- Uses neural features from data/processed/neural mapped by data/splits manifests
- Saves distilled student as models/trained/neural/student_cnn_distilled.h5
- Writes evaluation report and confusion matrix for the test split

Usage:
  python scripts/05_distill_student.py \
    --teacher-path models/trained/neural/teacher_cnn.h5 \
    --epochs 50 --batch-size 64 --alpha 0.1 --temperature 3
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
                # Saved as [C, n_mels, T] → channels-last
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


def build_student_cnn(input_shape, num_classes, cfg: Dict) -> keras.Model:
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


class Distiller(keras.Model):
    def __init__(self, student, teacher, alpha=0.1, temperature=3.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.student_loss_fn = keras.losses.CategoricalCrossentropy()
        self.distill_loss_fn = keras.losses.KLDivergence()
        self.acc_metric = keras.metrics.CategoricalAccuracy(name='accuracy')

    def compile(self, optimizer, metrics=None):
        super().compile(optimizer=optimizer, metrics=metrics)

    def train_step(self, data):
        (x, y_true) = data
        # Forward pass teacher (no grad)
        teacher_logits = self.teacher(x, training=False)
        # Student forward + losses
        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            # Hard loss
            s_loss = self.student_loss_fn(y_true, student_logits)
            # Soft targets with temperature
            T = self.temperature
            t_soft = tf.nn.softmax(teacher_logits / T, axis=-1)
            s_soft = tf.nn.softmax(student_logits / T, axis=-1)
            kd_loss = self.distill_loss_fn(t_soft, s_soft) * (T * T)
            loss = self.alpha * s_loss + (1.0 - self.alpha) * kd_loss
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.acc_metric.update_state(y_true, student_logits)
        return {"loss": loss, "s_loss": s_loss, "kd_loss": kd_loss, "accuracy": self.acc_metric.result()}

    def test_step(self, data):
        x, y_true = data
        student_logits = self.student(x, training=False)
        s_loss = self.student_loss_fn(y_true, student_logits)
        self.acc_metric.update_state(y_true, student_logits)
        return {"loss": s_loss, "accuracy": self.acc_metric.result()}


def compute_class_weights(y: np.ndarray, num_classes: int) -> Dict[int, float]:
    counts = np.bincount(y, minlength=num_classes)
    total = counts.sum()
    weights = {i: (total / (counts[i] if counts[i] > 0 else 1)) for i in range(num_classes)}
    mean_w = np.mean(list(weights.values()))
    return {k: v / mean_w for k, v in weights.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher-path', type=str, default=str(MODELS_DIR / 'teacher_cnn.h5'))
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--alpha', type=float, default=None)
    ap.add_argument('--temperature', type=float, default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    dcfg = cfg.get('config', {})
    tcfg = cfg.get('training', {})
    ncfg = tcfg.get('neural', {})
    dist_cfg = tcfg.get('distillation', {})

    classes = dcfg.get('classes', [])
    num_classes = len(classes)

    # Load features
    idx = build_split_index()
    X_train, y_train, _ = collect_neural_for_split(idx['train'], classes)
    X_val,   y_val,   _ = collect_neural_for_split(idx['val'], classes)
    X_test,  y_test,  _ = collect_neural_for_split(idx['test'], classes)

    if X_train.size == 0 or X_val.size == 0:
        print('[ERROR] No neural features found. Run scripts/02_preprocess_all.py with --only neural or default to generate.')
        return

    input_shape = X_train.shape[1:]
    y_train_oh = keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = keras.utils.to_categorical(y_val,   num_classes)
    y_test_oh  = keras.utils.to_categorical(y_test,  num_classes)

    # Hyperparameters
    batch_size = args.batch_size or int(tcfg.get('batch_size', 32))
    epochs     = args.epochs or int(tcfg.get('epochs', 100))
    alpha      = args.alpha or float(dist_cfg.get('alpha', 0.1))
    temperature= args.temperature or float(dist_cfg.get('temperature', 3.0))

    # Load teacher and build student
    teacher = keras.models.load_model(args.teacher_path)
    student = build_student_cnn(input_shape, num_classes, ncfg.get('student_cnn', {}))

    # Freeze teacher
    teacher.trainable = False

    # Distiller model
    distiller = Distiller(student=student, teacher=teacher, alpha=alpha, temperature=temperature)
    distiller.compile(optimizer=keras.optimizers.Adam(learning_rate=float(tcfg.get('learning_rate', 1e-3))))

    # Callbacks
    early_pat = int(tcfg.get('early_stopping_patience', 15))
    rlrop_pat = int(tcfg.get('reduce_lr_patience', 10))
    rlrop_fac = float(tcfg.get('reduce_lr_factor', 0.5))

    ckpt_path = MODELS_DIR / 'student_cnn_distilled.h5'
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_pat, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=rlrop_fac, patience=rlrop_pat, verbose=1),
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor='val_loss', save_best_only=True, save_weights_only=False)
    ]

    class_weights = compute_class_weights(y_train, num_classes)

    hist = distiller.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Save final student
    distiller.student.save(ckpt_path)

    # Evaluate on test
    y_pred = np.argmax(distiller.student.predict(X_test, batch_size=batch_size), axis=1)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    with open(EVALS_DIR / 'student_distilled_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Student (Distilled) — Test Confusion')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(PLOTS_DIR / 'student_distilled_test_confusion.png', dpi=150); plt.close()

    print('Distillation complete. Saved:', ckpt_path)


if __name__ == '__main__':
    main()
