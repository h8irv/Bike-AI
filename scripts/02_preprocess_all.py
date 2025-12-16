#!/usr/bin/env python3
"""
02_preprocess_all.py — Universal, Classical, and Neural preprocessing
=====================================================================

Purpose
- Convert raw audio into fixed-length, normalized segments for model training ("universal").
- Extract classical ML features (MFCCs, spectral/temporal/chroma) for SVM/RF/XGB.
- Generate neural features (log-mel spectrograms, with optional deltas) for CNNs.
- Optionally apply audio augmentations on training split only.

Inputs (from config/config.yaml)
- config.raw_dir, config.processed_dir, config.sample_rate, config.duration, config.overlap
- preprocessing.universal: normalize_method ("peak"|"rms"), highpass_cutoff (Hz)
- preprocessing.classical: n_mfcc, n_fft, hop_length, win_length, include_* flags
- preprocessing.neural: n_mels, n_mfcc, n_fft, hop_length, win_length, include_deltas, include_delta_deltas
- preprocessing.augmentation: enabled and per-transform settings

Outputs (directory structure)
- data/processed/universal/<class>/<basename>_segXXXX.wav
- data/processed/classical/<class>/<basename>_segXXXX.npy   (feature vector per segment)
- data/processed/classical/features.csv                      (path,label + flattened features)
- data/processed/neural/<class>/<basename>_segXXXX.npy      (log-mel [+ deltas] per segment)

Notes
- Augmentations are applied only for training files listed in data/splits/train.txt.
- Validation/test are not augmented.
- Requires: numpy, librosa, soundfile, scipy.

Usage
- python scripts/02_preprocess_all.py [--only universal|classical|neural] [--limit N]
"""

from __future__ import annotations
import os
import sys
import csv
import yaml
import math
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, sosfilt
try:
    import soundfile as sf
    HAS_SF = True
except Exception:
    HAS_SF = False
from scipy.io import wavfile as sp_wav

def write_wav_pcm16(out_path: Path, sr: int, y: np.ndarray):
    # Convert float32 [-1,1] to int16 for SciPy writer
    y_clip = np.clip(y, -1.0, 1.0)
    y_int16 = (y_clip * 32767.0).astype(np.int16)
    sp_wav.write(str(out_path), sr, y_int16)


# --------------------------
# Utilities and configuration
# --------------------------

def project_root_from_file() -> Path:return Path(__file__).resolve().parents[1]

ROOT = project_root_from_file()
CONFIG_PATH = ROOT / 'config' / 'config.yaml'
SPLITS_DIR = ROOT / 'data' / 'splits'

from dataclasses import dataclass

@dataclass
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    sample_rate: int
    duration: float
    overlap: float
    classes: list[str]

@dataclass
class UniversalCfg:
    normalize_method: str
    highpass_cutoff: float

@dataclass
class ClassicalCfg:
    n_mfcc: int
    n_fft: int
    hop_length: int
    win_length: int
    include_spectral: bool
    include_temporal: bool
    include_chroma: bool

@dataclass
class NeuralCfg:
    n_mels: int
    n_mfcc: int
    n_fft: int
    hop_length: int
    win_length: int
    include_deltas: bool
    include_delta_deltas: bool

@dataclass
class AugmentCfg:
    enabled: bool
    gaussian_noise: dict
    time_stretch: dict
    pitch_shift: dict
    time_shift: dict
    gain: dict

@dataclass
class PreprocessConfig:
    data: DataConfig
    universal: UniversalCfg
    classical: ClassicalCfg
    neural: NeuralCfg
    augment: AugmentCfg



def load_yaml(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_config() -> PreprocessConfig:
    cfg = load_yaml(CONFIG_PATH)
    d = cfg.get('config', {})
    p = cfg.get('preprocessing', {})

    data_cfg = DataConfig(
        raw_dir=(ROOT / d.get('raw_dir', 'data/raw')).resolve(),
        processed_dir=(ROOT / d.get('processed_dir', 'data/processed')).resolve(),
        sample_rate=int(d.get('sample_rate', 16000)),
        duration=float(d.get('duration', 1.0)),
        overlap=float(d.get('overlap', 0.0)),
        classes=list(d.get('classes', [])),
    )

    uni_cfg = p.get('universal', {})
    universal = UniversalCfg(
        normalize_method=str(uni_cfg.get('normalize_method', 'peak')).lower(),
        highpass_cutoff=float(uni_cfg.get('highpass_cutoff', 0.0)),
    )

    cls_cfg = p.get('classical', {})
    classical = ClassicalCfg(
        n_mfcc=int(cls_cfg.get('n_mfcc', 13)),
        n_fft=int(cls_cfg.get('n_fft', 512)),
        hop_length=int(cls_cfg.get('hop_length', 160)),
        win_length=int(cls_cfg.get('win_length', 400)),
        include_spectral=bool(cls_cfg.get('include_spectral', True)),
        include_temporal=bool(cls_cfg.get('include_temporal', True)),
        include_chroma=bool(cls_cfg.get('include_chroma', True)),
    )

    neu_cfg = p.get('neural', {})
    neural = NeuralCfg(
        n_mels=int(neu_cfg.get('n_mels', 40)),
        n_mfcc=int(neu_cfg.get('n_mfcc', 13)),
        n_fft=int(neu_cfg.get('n_fft', 512)),
        hop_length=int(neu_cfg.get('hop_length', 160)),
        win_length=int(neu_cfg.get('win_length', 400)),
        include_deltas=bool(neu_cfg.get('include_deltas', True)),
        include_delta_deltas=bool(neu_cfg.get('include_delta_deltas', False)),
    )

    aug_cfg = p.get('augmentation', {})
    augment = AugmentCfg(
        enabled=bool(aug_cfg.get('enabled', False)),
        gaussian_noise=dict(aug_cfg.get('gaussian_noise', {})),
        time_stretch=dict(aug_cfg.get('time_stretch', {})),
        pitch_shift=dict(aug_cfg.get('pitch_shift', {})),
        time_shift=dict(aug_cfg.get('time_shift', {})),
        gain=dict(aug_cfg.get('gain', {})),
    )

    return PreprocessConfig(
        data=data_cfg,
        universal=universal,
        classical=classical,
        neural=neural,
        augment=augment,
    )

# --------------------------
# Splits and raw file loading
# --------------------------

def read_manifest(manifest: Path) -> List[Tuple[Path,str]]:
    pairs = []
    if not manifest.exists():
        return pairs
    with open(manifest, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel, label = line.split(',', 1)
            pairs.append(((ROOT / rel).resolve(), label))
    return pairs


def load_all_files_from_manifests(classes: List[str]) -> Tuple[List[Tuple[Path,str]], set]:
    train = read_manifest(SPLITS_DIR / 'train.txt')
    val = read_manifest(SPLITS_DIR / 'val.txt')
    test = read_manifest(SPLITS_DIR / 'test.txt')

    seen = set()
    all_files = []
    for pair in train + val + test:
        if pair[0].exists() and pair[1] in classes and pair[0] not in seen:
            all_files.append(pair)
            seen.add(pair[0])

    train_set = {p for p,_ in train}
    return all_files, train_set

# --------------------------
# Signal processing helpers
# --------------------------

def butter_highpass(cutoff: float, sr: int, order: int = 4):
    nyq = 0.5 * sr
    norm = max(1e-6, cutoff / nyq)
    return butter(order, norm, btype='highpass', output='sos')


def apply_highpass(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    if cutoff <= 0:
        return y
    sos = butter_highpass(cutoff, sr)
    return sosfilt(sos, y).astype(np.float32)


def peak_normalize(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = np.max(np.abs(y)) + 1e-12
    return (y / m * peak).astype(np.float32)


def rms_normalize(y: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    # Convert to RMS linear target
    target_rms = 10 ** (target_db / 20.0)
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    return (y * (target_rms / rms)).astype(np.float32)


def normalize_audio(y: np.ndarray, method: str) -> np.ndarray:
    if method == 'rms':
        return rms_normalize(y)
    return peak_normalize(y)


def segment_audio(y: np.ndarray, sr: int, duration_s: float, overlap: float) -> List[np.ndarray]:
    seg_len = int(duration_s * sr)
    if seg_len <= 0:
        return []
    hop = max(1, int(seg_len * (1.0 - overlap)))
    if len(y) < seg_len:
        # pad
        out = np.zeros(seg_len, dtype=np.float32)
        out[:len(y)] = y
        return [out]
    segs = []
    for start in range(0, len(y) - seg_len + 1, hop):
        segs.append(y[start:start+seg_len].astype(np.float32))
    # trailing pad if needed
    rem = (len(y) - seg_len) % hop
    if rem != 0:
        pad_seg = np.zeros(seg_len, dtype=np.float32)
        pad_seg[:len(y[-seg_len:])] = y[-seg_len:]
        segs.append(pad_seg)
    return segs

# --------------------------
# Augmentations
# --------------------------

def aug_gaussian_noise(y: np.ndarray, min_amp: float, max_amp: float) -> np.ndarray:
    amp = np.random.uniform(min_amp, max_amp)
    return (y + np.random.normal(0, amp, size=y.shape).astype(np.float32)).astype(np.float32)


def aug_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    y2 = librosa.effects.time_stretch(y=y, rate=rate)
    return y2.astype(np.float32)


def aug_pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    y2 = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)
    return y2.astype(np.float32)


def aug_time_shift(y: np.ndarray, fraction: float) -> np.ndarray:
    shift = int(len(y) * fraction)
    return np.roll(y, shift).astype(np.float32)


def aug_gain(y: np.ndarray, min_db: float, max_db: float) -> np.ndarray:
    db = np.random.uniform(min_db, max_db)
    gain = 10 ** (db / 20.0)
    return (y * gain).astype(np.float32)


def maybe_apply_augmentations(y: np.ndarray, sr: int, cfg: AugmentCfg) -> np.ndarray:
    if not cfg.enabled:
        return y
    out = y.copy()
    # Gaussian noise
    g = cfg.gaussian_noise
    if g.get('enabled', False) and random.random() < float(g.get('prob', 0.0)):
        out = aug_gaussian_noise(out, float(g.get('min_amplitude', 0.001)), float(g.get('max_amplitude', 0.015)))
    # Time stretch
    ts = cfg.time_stretch
    if ts.get('enabled', False) and random.random() < float(ts.get('prob', 0.0)):
        rate = np.random.uniform(float(ts.get('min_rate', 0.9)), float(ts.get('max_rate', 1.1)))
        out = aug_time_stretch(out, rate)
    # Pitch shift
    ps = cfg.pitch_shift
    if ps.get('enabled', False) and random.random() < float(ps.get('prob', 0.0)):
        semi = np.random.uniform(float(ps.get('min_semitones', -2)), float(ps.get('max_semitones', 2)))
        out = aug_pitch_shift(out, sr, semi)
    # Time shift
    sh = cfg.time_shift
    if sh.get('enabled', False) and random.random() < float(sh.get('prob', 0.0)):
        frac = np.random.uniform(float(sh.get('min_fraction', -0.3)), float(sh.get('max_fraction', 0.3)))
        out = aug_time_shift(out, frac)
    # Gain
    gn = cfg.gain
    if gn.get('enabled', False) and random.random() < float(gn.get('prob', 0.0)):
        out = aug_gain(out, float(gn.get('min_gain_db', -6)), float(gn.get('max_gain_db', 6)))
    return out.astype(np.float32)

# --------------------------
# Feature extraction
# --------------------------

def extract_classical_features(y: np.ndarray, sr: int, cfg: ClassicalCfg) -> np.ndarray:
    feats = []
    # MFCCs: mean and std across time
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length)
    feats.extend([mfcc.mean(axis=1), mfcc.std(axis=1)])

    if cfg.include_spectral:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        feats.extend([centroid.mean(axis=1), bandwidth.mean(axis=1), rolloff.mean(axis=1)])

    if cfg.include_temporal:
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=cfg.win_length, hop_length=cfg.hop_length)
        feats.append(zcr.mean(axis=1))

    if cfg.include_chroma:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        feats.extend([chroma.mean(axis=1), chroma.std(axis=1)])

    flat = np.concatenate(feats, axis=0).astype(np.float32)
    return flat


def extract_neural_features(y: np.ndarray, sr: int, cfg: NeuralCfg) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    mats = [S_db]
    if cfg.include_deltas:
        d1 = librosa.feature.delta(S_db)
        mats.append(d1.astype(np.float32))
        if cfg.include_delta_deltas:
            d2 = librosa.feature.delta(S_db, order=2)
            mats.append(d2.astype(np.float32))
    feat = np.stack(mats, axis=0)  # [C, n_mels, T]
    return feat

# --------------------------
# Processing pipeline
# --------------------------

def ensure_dirs(base: Path, classes: List[str]):
    (base / 'universal').mkdir(parents=True, exist_ok=True)
    (base / 'classical').mkdir(parents=True, exist_ok=True)
    (base / 'neural').mkdir(parents=True, exist_ok=True)
    for sub in ['universal', 'classical', 'neural']:
        for c in classes:
            (base / sub / c).mkdir(parents=True, exist_ok=True)


def save_universal(seg: np.ndarray, sr: int, out_path: Path):
    if HAS_SF:
        sf.write(str(out_path), seg.astype(np.float32), sr)
    else:
        write_wav_pcm16(out_path, sr, seg.astype(np.float32))



def save_classical(feat: np.ndarray, out_path: Path):
    np.save(out_path, feat)


def save_neural(feat: np.ndarray, out_path: Path):
    np.save(out_path, feat)


def process_file(path: Path, label: str, is_train: bool, cfg: PreprocessConfig,
                 do_universal: bool, do_classical: bool, do_neural: bool,
                 idx_counter: Dict[str,int]) -> List[Tuple[str, Path]]:
    # returns list of (kind, output_path) created
    created = []
    sr = cfg.data.sample_rate
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return created

    # Highpass and normalization before segmentation
    y = apply_highpass(y, sr, cfg.universal.highpass_cutoff)
    y = normalize_audio(y, cfg.universal.normalize_method)

    segs = segment_audio(y, sr, cfg.data.duration, cfg.data.overlap)
    base = path.stem

    for seg in segs:
        seg_out = seg
        # Augment only for train
        if is_train:
            seg_out = maybe_apply_augmentations(seg_out, sr, cfg.augment)
            # After augmentation, re-segment to ensure fixed length
            seg_out = segment_audio(seg_out, sr, cfg.data.duration, 0.0)[0]
            # And re-normalize
            seg_out = normalize_audio(seg_out, cfg.universal.normalize_method)

        seg_idx = idx_counter[label]
        idx_counter[label] += 1
        fname = f"{base}_seg{seg_idx:04d}"

        if do_universal:
            u_path = cfg.data.processed_dir / 'universal' / label / f"{fname}.wav"
            save_universal(seg_out, sr, u_path)
            created.append(('universal', u_path))

        if do_classical:
            c_feat = extract_classical_features(seg_out, sr, cfg.classical)
            c_path = cfg.data.processed_dir / 'classical' / label / f"{fname}.npy"
            save_classical(c_feat, c_path)
            created.append(('classical', c_path))

        if do_neural:
            n_feat = extract_neural_features(seg_out, sr, cfg.neural)
            n_path = cfg.data.processed_dir / 'neural' / label / f"{fname}.npy"
            save_neural(n_feat, n_path)
            created.append(('neural', n_path))

    return created


def write_classical_manifest(processed_dir: Path, classes: List[str]):
    # Build a simple CSV manifest for classical features
    out_csv = processed_dir / 'classical' / 'features.csv'
    rows = []
    for c in classes:
        for f in (processed_dir / 'classical' / c).glob('*.npy'):
            rows.append({'path': str(f), 'label': c})
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'label'])
        w.writeheader()
        w.writerows(rows)

# --------------------------
# Main
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['universal', 'classical', 'neural'], default=None,
                        help='Restrict to a single output type')
    parser.add_argument('--limit', type=int, default=None, help='Process only first N files (for debug)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for augmentations')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = get_config()
    ensure_dirs(cfg.data.processed_dir, cfg.data.classes)

    all_pairs, train_set = load_all_files_from_manifests(cfg.data.classes)
    if args.limit is not None:
        all_pairs = all_pairs[:args.limit]

    do_universal = (args.only is None) or (args.only == 'universal')
    do_classical = (args.only is None) or (args.only == 'classical')
    do_neural    = (args.only is None) or (args.only == 'neural')

    # Progress
    print(f"Files to process: {len(all_pairs)}")
    t0 = time.time()

    idx_counter = {c: 0 for c in cfg.data.classes}
    created_total = 0

    for i, (path, label) in enumerate(all_pairs, 1):
        is_train = path in train_set
        created = process_file(path, label, is_train, cfg, do_universal, do_classical, do_neural, idx_counter)
        created_total += len(created)
        if i % 50 == 0:
            dt = time.time() - t0
            print(f"Processed {i}/{len(all_pairs)} files, created {created_total} items in {dt:.1f}s")

    if do_classical:
        write_classical_manifest(cfg.data.processed_dir, cfg.data.classes)

    dt = time.time() - t0
    print(f"Done. Created {created_total} outputs in {dt:.1f}s")


if __name__ == '__main__':
    main()
