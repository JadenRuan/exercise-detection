"""
Exercise Classifier: Bicep Curl vs Shoulder Press
Using SVM on accelerometer + gyroscope data

Assumes:
  - Training files live in class-named subfolders: bicep_curl/, shoulder_press/
  - Format: CSV with header → time_sec, ax, ay, az, gx, gy, gz, temp_c
  - Each file = one full recording session (multiple reps)

Each file is sliced into fixed-length windows that match the real-time
inference window so the training distribution is identical to inference.

Usage:
  pip install numpy pandas scikit-learn matplotlib seaborn
  python classify.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

DATA_DIR    = "."     # root folder; training files live in CLASS_MAP-named subfolders
SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]

# MUST match CLASSIFY_WINDOW_SEC in realtime_classify.py
WINDOW_SEC  = 3.0    # length of each training/inference window in seconds
STEP_SEC    = 0.5    # sliding step — smaller = more windows = more training samples
MIN_SAMPLES = 50     # discard windows with fewer samples than this

CLASS_MAP = {
    "bicep_curl":      0,
    "shoulder_press":  1,
    "lateral_raise":   2,
    "chest_fly":       3,
    "row":             4,
    "tricep_pushdown": 5,
    "squat":           6,
}
CLASS_NAMES = ["Bicep Curl", "Shoulder Press", "Lateral Raise", "Chest Fly", "Row", "Tricep Pushdown", "Squat"]


# ─────────────────────────────────────────────
# 2. WINDOWING
# ─────────────────────────────────────────────

def segment_windows(df: pd.DataFrame, window_sec: float, step_sec: float):
    """
    Slice df into fixed-length overlapping windows.
    Uses time_sec column if present, otherwise assumes constant 100 Hz.
    Returns a list of DataFrames, one per window.
    """
    if "time_sec" in df.columns:
        t0 = df["time_sec"].iloc[0]
        t_end = df["time_sec"].iloc[-1]
        windows = []
        t = t0
        while t + window_sec <= t_end:
            mask = (df["time_sec"] >= t) & (df["time_sec"] < t + window_sec)
            win = df[mask].reset_index(drop=True)
            if len(win) >= MIN_SAMPLES:
                windows.append(win)
            t += step_sec
        return windows
    else:
        # Fallback: sample-based at assumed 100 Hz
        hz    = 100
        win_n  = int(window_sec * hz)
        step_n = int(step_sec   * hz)
        return [
            df.iloc[i : i + win_n].reset_index(drop=True)
            for i in range(0, len(df) - win_n + 1, step_n)
            if win_n >= MIN_SAMPLES
        ]


# ─────────────────────────────────────────────
# 3. FEATURE EXTRACTION
#    18 stats × 6 channels = 108 features per window
#    All features are normalised so short and long windows are comparable.
# ─────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Extract statistical features from one window DataFrame."""
    features = []
    for col in SENSOR_COLS:
        signal = df[col].values
        n = len(signal)

        mean   = np.mean(signal)
        std    = np.std(signal)
        vari   = np.var(signal)
        mini   = np.min(signal)
        maxi   = np.max(signal)
        rang   = maxi - mini
        median = np.median(signal)
        p25    = np.percentile(signal, 25)
        p75    = np.percentile(signal, 75)
        iqr    = p75 - p25
        rms    = np.sqrt(np.mean(signal ** 2))
        energy = np.mean(signal ** 2)              # mean(x²) = length-independent

        skew   = pd.Series(signal).skew()
        kurt   = pd.Series(signal).kurtosis()

        # Zero-crossing rate (already per-sample)
        zcr    = np.sum(np.diff(np.sign(signal)) != 0) / n

        # Signal magnitude area (already per-sample)
        sma    = np.sum(np.abs(signal)) / n

        # Peak rate (per-sample so short/long windows are comparable)
        peaks  = np.sum(
            (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
        ) / n

        # Dominant frequency magnitude (normalised by N so it doesn't scale with length)
        fft_vals       = np.abs(np.fft.rfft(signal)) / n
        dom_freq_idx   = np.argmax(fft_vals[1:]) + 1
        dom_freq_power = fft_vals[dom_freq_idx]

        features.extend([
            mean, std, vari, mini, maxi, rang,
            median, p25, p75, iqr,
            rms, energy, skew, kurt,
            zcr, sma, peaks, dom_freq_power,
        ])

    return np.array(features)


# ─────────────────────────────────────────────
# 4. LOAD ALL FILES — produce one sample per window, not per file
# ─────────────────────────────────────────────

def load_dataset(data_dir: str):
    """
    Returns:
      X         — (N_windows, n_features)
      y         — (N_windows,)  class labels
      groups    — (N_windows,)  source-file index (used by GroupKFold)
      filenames — list of source file names, indexed by groups
    """
    X, y, groups, filenames = [], [], [], []
    file_idx = 0

    all_files = []
    for keyword, class_id in CLASS_MAP.items():
        subdir  = os.path.join(data_dir, keyword)
        matches = sorted(glob.glob(os.path.join(subdir, "*.txt")))
        all_files.extend((fpath, class_id) for fpath in matches)

    if not all_files:
        raise FileNotFoundError(
            f"No .txt files found under '{data_dir}'. "
            f"Expected subfolders: {list(CLASS_MAP.keys())}"
        )

    for fpath, label in all_files:
        fname = os.path.basename(fpath).lower()

        try:
            df = pd.read_csv(fpath, sep=None, engine="python")
            df.columns = df.columns.str.strip()
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=SENSOR_COLS)
            if df.empty:
                print(f"  [ERROR] No valid rows in {fname}")
                continue
        except Exception as e:
            print(f"  [ERROR] Failed to read {fname}: {e}")
            continue

        missing = [c for c in SENSOR_COLS if c not in df.columns]
        if missing:
            print(f"  [ERROR] Missing columns {missing} in {fname}")
            continue

        windows = segment_windows(df, WINDOW_SEC, STEP_SEC)
        if not windows:
            print(f"  [WARN]  No windows extracted from {fname} "
                  f"(file may be shorter than {WINDOW_SEC}s)")
            continue

        for win in windows:
            X.append(extract_features(win))
            y.append(label)
            groups.append(file_idx)

        filenames.append(fname)
        print(f"  Loaded: {fname}  →  {CLASS_NAMES[label]}  "
              f"rows={len(df)}  windows={len(windows)}")
        file_idx += 1

    return np.array(X), np.array(y), np.array(groups), filenames


# ─────────────────────────────────────────────
# 5. EVALUATE WITH GROUP K-FOLD
#    GroupKFold ensures all windows from the same file stay in the same fold,
#    preventing data leakage from overlapping windows.
# ─────────────────────────────────────────────

def evaluate(X, y, groups, filenames):
    n_files  = len(filenames)
    n_splits = min(n_files, 5)          # at most 5 folds, at most one per file
    cv       = GroupKFold(n_splits=n_splits)

    print(f"\n{'─'*50}")
    print(f"GroupKFold CV  ({n_splits} folds, {n_files} source files, "
          f"{len(y)} windows total)")
    print(f"{'─'*50}")

    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", probability=True, random_state=42)),
    ])

    y_pred = cross_val_predict(svm_pipe, X, y, groups=groups, cv=cv)

    acc = accuracy_score(y, y_pred)
    print(f"\nOverall Accuracy: {acc:.1%}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES))

    return y, y_pred, acc


# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, acc):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix  (Accuracy: {acc:.1%})")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nSaved: confusion_matrix.png")
    plt.show()


def plot_feature_importance(X, y):
    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(X)

    stat_names = [
        "mean", "std", "var", "min", "max", "range",
        "median", "p25", "p75", "iqr",
        "rms", "energy", "skew", "kurt",
        "zcr", "sma", "peaks", "dom_freq",
    ]
    n_stats      = len(stat_names)
    feature_names = [f"{col}_{s}" for col in SENSOR_COLS for s in stat_names]

    subset_idx   = [i * n_stats for i in range(len(SENSOR_COLS))]  # 'mean' per channel
    subset_names = [feature_names[i] for i in subset_idx]

    fig, ax = plt.subplots(figsize=(9, 4))
    x, width = np.arange(len(subset_idx)), 0.35

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        vals = X_scaled[y == cls_id][:, subset_idx]
        ax.bar(x + cls_id * width, vals.mean(axis=0),
               width, yerr=vals.std(axis=0),
               label=cls_name, alpha=0.8, capsize=4)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(subset_names, rotation=30, ha="right")
    ax.set_ylabel("Scaled value (mean ± std across windows)")
    ax.set_title("Mean Accelerometer/Gyroscope Feature Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig("feature_comparison.png", dpi=150)
    print("Saved: feature_comparison.png")
    plt.show()


# ─────────────────────────────────────────────
# 7. TRAIN FINAL MODEL (on all windows)
# ─────────────────────────────────────────────

def train_final_model(X, y):
    """Train on all windows — this is the model used for real-time inference."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", probability=True, random_state=42)),
    ])
    pipe.fit(X, y)
    print("\nFinal model trained on all windows and ready for inference.")
    return pipe


def predict_new_file(model, fpath: str):
    """Predict every window in a new file and report the majority vote."""
    df = pd.read_csv(fpath, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=SENSOR_COLS)

    windows = segment_windows(df, WINDOW_SEC, STEP_SEC)
    if not windows:
        print(f"[WARN] No windows in {fpath}")
        return None

    preds = [model.predict(extract_features(w).reshape(1, -1))[0] for w in windows]
    vote  = max(set(preds), key=preds.count)
    votes_str = "  |  ".join(f"{CLASS_NAMES[i]}={preds.count(i)}" for i in range(len(CLASS_NAMES)))
    print(f"\nPrediction for '{os.path.basename(fpath)}':")
    print(f"  Windows: {len(preds)}  |  Votes: {votes_str}")
    print(f"  → {CLASS_NAMES[vote]}")
    return CLASS_NAMES[vote]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Exercise Classifier: SVM on IMU Windows")
    print("=" * 50)
    print(f"  Window: {WINDOW_SEC}s   Step: {STEP_SEC}s\n")

    X, y, groups, filenames = load_dataset(DATA_DIR)

    print(f"\nDataset: {len(filenames)} files  →  {len(y)} windows total")
    print(f"  Bicep Curl: {(y==0).sum()}  |  Shoulder Press: {(y==1).sum()}")
    print(f"  Feature vector size: {X.shape[1]}")

    y_true, y_pred, acc = evaluate(X, y, groups, filenames)

    plot_confusion_matrix(y_true, y_pred, acc)
    plot_feature_importance(X, y)

    final_model = train_final_model(X, y)