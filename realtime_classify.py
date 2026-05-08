"""
Real-time exercise classifier: Bicep Curl, Shoulder Press, Lateral Raise, Chest Fly, Row, Tricep Pushdown, Squat
Combines imu.py (serial reader + live plot) with classify.py (SVM model).

Usage:
  python realtime_classify.py
  python realtime_classify.py --port /dev/tty.usbserial-11130
  python realtime_classify.py --classify-window 4   # seconds of IMU data per prediction
"""

import argparse
import sys
import time
import threading
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import serial
import serial.tools.list_ports

# ── Import shared logic from existing files ───────────────────────────────────
# classify.py must be in the same directory
from classify import (
    SENSOR_COLS, CLASS_NAMES, WINDOW_SEC,
    extract_features, load_dataset, train_final_model,
)
from imu import SerialReader, parse_line, find_port, style_axes, C

# ── Configuration ─────────────────────────────────────────────────────────────
BAUD_RATE        = 115200
PLOT_WINDOW_SEC  = 5      # seconds shown in the scrolling plot
CLASSIFY_WINDOW_SEC = WINDOW_SEC   # must match training — imported from classify.py
CLASSIFY_EVERY_MS   = 1500  # run classifier at most this often (ms)
MIN_SAMPLES      = 50     # skip prediction if fewer samples in window
DATA_DIR         = "."    # folder with training .txt files
SAMPLE_HZ        = 100

# ── Shake-to-start configuration ──────────────────────────────────────────────
SHAKE_THRESHOLD  = 2.5    # g — acceleration magnitude to trigger classification
SHAKE_MIN_HITS   = 3      # number of samples above threshold within the window
SHAKE_WINDOW_SEC = 0.5    # seconds of recent samples to scan for shake


# ── Colours ───────────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "Bicep Curl":      "#4fc3f7",
    "Shoulder Press":  "#f48fb1",
    "Lateral Raise":   "#a5d6a7",
    "Chest Fly":       "#ffcc80",
    "Row":             "#ce93d8",
    "Tricep Pushdown": "#ef9a9a",
    "Squat":           "#80cbc4",
    "…":               "#888888",
}


def build_dataframe(t, ax, ay, az, gx, gy, gz) -> pd.DataFrame:
    """Build a DataFrame from raw buffer lists, matching classify.py's format."""
    return pd.DataFrame({
        "time_sec": t,
        "ax": ax, "ay": ay, "az": az,
        "gx": gx, "gy": gy, "gz": gz,
    })


def classify_snapshot(model, t, ax, ay, az, gx, gy, gz, window_sec: float):
    """
    Take the most recent `window_sec` of buffered data, extract features,
    and return (label_str, prob_bicep, prob_shoulder).
    Returns None if there are too few samples.
    """
    if len(t) < 2:
        return None

    t_cutoff = t[-1] - window_sec
    idx = [i for i, ti in enumerate(t) if ti >= t_cutoff]

    if len(idx) < MIN_SAMPLES:
        return None

    sl = slice(idx[0], idx[-1] + 1)
    df = build_dataframe(
        t[sl], ax[sl], ay[sl], az[sl],
        gx[sl], gy[sl], gz[sl],
    )

    features = extract_features(df).reshape(1, -1)
    pred  = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    return CLASS_NAMES[pred], probs


def main():
    parser = argparse.ArgumentParser(description="Real-time exercise classifier")
    parser.add_argument("--port",             default=None,               help="Serial port")
    parser.add_argument("--baud",             default=BAUD_RATE, type=int)
    parser.add_argument("--plot-window",      default=PLOT_WINDOW_SEC,  type=float)
    parser.add_argument("--classify-window",  default=CLASSIFY_WINDOW_SEC, type=float)
    parser.add_argument("--outfile",          default="imu_log.txt")
    args = parser.parse_args()

    # ── 1. Train model on existing files ──────────────────────────────────────
    print("Loading training data …")
    try:
        X, y, groups, filenames = load_dataset(DATA_DIR)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"Training SVM on {len(y)} files …")
    model = train_final_model(X, y)
    print("Model ready.\n")

    # ── 2. Start serial reader ────────────────────────────────────────────────
    port     = args.port or find_port()
    buf_size = int(args.plot_window * SAMPLE_HZ * 2)

    print(f"[INFO] Opening {port} at {args.baud} baud …")
    reader = SerialReader(port, args.baud, buf_size, outfile=args.outfile)
    reader.start()

    # ── 3. Build figure ───────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 9), facecolor=C["bg"])
    fig.canvas.manager.set_window_title("Exercise Classifier — Real-Time")

    # Extra row at top for the classification label
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.5,
                           left=0.08, right=0.97, top=0.93, bottom=0.07,
                           height_ratios=[0.8, 2, 2, 1.2])

    ax_label = fig.add_subplot(gs[0])
    ax_acc   = fig.add_subplot(gs[1])
    ax_gyro  = fig.add_subplot(gs[2])
    ax_temp  = fig.add_subplot(gs[3])

    # Label panel — plain text display, no axes chrome
    ax_label.set_facecolor(C["panel"])
    ax_label.set_xticks([]); ax_label.set_yticks([])
    for sp in ax_label.spines.values():
        sp.set_edgecolor(C["grid"])

    label_text = ax_label.text(
        0.5, 0.5, "Shake wrist to start classifying",
        transform=ax_label.transAxes,
        ha="center", va="center",
        fontsize=16, fontweight="bold",
        color=LABEL_COLORS["…"],
    )
    prob_text = ax_label.text(
        0.5, 0.05, "",
        transform=ax_label.transAxes,
        ha="center", va="bottom",
        fontsize=8, color="#aaaaaa",
    )
    ax_label.set_title("Detected Exercise", color=C["text"], fontsize=9, pad=3)

    style_axes(ax_acc,  "Acceleration (g)")
    style_axes(ax_gyro, "Angular rate (°/s)")
    style_axes(ax_temp, "Temperature (°C)")
    ax_temp.set_xlabel("Time (s)", fontsize=9)

    status_txt = fig.text(0.5, 0.003, reader.status,
                          ha="center", fontsize=8, color="#888888")

    # Animated line handles
    la_x, = ax_acc.plot([], [], color=C["ax"], lw=1.2, label="Accel X")
    la_y, = ax_acc.plot([], [], color=C["ay"], lw=1.2, label="Accel Y")
    la_z, = ax_acc.plot([], [], color=C["az"], lw=1.2, label="Accel Z")
    lg_x, = ax_gyro.plot([], [], color=C["gx"], lw=1.2, label="Gyro X")
    lg_y, = ax_gyro.plot([], [], color=C["gy"], lw=1.2, label="Gyro Y")
    lg_z, = ax_gyro.plot([], [], color=C["gz"], lw=1.2, label="Gyro Z")
    lt,   = ax_temp.plot([], [], color=C["temp"], lw=1.5, label="Temperature")

    for a, lines in [(ax_acc,  [la_x, la_y, la_z]),
                     (ax_gyro, [lg_x, lg_y, lg_z]),
                     (ax_temp, [lt])]:
        a.legend(handles=lines, loc="upper left", fontsize=7,
                 facecolor=C["panel"], edgecolor=C["grid"],
                 labelcolor=C["text"])

    # Live readout annotations
    def readout(ax, y, text=""):
        return ax.text(1.001, y, text, transform=ax.transAxes,
                       color=C["text"], fontsize=7.5, va="center",
                       fontfamily="monospace")

    ro_ax = readout(ax_acc,  0.83); ro_ay = readout(ax_acc,  0.50)
    ro_az = readout(ax_acc,  0.17); ro_gx = readout(ax_gyro, 0.83)
    ro_gy = readout(ax_gyro, 0.50); ro_gz = readout(ax_gyro, 0.17)
    ro_t  = readout(ax_temp, 0.50)

    # ── 4. Animation update callback ──────────────────────────────────────────
    last_classify_ms  = [0.0]   # mutable container so the closure can update it
    classifying       = [False]  # toggled on/off by shake gesture
    last_shake_ms     = [-2000.0]  # cooldown — prevents double-trigger from one shake
    SHAKE_COOLDOWN_MS = 2000

    def detect_shake(t, ax_, ay_, az_):
        """Return True if a shake is detected in the most recent SHAKE_WINDOW_SEC."""
        if len(t) < 2:
            return False
        t_cutoff = t[-1] - SHAKE_WINDOW_SEC
        hits = sum(
            1 for i, ti in enumerate(t)
            if ti >= t_cutoff and
            (ax_[i]**2 + ay_[i]**2 + az_[i]**2) ** 0.5 > SHAKE_THRESHOLD
        )
        return hits >= SHAKE_MIN_HITS

    def update(_):
        t, ax_, ay_, az_, gx_, gy_, gz_, tmp = reader.snapshot()

        if len(t) < 2:
            return

        t_now = t[-1]
        t_lo  = t_now - args.plot_window

        def trim(xs):
            return [x for x, ti in zip(xs, t) if ti >= t_lo]

        tv   = [ti for ti in t if ti >= t_lo]
        axv  = trim(ax_); ayv = trim(ay_); azv = trim(az_)
        gxv  = trim(gx_); gyv = trim(gy_); gzv = trim(gz_)
        tmpv = trim(tmp)

        la_x.set_data(tv, axv); la_y.set_data(tv, ayv); la_z.set_data(tv, azv)
        lg_x.set_data(tv, gxv); lg_y.set_data(tv, gyv); lg_z.set_data(tv, gzv)
        lt.set_data(tv, tmpv)

        for a in (ax_acc, ax_gyro, ax_temp):
            a.set_xlim(t_lo, t_now)

        def auto_ylim(ax, *series):
            vals = [v for s in series for v in s]
            if vals:
                lo, hi = min(vals), max(vals)
                pad = max((hi - lo) * 0.15, 0.05)
                ax.set_ylim(lo - pad, hi + pad)

        auto_ylim(ax_acc,  axv, ayv, azv)
        auto_ylim(ax_gyro, gxv, gyv, gzv)
        auto_ylim(ax_temp, tmpv)

        if ax_:
            ro_ax.set_text(f"AX {ax_[-1]:+7.3f}")
            ro_ay.set_text(f"AY {ay_[-1]:+7.3f}")
            ro_az.set_text(f"AZ {az_[-1]:+7.3f}")
            ro_gx.set_text(f"GX {gx_[-1]:+7.2f}")
            ro_gy.set_text(f"GY {gy_[-1]:+7.2f}")
            ro_gz.set_text(f"GZ {gz_[-1]:+7.2f}")
            ro_t.set_text(f"{tmp[-1]:.2f} °C")

        # ── Shake to toggle classification on/off ────────────────────────────
        now_ms = time.perf_counter() * 1000
        if detect_shake(t, ax_, ay_, az_) and now_ms - last_shake_ms[0] >= SHAKE_COOLDOWN_MS:
            last_shake_ms[0] = now_ms
            classifying[0] = not classifying[0]
            if classifying[0]:
                label_text.set_text("…")
                label_text.set_fontsize(22)
                label_text.set_color(LABEL_COLORS["…"])
                prob_text.set_text("")
                print("[INFO] Shake detected — classification started.")
            else:
                label_text.set_text("Paused — shake to resume")
                label_text.set_fontsize(16)
                label_text.set_color(LABEL_COLORS["…"])
                prob_text.set_text("")
                print("[INFO] Shake detected — classification paused.")

        # ── Classify at most every CLASSIFY_EVERY_MS ms ───────────────────────
        if classifying[0] and now_ms - last_classify_ms[0] >= CLASSIFY_EVERY_MS:
            last_classify_ms[0] = now_ms
            result = classify_snapshot(
                model, t, ax_, ay_, az_, gx_, gy_, gz_,
                args.classify_window,
            )
            if result is not None:
                exercise, probs = result
                label_text.set_text(exercise)
                label_text.set_fontsize(22)
                label_text.set_color(LABEL_COLORS.get(exercise, C["text"]))
                prob_text.set_text(
                    "   ".join(f"{CLASS_NAMES[i]} {probs[i]:.0%}" for i in range(len(CLASS_NAMES)))
                )

        status_txt.set_text(reader.status)

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
