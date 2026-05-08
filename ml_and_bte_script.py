"""
Real-time exercise classifier: Bicep Curl vs Shoulder Press
Combines imu.py (serial reader + live plot) with classify.py (SVM model).

Usage:
  python realtime_classify.py
  python realtime_classify.py --port /dev/tty.usbserial-11130
  python realtime_classify.py --classify-window 4   # seconds of IMU data per prediction
"""

import argparse
import math
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
from scipy.signal import filtfilt, butter, find_peaks
from itertools import groupby

import asyncio
from bleak import BleakScanner, BleakClient

# ── Import shared logic from existing files ───────────────────────────────────
# classify.py must be in the same directory
from classify import (
    SENSOR_COLS, CLASS_NAMES, WINDOW_SEC,
    extract_features, load_dataset, train_final_model,
)
from imu import C, style_axes
# from imu import SerialReader, parse_line, find_port, style_axes, C

# ── Configuration ─────────────────────────────────────────────────────────────
BAUD_RATE        = 115200
PLOT_WINDOW_SEC  = 5      # seconds shown in the scrolling plot
CLASSIFY_WINDOW_SEC = 3   # must match training — imported from classify.py
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

# BTE CODE
DEVICE_NAME  = "ESP32_IMU"
CHAR_UUID    = "12345678-1234-1234-1234-123456789abd"
DEVICE_ADDRESS = "68:B6:B3:3E:11:22"

def parse_ble_line(line):
    # "AX:-0.070 AY:0.054 AZ:1.024 | GX:1.145 GY:-0.458 GZ:0.855 | T:24.86 C"
    try:
        parts = line.replace("|", "").split()
        values = {p.split(":")[0]: float(p.split(":")[1]) for p in parts if ":" in p}
        return (
            values["AX"], values["AY"], values["AZ"],
            values["GX"], values["GY"], values["GZ"],
            values["T"]
        )
    except (ValueError, IndexError, KeyError):
        return None
    

# REP COUNT PARAMETER CONFIGS
ALPHA = 0.8
ACC_CUTOFF = 2
GYRO_CUTOFF = 1
# MIN_REP_TIME = 0.05
AMPLITUDE_THRESHOLD = 36
# DX_EPS = 0.01
# GROUP_LEN = 4

SAMPLE_RATE = 20          # Hz
MIN_REP_SECONDS = 0.5     # fastest half-rep you'd ever do
PEAK_DISTANCE = 20  # = 10 samples
PEAK_PROMINENCE = 20      # tune to your signal's amplitude scale
PEAK_WIDTH = 2            # samples — filters noise spikes

def main():
    parser = argparse.ArgumentParser(description="Real-time exercise classifier")
    parser.add_argument("--port",             default=None,               help="Serial port")
    parser.add_argument("--baud",             default=BAUD_RATE, type=int)
    parser.add_argument("--plot-window",      default=PLOT_WINDOW_SEC,  type=float)
    parser.add_argument("--classify-window",  default=CLASSIFY_WINDOW_SEC, type=float)
    parser.add_argument("--outfile",          default="imu_log.txt")
    args = parser.parse_args()

    # # ── 1. Train model on existing files ──────────────────────────────────────
    print("Loading training data …")
    try:
        X, y, groups, filenames = load_dataset(DATA_DIR)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"Training SVM on {len(y)} files …")
    model = train_final_model(X, y)
    print("Model ready.\n")

    # # ── 2. Setup Bluetooth ────────────────────────────────────────────────
    # port     = args.port or find_port()
    # buf_size = int(args.plot_window * SAMPLE_HZ * 2)

    # print(f"[INFO] Opening {port} at {args.baud} baud …")
    # reader = SerialReader(port, args.baud, buf_size, outfile=args.outfile)
    # reader.start()
    buf_size = int(args.plot_window * SAMPLE_HZ * 2)
    ble_buffer = deque(maxlen=buf_size)
    async def bte():
        def on_notify(_handle, data: bytearray):
            imu_data = data.decode("utf-8", errors="replace")
            # print(imu_data)
            line = imu_data.strip()
            parsed = parse_ble_line(line)
            if parsed:
                ax, ay, az, gx, gy, gz, tmp = parsed
                ble_buffer.append((time.time(), ax, ay, az, gx, gy, gz, tmp))

            
        print("Connecting to ESP32_IMU ...")
        async with BleakClient(DEVICE_ADDRESS) as client:
            print("Connected!")
            await client.start_notify(CHAR_UUID, on_notify)
            print("Streaming — Ctrl-C to stop")
            await asyncio.sleep(3600)

    threading.Thread(target=lambda: asyncio.run(bte()), daemon=True).start()



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
    # ax_gyro  = fig.add_subplot(gs[2])
    # ax_temp  = fig.add_subplot(gs[3])

    # Label panel — plain text display, no axes chrome
    ax_label.set_facecolor(C["panel"])
    ax_label.set_xticks([]); ax_label.set_yticks([])
    for sp in ax_label.spines.values():
        sp.set_edgecolor(C["grid"])

    label_text = ax_label.text(
        0.2, 0.5, "…",
        transform=ax_label.transAxes,
        ha="center", va="center",
        fontsize=22, fontweight="bold",
        color=LABEL_COLORS["…"],
    )
    label_text_2 = ax_label.text(
        0.8, 0.5, "…",
        transform=ax_label.transAxes,
        ha="center", va="center",
        fontsize=22, fontweight="bold",
        color=LABEL_COLORS["…"],
    )
    prob_text = ax_label.text(
        0.5, 0.05, "",
        transform=ax_label.transAxes,
        ha="center", va="bottom",
        fontsize=8, color="#aaaaaa",
    )
    ax_label.set_title("Shake wrist to start classifying", color=C["text"], fontsize=9, pad=3)

    style_axes(ax_acc,  "Acceleration Magnitude")
    # style_axes(ax_gyro, "Acc + Gyro Mag")
    # style_axes(ax_temp, "Temperature (°C)")
    # ax_temp.set_xlabel("Time (s)", fontsize=9)

    # status_txt = fig.text(0.5, 0.003, reader.status,
    #                       ha="center", fontsize=8, color="#888888")

    # Animated line handles
    # la_x, = ax_acc.plot([], [], color=C["ax"], lw=1.2, label="Accel X")
    # la_y, = ax_acc.plot([], [], color=C["ay"], lw=1.2, label="Accel Y")
    # la_z, = ax_acc.plot([], [], color=C["az"], lw=1.2, label="Accel Z")
    # lg_x, = ax_gyro.plot([], [], color=C["gx"], lw=1.2, label="Gyro X")
    # lg_y, = ax_gyro.plot([], [], color=C["gy"], lw=1.2, label="Gyro Y")
    # lg_z, = ax_gyro.plot([], [], color=C["gz"], lw=1.2, label="Gyro Z")
    # lt,   = ax_temp.plot([], [], color=C["temp"], lw=1.5, label="Temperature")
    la_mag, = ax_acc.plot([], [], color=C["ax"], lw=1.2, label="Accel Mag")

    for a, lines in [(ax_acc,  [la_mag])]:
                #  (ax_gyro, [lg_x, lg_y, lg_z]),
                #  (ax_temp, [lt])]:
        a.legend(handles=lines, loc="upper left", fontsize=7,
                 facecolor=C["panel"], edgecolor=C["grid"],
                 labelcolor=C["text"])

    # Live readout annotations
    def readout(ax, y, text=""):
        return ax.text(1.001, y, text, transform=ax.transAxes,
                       color=C["text"], fontsize=7.5, va="center",
                       fontfamily="monospace")

    ro_ax = readout(ax_acc,  0.83); ro_ay = readout(ax_acc,  0.50)
    # ro_az = readout(ax_acc,  0.17); ro_gx = readout(ax_gyro, 0.83)
    # ro_gy = readout(ax_gyro, 0.50); ro_gz = readout(ax_gyro, 0.17)
    # ro_t  = readout(ax_temp, 0.50)
    
    # REP COUNT PROCESS
    rep_count = [0.0]
    last_processed_t = [0.0]   # mutable container so the closure can update it
    last_rep_checked = [0.0]
    last_rest_recorded = [0.0]
    rest_time = [0.0]
    is_rest = [False]
    

    def lowpass(data, cutoff_hz=3, sample_hz=100, order=4):
        nyq = sample_hz / 2
        b, a = butter(order, cutoff_hz / nyq, btype='low')
        return filtfilt(b, a, data)
    
    def calc_mag(axv, ayv, azv, gxv,gyv,gzv, alpha=0.5):
        a_mag = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(axv, ayv, azv)]
        g_mag = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(gxv, gyv, gzv)]
        
        try:
            magv_passed = lowpass(a_mag, cutoff_hz=ACC_CUTOFF)
        except Exception as e:
            print(f"filter error: {e}")
            magv_passed = a_mag  # fall back to unfiltered
        try:
            gmag_filtered = lowpass(g_mag, cutoff_hz=GYRO_CUTOFF)
        except Exception as e:
            print(f"filter error: {e}")
            gmag_filtered = g_mag  # fall back to unfiltered

        combined = alpha*np.array(magv_passed) + (1-alpha)*np.array(gmag_filtered)
        # print(f'array size: {len(combined)}')
        return combined
    
    # def check_rep_state(signal, tv, rep_count):

    #     # cooldown
    #     if tv[-1] - last_rep_checked[0] < MIN_REP_TIME:
    #         return
        
    #     # get unseen sample window to assess (1 update has 5 samples to assess with an interval of 50 which is 20 Hz)
    #     new_samples = [(ti, mi) for ti, mi in zip(tv, signal) if ti > last_processed_t[0]]
    #     if len(new_samples) < 2:
    #         return
    #     last_processed_t[0] = tv[-1]

    #     # take derivative of sample window
    #     new_vals = [mi for ti, mi in new_samples]
    #     if max(new_vals) < AMPLITUDE_THRESHOLD:
    #         return  # signal too flat, skip this window
    #     diffs = [new_vals[i] - new_vals[i-1] for i in range(1, len(new_vals))]

    #     # need to find a way to count number of peaks
    #     def sign(x, epsilon=DX_EPS):
    #         if x > epsilon: return 1
    #         if x < -epsilon: return -1
    #         return 0  # dead zone
        
    #     signs = [sign(d) for d in diffs]

    #     # then look for sign changes in the new array
    #     groups = [(k, len(list(v))) for k, v in groupby(signs)]
    #     transitions = sum(1 for k, length in groups if k == 0 and length >= GROUP_LEN)
    #     rep_count[0] += transitions / 2  # every two zero crossings = one rep
    #     label_text_2.set_text(rep_count[0])        
    #     last_rep_checked[0] = tv[-1]

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
    # Accumulate ALL samples in a rolling buffer instead of windowed chunks
    signal_buffer = []
    time_buffer = []
    prev_buffer_peak_count = [0.0]
    peak_count_offset = [0.0]
    counted_peak_times = set()


    def check_rep_state(signal, tv, rep_count):
        # Append new samples to the rolling buffer
        new_samples = [(ti, mi) for ti, mi in zip(tv, signal) if ti > last_processed_t[0]]
        if not new_samples:
            return
        last_processed_t[0] = tv[-1]

        times, vals = zip(*new_samples)
        signal_buffer.extend(vals)
        time_buffer.extend(times)

        # Keep buffer from growing forever — hold last 10 seconds of data
        max_samples = SAMPLE_RATE * 10
        if len(signal_buffer) > max_samples:
            signal_buffer[:] = signal_buffer[-max_samples:]
            time_buffer[:] = time_buffer[-max_samples:]
        print(f"min: {min(signal_buffer):.2f}  max: {max(signal_buffer):.2f}  range: {max(signal_buffer) - min(signal_buffer):.2f}")

        # Need enough data for find_peaks to be meaningful
        if len(signal_buffer) < PEAK_DISTANCE * 2:
            return

        if max(signal_buffer) < AMPLITUDE_THRESHOLD:
            return  # signal too flat, user not moving
        arr = np.array(signal_buffer)
        peaks, _ = find_peaks(arr, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE, width=PEAK_WIDTH)
        # current_peak_count = len(peaks)

        # # If the buffer just trimmed old data, peaks in the window dropped.
        # # Save whatever was lost into the offset before updating.
        # if current_peak_count < prev_buffer_peak_count[0]:
        #     peak_count_offset[0] += prev_buffer_peak_count[0]

        # prev_buffer_peak_count[0] = current_peak_count
        # rep_count[0] = (peak_count_offset[0] + current_peak_count) / 2
        for p in peaks:
            peak_time = round(time_buffer[p] * 5) / 5  # snap to 0.2s buckets
            if peak_time not in counted_peak_times:
                counted_peak_times.add(peak_time)
                rep_count[0] += 0.5
                rest_time[0] = 0
        label_text_2.set_text(f'reps: {rep_count[0]}')

    def update_rest(signal, tv):
        label_text.set_text(f'rest time: {rest_time[0]:1.0f} s')
        time_diff = tv[-1] - last_rest_recorded[0]

        if rest_time[0] > 3:
            rep_count[0] = 0

        if abs(signal[-1]) <= AMPLITUDE_THRESHOLD and last_rest_recorded[0] != 0:
            if not is_rest[0]: is_rest[0] = True
            rest_time[0] = rest_time[0] + time_diff
        else:
            is_rest[0] = False
            rest_time[0] = 0.0

        last_rest_recorded[0] = tv[-1]


    def update(_):
        # t, ax_, ay_, az_, gx_, gy_, gz_, tmp = reader.snapshot()
        data = list(ble_buffer)
        if len(data) < 2:
            return
        t, ax_, ay_, az_, gx_, gy_, gz_, tmp = zip(*data)
        t, ax_, ay_, az_, gx_, gy_, gz_, tmp = list(t), list(ax_), list(ay_), list(az_), list(gx_), list(gy_), list(gz_), list(tmp)

        t_now = t[-1]
        t_lo  = t_now - args.plot_window

        def trim(xs):
            return [x for x, ti in zip(xs, t) if ti >= t_lo]

        tv   = [ti for ti in t if ti >= t_lo]
        axv  = trim(ax_); ayv = trim(ay_); azv = trim(az_)
        gxv  = trim(gx_); gyv = trim(gy_); gzv = trim(gz_)
        tmpv = trim(tmp)

        # REP COUNT PROCESS
        f = calc_mag(axv, ayv, azv, gxv, gyv, gzv, alpha=ALPHA)

        la_mag.set_data(tv, f)
        # lg_x.set_data(tv, gxv) #; lg_y.set_data(tv, gyv); lg_z.set_data(tv, gzv)
        # lt.set_data(tv, tmpv)

        # for a in (ax_acc, ax_gyro, ax_temp):
        #     a.set_xlim(t_lo, t_now)
        ax_acc.set_xlim(t_lo, t_now)

        def auto_ylim(ax, *series):
            vals = [v for s in series for v in s]
            if vals:
                lo, hi = min(vals), max(vals)
                pad = max((hi - lo) * 0.15, 0.05)
                ax.set_ylim(lo - pad, hi + pad)

        auto_ylim(ax_acc,  f)
        # ax_acc.set_ylim(0,2)
        # auto_ylim(ax_gyro, gxv, gyv, gzv)
        # auto_ylim(ax_temp, tmpv)

        if ax_:
            mag_now = math.sqrt(ax_[-1]**2 + ay_[-1]**2 + az_[-1]**2)
            ro_ax.set_text(f"MAG {mag_now:+7.3f}")
            ro_ay.set_text("")
            # ro_az.set_text("")
            # ro_gx.set_text(f"GX {gx_[-1]:+7.2f}")
            # ro_gy.set_text(f"GY {gy_[-1]:+7.2f}")
            # ro_gz.set_text(f"GZ {gz_[-1]:+7.2f}")
            # ro_t.set_text(f"{tmp[-1]:.2f} °C")
        
# ── Shake to toggle classification on/off ────────────────────────────
        now_ms = time.perf_counter() * 1000
        if detect_shake(t, ax_, ay_, az_) and now_ms - last_shake_ms[0] >= SHAKE_COOLDOWN_MS:
            last_shake_ms[0] = now_ms
            classifying[0] = not classifying[0]
            if classifying[0]:
                ax_label.set_title("…")
                # label_text.set_fontsize(22)
                ax_label.set_facecolor(LABEL_COLORS["…"])
                prob_text.set_text("")
                print("[INFO] Shake detected — classification started.")
            else:
                ax_label.set_title("Paused — shake to resume")
                # label_text.set_fontsize(16)
                ax_label.set_facecolor(LABEL_COLORS["…"])
                prob_text.set_text("")
                print("[INFO] Shake detected — classification paused.")

        # ── Classify at most every CLASSIFY_EVERY_MS ms ───────────────────────
        if classifying[0] and now_ms - last_classify_ms[0] >= CLASSIFY_EVERY_MS:
            last_classify_ms[0] = now_ms
            result = classify_snapshot(
                model, t, ax_, ay_, az_, gx_, gy_, gz_,
                args.classify_window,
            )
            # print(result)
            if result is not None:
                exercise, probs = result
                ax_label.set_title(f'Detected Exercise: {exercise}')
                # label_text.set_fontsize(22)
                ax_label.set_facecolor(LABEL_COLORS.get(exercise, C["text"]))
                prob_text.set_text(
                    "   ".join(f"{CLASS_NAMES[i]} {probs[i]:.0%}" for i in range(len(CLASS_NAMES)))
                )

                update_rest(f,tv)
                check_rep_state(f, tv, rep_count)


        # status_txt.set_text(reader.status)
    
    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
