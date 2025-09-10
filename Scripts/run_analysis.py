"""
Visualize SMPL (24-joint) skeletons and measure height, wrist distance,
and walking displacement from start to end (with pause detection).
What it does:
1) Loads frames from JSON.
2) Plots skeletons (optional: show/save PNGs).
3) Detects start & end pauses (5s stand still) OR uses manual frame indices.
4) Computes:
   - Median height
   - Median wrist-to-wrist distance
   - Walking distance (3D + ground-plane)
   - Distance/Height ratio
   - Error vs tape measurement (if provided)
   - Error vs true height (calibration)
5) Saves results to skeleton_measurements.csv
"""


import json
import os
import csv
from pathlib import Path
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from measure import REF_JOINT_MODE  # noqa: F401

# ========= USER SETTINGS =========
JSON_PATH = r"C:\Users\vaish\Documents\VaishDoc\Learning\Dessertation\ml-comotion\results\3m_towards_camera_run1.json"
OUT_DIR   = "skeleton_outputs"

SHOW      = True
SAVE      = True
MAX_FRAMES = None
FIGSIZE = (7, 7)
DPI = 160

# --- Experiment settings ---
FPS = 5
AUTO_TRIM = True
MANUAL_START_FRAME = None
MANUAL_END_FRAME = None

PAUSE_SECONDS = 5.0
STILL_SPEED_THR_NORMALIZED = 0.03
SMOOTH_WINDOW = 3

# Coordinate assumptions
VERTICAL_AXIS = 1
GROUND_AXES = (0, 2)

# Plotting/view options
FRONT_VIEW = True
CENTER_PELVIS = True

RESULTS_CSV = "skeleton_measurements.csv"
TRIAL_ID = "3m_towards_camera_run1"

# Optional ground-truth
TAPE_DISTANCE_M = 3.0
TRUE_HEIGHT_M = 1.53

# ---- SMPL 24-joint index map ----
JOINTS = {
    "pelvis": 0,
    "l_hip": 1, "r_hip": 2,
    "spine1": 3,
    "l_knee": 4, "r_knee": 5,
    "spine2": 6,
    "l_ankle": 7, "r_ankle": 8,
    "spine3": 9,
    "l_foot": 10, "r_foot": 11,
    "neck": 12,
    "l_collar": 13, "r_collar": 14,
    "head": 15,
    "l_shoulder": 16, "r_shoulder": 17,
    "l_elbow": 18, "r_elbow": 19,
    "l_wrist": 20, "r_wrist": 21,
    "l_hand": 22, "r_hand": 23,
}
LEFT_WRIST = JOINTS["l_wrist"]
RIGHT_WRIST = JOINTS["r_wrist"]
HEAD = JOINTS["head"]
L_ANKLE = JOINTS["l_ankle"]
R_ANKLE = JOINTS["r_ankle"]
PELVIS = JOINTS["pelvis"]
REF_JOINT_MODE = "ankle_mid"

BONES = [
    (0,3),(3,6),(6,9),(9,12),(12,15),
    (0,1),(1,4),(4,7),(7,10),
    (0,2),(2,5),(5,8),(8,11),
    (12,13),(13,16),(16,18),(18,20),(20,22),
    (12,14),(14,17),(17,19),(19,21),(21,23)
]

KEY_GROUPS = {
    "head":   {"indices":[HEAD], "color":"red"},
    "wrists": {"indices":[LEFT_WRIST, RIGHT_WRIST], "color":"orange"},
    "elbows": {"indices":[JOINTS["l_elbow"], JOINTS["r_elbow"]], "color":"purple"},
    "ankles": {"indices":[L_ANKLE, R_ANKLE], "color":"green"},
}
OTHER_POINT_COLOR = "royalblue"
BONE_COLOR = "gray"

# ====== Helpers ======
def ensure_outdir():
    if SAVE:
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def get_people(frame_obj):
    ppl = frame_obj.get("people", None)
    if ppl is None:
        ppl = frame_obj.get("persons", [])
    return ppl

def set_axes_equal(ax):
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    ranges = [abs(x_limits[1]-x_limits[0]), abs(y_limits[1]-y_limits[0]), abs(z_limits[1]-z_limits[0])]
    max_range = max(ranges)
    centers = [np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)]
    ax.set_xlim3d([centers[0]-max_range/2, centers[0]+max_range/2])
    ax.set_ylim3d([centers[1]-max_range/2, centers[1]+max_range/2])
    ax.set_zlim3d([centers[2]-max_range/2, centers[2]+max_range/2])

def remap_for_plot(joints_np):
    xs = joints_np[:, 0]
    vert = joints_np[:, VERTICAL_AXIS]
    zs = -vert
    depth_axis = 2 if VERTICAL_AXIS == 1 else (1 if VERTICAL_AXIS == 2 else 1)
    ys = joints_np[:, depth_axis]
    return xs, ys, zs

def plot_person(ax, joints):
    joints = np.asarray(joints, float)
    if CENTER_PELVIS and len(joints) > PELVIS:
        joints = joints - joints[PELVIS]
    xs, ys, zs = remap_for_plot(joints)
    for i, j in BONES:
        if i < len(joints) and j < len(joints):
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], c=BONE_COLOR, lw=2, alpha=0.85)
    painted = set()
    for group in KEY_GROUPS.values():
        for k in group["indices"]:
            if k < len(joints):
                ax.scatter(xs[k], ys[k], zs[k], s=60, c=group["color"], depthshade=True)
                painted.add(k)
    for i in range(len(joints)):
        if i not in painted:
            ax.scatter(xs[i], ys[i], zs[i], s=30, c=OTHER_POINT_COLOR, alpha=0.9, depthshade=True)
    for i in range(len(joints)):
        ax.text(xs[i], ys[i], zs[i], str(i), fontsize=8, color="black")

def plot_frame(frame_obj, title=None, fname=None):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")
    people = get_people(frame_obj)
    for person in people:
        plot_person(ax, person["joints_3d"])
    ax.set_xlabel("X (left-right)"); ax.set_ylabel("Y (depth)"); ax.set_zlabel("Z (up)")
    ax.view_init(elev=0, azim=0) if FRONT_VIEW else ax.view_init(elev=20, azim=-60)
    set_axes_equal(ax); ax.grid(True, alpha=0.25)
    if title: ax.set_title(title)
    if SAVE and fname:
        ensure_outdir()
        fig.savefig(Path(OUT_DIR)/fname, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close(fig)

def moving_average(arr, k=3):
    if k <= 1: return np.asarray(arr)
    out = []; q = deque(maxlen=k)
    for x in arr:
        q.append(np.asarray(x)); out.append(np.mean(q, axis=0))
    return np.vstack(out)

def compute_speeds(positions, fps):
    diffs = positions[1:] - positions[:-1]
    d = np.linalg.norm(diffs, axis=1)
    speeds = np.concatenate([[0.0], d * fps])
    return speeds

def find_pause_windows(speeds, fps, pause_seconds, speed_thr):
    min_len = int(pause_seconds * fps)
    slow = speeds < speed_thr
    windows = []; i = 0; n = len(slow)
    while i < n:
        if slow[i]:
            j = i
            while j < n and slow[j]: j += 1
            if (j - i) >= min_len:
                windows.append((i, j - 1))
            i = j
        else:
            i += 1
    return windows

def ground_distance(p1, p2):
    a, b = np.asarray(p1), np.asarray(p2)
    ax1, ax2 = GROUND_AXES
    return np.linalg.norm([a[ax1]-b[ax1], a[ax2]-b[ax2]])

def straight_line_displacement(start_pos, end_pos):
    p1, p2 = np.asarray(start_pos), np.asarray(end_pos)
    d3 = np.linalg.norm(p2 - p1)
    d2 = ground_distance(p1, p2)
    return d3, d2

# ====== Main ======
def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data if isinstance(data, list) else [data]
    if MAX_FRAMES: frames = frames[:MAX_FRAMES]

    track_counts = {}
    for fr in frames:
        for p in get_people(fr):
            tid = p.get("track_id")
            if tid is not None:
                track_counts[tid] = track_counts.get(tid, 0) + 1
    main_track_id = max(track_counts, key=track_counts.get) if track_counts else None
    print(f"Main track_id: {main_track_id}")

    ref_positions = []; valid_idx = []
    for idx, fr in enumerate(frames):
        ppl = get_people(fr)
        if not ppl: continue
        person = next((p for p in ppl if p.get("track_id") == main_track_id), ppl[0]) if main_track_id is not None else ppl[0]
        joints = np.asarray(person["joints_3d"], float)
        la, ra = L_ANKLE, R_ANKLE
        ankle_mid = (joints[la] + joints[ra]) / 2
        ref_positions.append(ankle_mid if REF_JOINT_MODE == "ankle_mid" else joints[PELVIS])
        valid_idx.append(idx)

    if len(ref_positions) < max(10, FPS * 2):
        print("Not enough frames with people")
        return

    ref_positions = np.array(ref_positions)
    ref_smooth = moving_average(ref_positions, k=SMOOTH_WINDOW)

    quick_heights = []
    for i, fr_idx in enumerate(valid_idx[:min(len(valid_idx), 50)]):
        ppl = get_people(frames[fr_idx])
        if not ppl: continue
        person = next((p for p in ppl if p.get("track_id") == main_track_id), ppl[0]) if main_track_id is not None else ppl[0]
        j = np.asarray(person["joints_3d"], float)
        h = np.linalg.norm(j[HEAD] - (j[L_ANKLE] + j[R_ANKLE]) / 2)
        quick_heights.append(h)
    med_h = np.median(quick_heights) if quick_heights else 1.0
    norm_thr = STILL_SPEED_THR_NORMALIZED * med_h

    speeds = compute_speeds(ref_smooth, FPS)

    if MANUAL_START_FRAME is not None or MANUAL_END_FRAME is not None:
        s = valid_idx[0] if MANUAL_START_FRAME is None else MANUAL_START_FRAME
        e = valid_idx[-1] if MANUAL_END_FRAME is None else MANUAL_END_FRAME
        w_start = np.searchsorted(valid_idx, s, side="left")
        w_end   = np.searchsorted(valid_idx, e, side="right") - 1
        start_pos, end_pos = ref_smooth[w_start], ref_smooth[w_end]
        used_autotrim = False
    else:
        used_autotrim = True
        pause_windows = find_pause_windows(speeds, FPS, PAUSE_SECONDS, norm_thr)
        if len(pause_windows) < 2:
            print("Could not find two pauses, using full sequence.")
            w_start, w_end = 0, len(ref_smooth) - 1
            start_pos, end_pos = ref_smooth[w_start], ref_smooth[w_end]
        else:
            start_pause, end_pause = pause_windows[0], pause_windows[-1]
            start_pos = np.mean(ref_smooth[start_pause[0]:start_pause[1]+1], axis=0)
            end_pos   = np.mean(ref_smooth[end_pause[0]:end_pause[1]+1], axis=0)
            w_start, w_end = start_pause[1] + 1, max(start_pause[1] + 2, end_pause[0] - 1)

    dist3d_raw, dist2d_raw = straight_line_displacement(start_pos, end_pos)
    print(f"Model distance 3D: {dist3d_raw:.3f} (model units) | Ground 2D: {dist2d_raw:.3f} (model units)")

    heights_segment, wrists_segment, frames_segment = [], [], []
    for i in range(w_start, w_end + 1):
        fr = frames[valid_idx[i]]
        ppl = get_people(fr)
        if not ppl: continue
        person = next((p for p in ppl if p.get("track_id") == main_track_id), ppl[0]) if main_track_id is not None else ppl[0]
        j = np.asarray(person["joints_3d"], float)

        head = j[HEAD]
        ankle_mid = (j[L_ANKLE] + j[R_ANKLE]) / 2
        h = float(np.linalg.norm(head - ankle_mid))
        heights_segment.append(h)

        lw, rw = j[LEFT_WRIST], j[RIGHT_WRIST]
        wrists_segment.append(float(np.linalg.norm(lw - rw)))

        frames_segment.append(valid_idx[i])

    height_med = float(np.median(heights_segment)) if heights_segment else float("nan")
    wrist_med  = float(np.median(wrists_segment)) if wrists_segment else float("nan")
    dist_by_height = (dist2d_raw / height_med) if height_med > 0 else float("nan")

    print(f"Median Height: {height_med:.3f} (model units) | Wrist-Wrist: {wrist_med:.3f} (model units)")
    print(f"Distance/Height ratio: {dist_by_height:.3f}")

    if TAPE_DISTANCE_M is not None:
        abs_err_dist_raw = abs(TAPE_DISTANCE_M - dist2d_raw)
        print("[Note] Raw vs tape not directly comparable (units differ). Calibrate first.")
    if TRUE_HEIGHT_M is not None and height_med > 0:
        abs_err_height = abs(TRUE_HEIGHT_M - height_med)
        rel_err_height = 100 * abs_err_height / TRUE_HEIGHT_M
        print(f"Height (raw) vs true: {abs_err_height:.3f} m difference if raw were meters (not meaningful pre-calib)")

    scale_h = (TRUE_HEIGHT_M / height_med) if (TRUE_HEIGHT_M and height_med > 0) else None
    if scale_h:
        dist3d_cal_h = dist3d_raw * scale_h
        dist2d_cal_h = dist2d_raw * scale_h
        wrist_med_cal_h = wrist_med * scale_h
        print(f"Calibration (by height) scale: {scale_h:.3f}")
        print(f"Height-calibrated: distance_3D={dist3d_cal_h:.3f} m | ground_2D={dist2d_cal_h:.3f} m | wrist={wrist_med_cal_h:.3f} m")
        if TAPE_DISTANCE_M is not None:
            abs_err_dist_h = abs(TAPE_DISTANCE_M - dist2d_cal_h)
            rel_err_dist_h = 100 * abs_err_dist_h / TAPE_DISTANCE_M
            print(f"Error vs tape (height-calibrated): {abs_err_dist_h:.3f} m ({rel_err_dist_h:.2f}%)")
    else:
        dist3d_cal_h = dist2d_cal_h = wrist_med_cal_h = None
        abs_err_dist_h = rel_err_dist_h = None

    tape_scale = (TAPE_DISTANCE_M / dist2d_raw) if (TAPE_DISTANCE_M and dist2d_raw > 0) else None
    if tape_scale:
        dist3d_cal_tape = dist3d_raw * tape_scale
        dist2d_cal_tape = dist2d_raw * tape_scale
        wrist_med_cal_tape = wrist_med * tape_scale
        print(f"Calibration (by tape) scale: {tape_scale:.3f}")
        print(f"Tape-calibrated: distance_3D={dist3d_cal_tape:.3f} m | ground_2D={dist2d_cal_tape:.3f} m | wrist={wrist_med_cal_tape:.3f} m")
    else:
        dist3d_cal_tape = dist2d_cal_tape = wrist_med_cal_tape = None

    ensure_outdir()
    if frames_segment:
        with open(Path(OUT_DIR)/f"{TRIAL_ID}_wrist_series.csv","w",newline="",encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["frame","wrist_model_units"])
            w.writerows(zip(frames_segment, wrists_segment))
        with open(Path(OUT_DIR)/f"{TRIAL_ID}_height_series.csv","w",newline="",encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["frame","height_model_units"])
            w.writerows(zip(frames_segment, heights_segment))

    summary = {
        "trial_id": TRIAL_ID,
        "fps": FPS,
        "used_autotrim": used_autotrim,
        "model_distance_3d_units": round(dist3d_raw, 4),
        "model_distance_ground_units": round(dist2d_raw, 4),
        "median_height_units": round(height_med, 4),
        "median_wrist_units": round(wrist_med, 4),
        "distance_by_height": round(dist_by_height, 4),
        "true_height_m": TRUE_HEIGHT_M,
        "tape_distance_m": TAPE_DISTANCE_M,
        "scale_by_height": round(scale_h, 5) if scale_h else None,
        "cal_h_distance_3d_m": round(dist3d_cal_h, 4) if dist3d_cal_h is not None else None,
        "cal_h_distance_ground_m": round(dist2d_cal_h, 4) if dist2d_cal_h is not None else None,
        "cal_h_wrist_m": round(wrist_med_cal_h, 4) if wrist_med_cal_h is not None else None,
        "scale_by_tape": round(tape_scale, 5) if tape_scale else None,
        "cal_tape_distance_3d_m": round(dist3d_cal_tape, 4) if dist3d_cal_tape is not None else None,
        "cal_tape_distance_ground_m": round(dist2d_cal_tape, 4) if dist2d_cal_tape is not None else None,
        "cal_tape_wrist_m": round(wrist_med_cal_tape, 4) if wrist_med_cal_tape is not None else None,
    }
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not file_exists: writer.writeheader()
        writer.writerow(summary)
    print(f"Saved summary to {RESULTS_CSV}")

    # --- Plots for the dissertation ---

    # Ground-plane path (calibrated to meters), with dashed GT line
    xs = ref_smooth[w_start:w_end+1, GROUND_AXES[0]]
    zs = ref_smooth[w_start:w_end+1, GROUND_AXES[1]]

    plt.figure(figsize=(5, 5))

    if TAPE_DISTANCE_M and len(xs) > 1:
        raw_dx, raw_dz = xs[-1] - xs[0], zs[-1] - zs[0]
        raw_len = np.hypot(raw_dx, raw_dz)

        if raw_len > 1e-8:
            scale = TAPE_DISTANCE_M / raw_len
            xs_m, zs_m = xs * scale, zs * scale

            # plot calibrated trajectory
            plt.plot(xs_m, zs_m, marker='.', label="CoMotion (meters)")

            # compute direction AFTER scaling (in meters)
            dx_m = xs_m[-1] - xs_m[0]
            dz_m = zs_m[-1] - zs_m[0]
            seg_len_m = np.hypot(dx_m, dz_m)
            if seg_len_m > 1e-8:
                ux, uz = dx_m / seg_len_m, dz_m / seg_len_m
                x2, z2 = xs_m[0] + ux * TAPE_DISTANCE_M, zs_m[0] + uz * TAPE_DISTANCE_M
                plt.plot([xs_m[0], x2], [zs_m[0], z2], linestyle="--", label=f"Tape {TAPE_DISTANCE_M} m")
    else:
        plt.plot(xs, zs, marker='.', label="CoMotion (model units)")

    plt.axis('equal'); plt.grid(True, alpha=0.3)
    plt.title("Ground-plane path (walking segment)")
    plt.xlabel("Meters" if TAPE_DISTANCE_M else f"Axis {GROUND_AXES[0]}")
    plt.ylabel("Meters" if TAPE_DISTANCE_M else f"Axis {GROUND_AXES[1]}")
    plt.legend()

    if SAVE:
        plt.savefig(Path(OUT_DIR)/f"{TRIAL_ID}_ground_path.png", dpi=DPI)
    if SHOW:
        plt.show()
    plt.close()

    # Wrist distance over time (model units)
    if frames_segment:
        plt.figure(figsize=(6,4))
        plt.plot(frames_segment, wrists_segment)
        plt.title("Wrist-to-wrist distance over time (model units)")
        plt.xlabel("Frame"); plt.ylabel("Distance (model units)")
        plt.grid(True, alpha=0.3)
        if SAVE: plt.savefig(Path(OUT_DIR)/f"{TRIAL_ID}_wrist_curve.png", dpi=DPI)
        if SHOW: plt.show()
        plt.close()

    # Height comparison bar
    if TRUE_HEIGHT_M and height_med > 0:
        plt.figure(figsize=(4,4))
        plt.bar(["Measured (m)", "Model (units)"], [TRUE_HEIGHT_M, height_med])
        plt.title("Height: measured vs model")
        if SAVE: plt.savefig(Path(OUT_DIR)/f"{TRIAL_ID}_height_bar.png", dpi=DPI)
        if SHOW: plt.show()
        plt.close()

    # First/last skeleton frames
    first_idx, last_idx = valid_idx[w_start], valid_idx[w_end]
    plot_frame(frames[first_idx], title=f"Frame {first_idx} (start)", fname=f"{TRIAL_ID}_frame_start.png")
    plot_frame(frames[last_idx],  title=f"Frame {last_idx} (end)",   fname=f"{TRIAL_ID}_frame_end.png")

if __name__ == "__main__":
    main()
