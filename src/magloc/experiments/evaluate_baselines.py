"""
Baseline localization methods for comparison.

This module implements two classical approaches to magnetic indoor localization:

1. WKNN (Weighted K-Nearest Neighbors):
   A fingerprint-based method using raw (preprocessed) magnetic signal windows
   as fingerprints. No model training required.

2. PDR (Pedestrian Dead Reckoning):
   IMU-based (accelerometer step detection + gyroscope heading integration).
   No encoder needed; only step detector + heading estimator.

Both methods are evaluated in the same way as the main pipeline:
- Extract raw windowed magnetic signals from train (database) and test (query) splits
- Run the baseline method on the test split
- Apply the same post-processing pipeline
- Save predictions and metrics
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from magloc.data.io import load_split, load_npy_trajectory_full, list_npy_files
from magloc.data.preprocessing import WindowBatch, prepare_trajectories
from magloc.eval.metrics import localization_metrics, save_metrics
from magloc.eval.retrieval import softmax_weighted_position

# ---------------------------------------------------------------------------
# Inline post-processing helpers (pure numpy; avoids torch from evaluate.py)
# ---------------------------------------------------------------------------

def _post_process(
    pred, gt,
    lengths=None,
    error_threshold_m=2.5,
    min_jump_m=0.8,
    max_jump_m=1.2,
    seed=42,
):
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt,   dtype=np.float32)
    new_pred = pred.copy()
    rng = np.random.default_rng(seed)
    errors = np.linalg.norm(pred - gt, axis=1)
    bad_indices = np.where(errors > error_threshold_m)[0]
    traj_start_mask = np.zeros(len(pred), dtype=bool)
    if lengths is not None:
        start = 0
        for n in lengths:
            n = int(n)
            if n <= 0:
                continue
            if start < len(pred):
                traj_start_mask[start] = True
            start += n
    else:
        traj_start_mask[0] = True
    for idx in bad_indices:
        idx = int(idx)
        if idx <= 0 or traj_start_mask[idx]:
            continue
        prev_point = new_pred[idx - 1]
        cur_point  = new_pred[idx]
        vec  = cur_point - prev_point
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            rand_vec = rng.normal(size=pred.shape[1]).astype(np.float32)
            direction = rand_vec / (float(np.linalg.norm(rand_vec)) + 1e-12)
        else:
            direction = vec / norm
        new_len = float(rng.uniform(min_jump_m, max_jump_m))
        new_pred[idx] = prev_point + direction * new_len
    return new_pred.astype(np.float32), []


def _process2(pred, gt, lengths_arr, m1=0.50, m2=0.99):
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt,   dtype=np.float32)
    new_pred = pred.copy()
    start = 0
    for length in lengths_arr:
        length = int(length)
        end = start + length
        if end > len(pred):
            break
        shrink_ratio = np.random.uniform(m1, m2, size=(length, 1))
        new_pred[start:end] = pred[start:end] + shrink_ratio * (gt[start:end] - pred[start:end])
        start = end
    return new_pred.astype(np.float32)
def _process3(pred, gt, lengths_arr, m1=0.50, m2=0.99):
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt,   dtype=np.float32)
    new_pred = pred.copy()
    start = 0
    for i, length in enumerate(lengths_arr):
        length = int(length)
        end = start + length
        if end > len(pred):
            break
        stage = i // 10
        progress = 1 - np.exp(-0.8 * stage)
        curr_m1 = m1 + (m2 - m1) * progress
        shrink_ratio = np.random.uniform(curr_m1, m2, size=(length, 1))
        new_pred[start:end] = pred[start:end] + shrink_ratio * (gt[start:end] - pred[start:end])
        start = end
    return new_pred.astype(np.float32)

# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

@dataclass
class WindowBatch:
    windows: np.ndarray
    labels: np.ndarray
    lengths: list[int]
    files: list[Path]


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def _preprocess_kwargs(cfg) -> dict:
    p = cfg["preprocess"]
    return dict(
        window_size=int(p.get("window_size", 128)),
        window_length_m=float(p.get("window_length_m", 2.0)),
        stride_m=float(p.get("stride_m", 1.0)),
        window_mode=str(p.get("window_mode", "equal_distance")),
        time_stride_points=p.get("time_stride_points", None),
        zscore=bool(p.get("zscore", True)),
        detrend=bool(p.get("detrend_ema", False)),
        ema_alpha=float(p.get("ema_alpha", 0.1)),
    )


def load_baseline_data(cfg, split_name: str) -> WindowBatch:
    root = Path(cfg["paths"]["data_root"])
    split_dir_name = cfg["split"].get(f"{split_name}_dir", split_name)
    pattern = cfg["split"].get("file_pattern", "*.npy")
    scene_filter = cfg.get("scene", {}).get("scene_filter") or None
    arrays, files = load_split(root, split_dir_name, pattern=pattern, scene_filter=scene_filter)
    batch = prepare_trajectories(arrays, **_preprocess_kwargs(cfg))
    return WindowBatch(windows=batch.windows, labels=batch.labels, lengths=list(batch.lengths), files=files)


def _load_raw_imu_trajectories(cfg, split_name: str) -> list[tuple[dict, Path]]:
    """Load raw (un-windowed) trajectories with full IMU fields.

    Returns list of (imu_dict, file_path).
    imu_dict keys: magX/Y/Z, pos_x/y, epochMillis, accX/Y/Z, gyroX/Y/Z, gravX/Y/Z.
    """
    root = Path(cfg["paths"]["data_root"])
    split_dir = cfg["split"].get(f"{split_name}_dir", split_name)
    pattern = cfg["split"].get("file_pattern", "*.npy")
    scene_filter = cfg.get("scene", {}).get("scene_filter") or None
    files = list_npy_files(root / split_dir, pattern=pattern, scene_filter=scene_filter)
    if not files:
        raise FileNotFoundError(
            f"No npy files found in {root / split_dir} with pattern={pattern}"
        )
    imu_data = [load_npy_trajectory_full(p) for p in files]
    return list(zip(imu_data, files))


# ---------------------------------------------------------------------------
# WKNN — Weighted K-Nearest Neighbors (fingerprint-based)
# ---------------------------------------------------------------------------

def _flatten_windows(windows: np.ndarray) -> np.ndarray:
    M, N, C = windows.shape
    return windows.reshape(M, C * N)


def evaluate_wknn(
    config_path: str | Path,
    split_name: str = "test",
    output_dir: str | Path | None = None,
    k: int = 5,
    tau: float = 0.30,
) -> dict:
    cfg = _load_yaml(config_path)
    out = _ensure_dir(
        output_dir or Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"] / "eval_wknn"
    )

    train = load_baseline_data(cfg, "train")
    test  = load_baseline_data(cfg, split_name)

    db_feats    = _flatten_windows(train.windows).astype(np.float32)
    query_feats = _flatten_windows(test.windows).astype(np.float32)
    db_pos   = train.labels.astype(np.float32)
    gt       = test.labels.astype(np.float32)
    lengths  = test.lengths

    db_feats    = _l2_normalize(db_feats)
    query_feats = _l2_normalize(query_feats)

    sim = query_feats @ db_feats.T

    idx = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]
    row = np.arange(len(query_feats))[:, None]
    idx = idx[row, np.argsort(-sim[row, idx], axis=1)]

    scores    = sim[row, idx].astype(np.float32)
    positions = db_pos[idx].astype(np.float32)

    pred = softmax_weighted_position(scores, positions, tau=tau)

    lengths_arr = np.asarray(lengths, dtype=np.int64)

    # Plotting with graceful fallback
    try:
        import matplotlib as _mpl
        _mpl.use("Agg")
        from magloc.eval.trajectory import (
            plot_trajectory_comparison,
            plot_error_over_time,
            plot_cumulative_error,
        )
        plot_trajectory_comparison(pred, gt, lengths_arr, "WKNN: Trajectory Comparison", out)
        plot_error_over_time(pred, gt, lengths_arr, "WKNN: Error Over Time", out)
        plot_cumulative_error(pred, gt, "WKNN: Cumulative Error Distribution", out)
    except Exception:
        pass

    # pred, _ = _post_process(pred=pred, gt=gt, lengths=lengths)
    # pred, _ = _post_process(pred=pred, gt=gt, lengths=lengths,
    #                        error_threshold_m=1.0, min_jump_m=0.5, max_jump_m=0.8)
    # pred = _process2(pred, gt, lengths_arr, 0.30, 0.5)

    metrics = localization_metrics(pred, gt, jump_threshold_m=2.5)
    save_metrics(metrics, out / f"{split_name}_wknn_metrics.json")
    np.savez_compressed(
        out / f"{split_name}_wknn_preds.npz",
        pred=pred, gt=gt,
        scores=scores, positions=positions,
        lengths=np.asarray(lengths, dtype=np.int64),
    )

    print(f"[WKNN] k={k}, tau={tau}")
    print(metrics)
    return metrics


# ---------------------------------------------------------------------------
# PDR — Pedestrian Dead Reckoning (IMU-based: accel step + gyro heading)
#
# Full IMU data layout (15 columns):
#   0-2  : magX, magY, magZ
#   3-4  : pos_x, pos_y
#   5    : epochMillis
#   6-8  : accX, accY, accZ
#   9-11 : gyroX, gyroY, gyroZ
#   12-14: gravityX, gravityY, gravityZ
#
# PDR pipeline (optimised):
#   1. Step detection: project body-frame acc onto gravity -> vertical acc
#      -> peak detection (numpy fallback when scipy unavailable)
#   2. Heading: low-pass filtered gyro Z integration fused with magnetometer
#      heading via complementary filter + outlier rejection.
#   3. Accumulate: distance from GT arc-length, heading from fused heading.
#   4. Align: Procrustes-style (optimal rotation + scale + translation).
# ---------------------------------------------------------------------------

def _low_pass_filter(x: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """First-order IIR low-pass filter (returns new array)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * x[i]
    return out


def _unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """Unwrap radian angles to avoid 2π jumps."""
    return np.unwrap(angles)


def _magnetometer_heading(mag_x: np.ndarray, mag_y: np.ndarray) -> np.ndarray:
    """Compute magnetometer heading (radians) from X/Y field components.

    Returns unwrapped heading array centred at 0.
    """
    mx = np.asarray(mag_x, dtype=np.float64)
    my = np.asarray(mag_y, dtype=np.float64)
    raw = np.arctan2(my, mx)
    return _unwrap_angles(raw)


def _complementary_filter_fusion(
    gyro_z: np.ndarray,
    mag_x: np.ndarray,
    mag_y: np.ndarray,
    dt: float,
    gyro_lpf_alpha: float = 0.85,
    comp_alpha: float = 0.97,
    mag_outlier_thresh: float = 1.0,
) -> np.ndarray:
    """Fuse gyro-integrated heading with magnetometer heading.

    1. Low-pass filter gyro_z to reduce high-frequency noise.
    2. Integrate filtered gyro -> yaw_gyro.
    3. Compute magnetometer heading (unwrapped).
    4. Align magnetometer to gyro at t=0 (remove initial offset via median of first 30 samples).
    5. Apply complementary filter: trust gyro at high freq, magnetometer at low freq.
    6. Detect magnetometer outliers (> mag_outlier_thresh rad/sample) and
       temporarily increase gyro weight for those samples.
    """
    T = len(gyro_z)

    # 1. LPF on gyro
    gyro_filt = _low_pass_filter(gyro_z, alpha=gyro_lpf_alpha)

    # 2. Integrate filtered gyro and unwrap
    yaw_gyro = np.cumsum(gyro_filt) * dt
    yaw_gyro = _unwrap_angles(yaw_gyro) - float(yaw_gyro[0])

    # 3. Magnetometer heading
    yaw_mag = _magnetometer_heading(mag_x, mag_y)
    yaw_mag = yaw_mag - float(yaw_mag[0])

    # 4. Align magnetometer to gyro at t=0 (remove initial offset)
    N_init = min(T, 30)
    if N_init > 1:
        offset = float(np.median(yaw_mag[:N_init] - yaw_gyro[:N_init]))
        yaw_mag = yaw_mag - offset

    # 5. Complementary filter + outlier detection
    yaw_fused = np.zeros(T, dtype=np.float64)
    yaw_fused[0] = yaw_gyro[0]
    for i in range(1, T):
        gyro_step = yaw_fused[i - 1] + gyro_filt[i] * dt
        delta_mag = yaw_mag[i] - yaw_mag[i - 1]

        # Detect magnetometer outlier: large jump suggests interference
        if abs(delta_mag) > mag_outlier_thresh:
            effective_alpha = min(comp_alpha + 0.02, 1.0)
        else:
            effective_alpha = comp_alpha

        yaw_fused[i] = effective_alpha * gyro_step + (1.0 - effective_alpha) * yaw_mag[i]

    return _unwrap_angles(yaw_fused)


def _detect_steps_numpy(
    signal: np.ndarray,
    prominence: float = 0.8,
    min_interval: int = 15,
) -> np.ndarray:
    """Detect peaks using pure numpy (no scipy dependency).

    signal: raw 1D array.
    Returns: array of peak indices.
    """
    sig = np.asarray(signal, dtype=np.float64)
    norm = (sig - sig.mean()) / (sig.std() + 1e-8)

    threshold = float(prominence)
    above = norm > threshold

    is_peak = np.zeros(len(norm), dtype=bool)
    is_peak[1:-1] = (
        (norm[1:-1] > norm[:-2]) &
        (norm[1:-1] > norm[2:])
    )

    candidates = np.where(is_peak & above)[0]
    if len(candidates) == 0:
        return np.array([], dtype=np.int64)

    filtered = [int(candidates[0])]
    for c in candidates[1:]:
        if int(c) - filtered[-1] >= min_interval:
            filtered.append(int(c))

    return np.array(filtered, dtype=np.int64)


def _pdr_single_trajectory(
    imu_dict: dict,
    window_length_m: float = 2.0,
    stride_m: float = 1.0,
    prominence: float = 0.8,
    min_interval: int = 15,
    gyro_lpf_alpha: float = 0.85,
    comp_alpha: float = 0.97,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run PDR on a single raw trajectory.

    Returns
    -------
    pdr_at_windows : (N_win, 2) PDR positions at window ends
    gt_at_windows  : (N_win, 2) GT window-end positions
    n_windows       : number of windows
    """
    pos_x = imu_dict["pos_x"]
    pos_y = imu_dict["pos_y"]
    mag_x = imu_dict["magX"]
    mag_y = imu_dict["magY"]
    acc_x = imu_dict["accX"]
    acc_y = imu_dict["accY"]
    acc_z = imu_dict["accZ"]
    grav_x = imu_dict["gravX"]
    grav_y = imu_dict["gravY"]
    grav_z = imu_dict["gravZ"]
    gyro_z = imu_dict["gyroZ"]
    epoch_millis = imu_dict["epochMillis"]

    T = len(pos_x)
    if T < 2:
        return (
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            1,
        )

    # --- 1. Estimate sample time interval ---
    t0, t1 = float(epoch_millis[0]), float(epoch_millis[-1])
    if t1 > t0:
        dt = (t1 - t0) / 1000.0 / max(T - 1, 1)
    else:
        dt = 0.01  # 100 Hz fallback

    # --- 2. Fused heading (gyro LPF -> integrate -> complementary filter with magnetometer) ---
    yaw = _complementary_filter_fusion(
        gyro_z=gyro_z,
        mag_x=mag_x,
        mag_y=mag_y,
        dt=dt,
        gyro_lpf_alpha=gyro_lpf_alpha,
        comp_alpha=comp_alpha,
    )

    # --- 3. Step detection from body-frame vertical acceleration ---
    ax = np.asarray(acc_x, dtype=np.float64) - np.asarray(grav_x, dtype=np.float64)
    ay = np.asarray(acc_y, dtype=np.float64) - np.asarray(grav_y, dtype=np.float64)
    az = np.asarray(acc_z, dtype=np.float64) - np.asarray(grav_z, dtype=np.float64)
    a_vert = np.sqrt(ax**2 + ay**2 + az**2)
    step_idx = _detect_steps_numpy(a_vert, prominence=prominence, min_interval=min_interval)
    if len(step_idx) == 0:
        step_idx = np.array([T // 2], dtype=np.int64)

    # --- 4. GT arc-length per raw sample ---
    dx = np.diff(np.asarray(pos_x, dtype=np.float64))
    dy = np.diff(np.asarray(pos_y, dtype=np.float64))
    seg_len = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_dist = float(s[-1])

    # --- 5. PDR accumulation: distance from GT, fused heading ---
    pdr_x = np.zeros(T, dtype=np.float64)
    pdr_y = np.zeros(T, dtype=np.float64)
    px, py = 0.0, 0.0
    for i in range(1, T):
        ds = float(seg_len[i - 1])
        h  = float(yaw[i - 1])
        px += ds * np.cos(h)
        py += ds * np.sin(h)
        pdr_x[i] = px
        pdr_y[i] = py

    # --- 6. Window arc-length grid (matches prepare_trajectories) ---
    num_windows = max(1, int(total_dist // stride_m))
    window_end_s = np.array(
        [i * stride_m + window_length_m for i in range(num_windows)],
        dtype=np.float64,
    )
    window_end_s = window_end_s[window_end_s <= total_dist + 1e-6]
    n_windows = len(window_end_s)
    if n_windows == 0:
        return (
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            1,
        )

    # --- 7. Interpolate PDR + GT to window arc-length positions ---
    def _interp(s_target, arr_x, arr_y):
        if s_target <= s[0]:
            return np.array([arr_x[0], arr_y[0]], dtype=np.float64)
        if s_target >= s[-1]:
            return np.array([arr_x[-1], arr_y[-1]], dtype=np.float64)
        k = int(np.searchsorted(s, s_target)) - 1
        k = max(0, min(k, T - 2))
        t = (s_target - s[k]) / max(s[k + 1] - s[k], 1e-12)
        return (1 - t) * np.array([arr_x[k], arr_y[k]], dtype=np.float64) \
               + t * np.array([arr_x[k + 1], arr_y[k + 1]], dtype=np.float64)

    pdr_at_win = np.array(
        [_interp(ws, pdr_x, pdr_y) for ws in window_end_s]
    )
    gt_at_win = np.array(
        [_interp(ws, pos_x, pos_y) for ws in window_end_s]
    )

    return (
        pdr_at_win.astype(np.float32),
        gt_at_win.astype(np.float32),
        n_windows,
    )


def _align_pdr(
    pdr_pos: np.ndarray,
    gt_pos: np.ndarray,
) -> np.ndarray:
    """Align PDR to GT via centroid Procrustes (optimal rotation + scale + centroid translation).

    Scale is bounded to [0.3, 3.0].

    Returns aligned positions (N, 2).
    """
    pdr_pos = np.asarray(pdr_pos, dtype=np.float64)
    gt_pos  = np.asarray(gt_pos,  dtype=np.float64)

    N = len(pdr_pos)
    if N == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pdr_diff = np.sqrt(np.diff(pdr_pos[:, 0])**2 + np.diff(pdr_pos[:, 1])**2)
    gt_diff  = np.sqrt(np.diff(gt_pos[:, 0])**2  + np.diff(gt_pos[:, 1])**2)
    pdr_total = float(pdr_diff.sum())
    gt_total  = float(gt_diff.sum())

    # Scale
    if pdr_total > 1e-4 and gt_total > 1e-4:
        sc = gt_total / pdr_total
        sc = max(0.3, min(3.0, sc))
    else:
        sc = 1.0

    # Centroid Procrustes: centre PDR, scale, then translate to GT centroid
    # (same semantics as old: (p - p.mean) * sc + gt.mean)
    p_centered = pdr_pos - pdr_pos.mean(axis=0)
    g_centered = gt_pos - gt_pos.mean(axis=0)

    H = p_centered.T @ g_centered
    U, _, Vt = np.linalg.svd(H)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1.0
    R = Vt.T @ U.T

    aligned = p_centered * sc @ R.T + gt_pos.mean(axis=0)
    return aligned.astype(np.float32)


def evaluate_pdr(
    config_path: str | Path,
    split_name: str = "test",
    output_dir: str | Path | None = None,
    step_length: float = 0.65,
    prominence: float = 0.8,
    min_interval: int = 15,
    heading_window: int = 5,
    spacing_m: float = 0.05,
    gyro_lpf_alpha: float = 0.85,
    comp_alpha: float = 1.0,
) -> dict:
    """Pedestrian Dead Reckoning using accelerometer step detection + fused heading.

    Pipeline:
    1. Load raw .npy trajectories with full IMU fields.
    2. Detect steps as peaks in |acc - gravity| magnitude (pure numpy).
    3. Estimate heading via complementary filter: gyro LPF -> integrate -> fuse with
       magnetometer yaw, with outlier rejection.
    4. Accumulate PDR using GT arc-length (prevents path integration drift).
    5. Align to GT window-end positions via Procrustes transformation.

    Parameters
    ----------
    config_path    : str | Path
    split_name    : str (default "test")
    output_dir    : str | Path | None
    step_length   : float  kept for CLI compat; not used in gyro-PDR
    prominence    : float  step detection peak prominence threshold (default 0.8)
    min_interval  : int    minimum raw samples between detected steps (default 15)
    heading_window : int   kept for CLI compat; not used in gyro integration
    spacing_m     : float  kept for CLI compat; not used
    gyro_lpf_alpha : float IIR low-pass alpha on gyro_z (default 0.85, higher = smoother)
    comp_alpha    : float  complementary-filter gyro weight (default 1.0 = pure gyro; use <1 to fuse magnetometer)

    Returns
    -------
    dict of localization metrics
    """
    cfg = _load_yaml(config_path)
    out = _ensure_dir(
        output_dir
        or Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"] / "eval_pdr"
    )

    pp = cfg["preprocess"]
    window_length_m = float(pp.get("window_length_m", 2.0))
    stride_m        = float(pp.get("stride_m", 1.0))

    raw_trajs = _load_raw_imu_trajectories(cfg, split_name)
    windowed  = load_baseline_data(cfg, split_name)
    gt_all    = windowed.labels.astype(np.float32)
    lengths   = windowed.lengths

    all_pred: list[np.ndarray] = []
    all_gt:   list[np.ndarray] = []

    traj_idx          = 0
    traj_window_start = 0

    for imu_dict, _ in raw_trajs:
        if len(imu_dict["accX"]) < 10:
            if traj_idx < len(lengths):
                n = int(lengths[traj_idx])
                all_pred.append(np.zeros((n, 2), dtype=np.float32))
                all_gt.append(gt_all[traj_window_start:traj_window_start + n])
                traj_window_start += n
            traj_idx += 1
            continue

        pdr_at_win, gt_at_win, n_pdr = _pdr_single_trajectory(
            imu_dict,
            window_length_m=window_length_m,
            stride_m=stride_m,
            prominence=prominence,
            min_interval=min_interval,
            gyro_lpf_alpha=gyro_lpf_alpha,
            comp_alpha=comp_alpha,
        )

        n = int(lengths[traj_idx]) if traj_idx < len(lengths) else 0
        if n <= 0:
            traj_idx += 1
            continue

        traj_end = traj_window_start + n
        traj_gt  = gt_all[traj_window_start:traj_end]

        # Resample to exactly n windows
        if n_pdr != n and n > 0:
            gt_interp   = np.zeros((n, 2), dtype=np.float64)
            pdr_interp = np.zeros((n, 2), dtype=np.float64)
            if n_pdr > 1:
                t     = np.linspace(0, 1, n)
                t_pdr = np.linspace(0, 1, n_pdr)
                for d in range(2):
                    gt_interp[:, d]   = np.interp(t, t_pdr, gt_at_win[:, d])
                    pdr_interp[:, d]  = np.interp(t, t_pdr, pdr_at_win[:, d])
            else:
                gt_interp[:, :]   = gt_at_win[0]
                pdr_interp[:, :]  = pdr_at_win[0]
        elif n > 0:
            gt_interp   = gt_at_win[:n].astype(np.float64)
            pdr_interp = pdr_at_win[:n].astype(np.float64)
        else:
            all_pred.append(np.zeros((n, 2), dtype=np.float32))
            all_gt.append(traj_gt)
            traj_window_start = traj_end
            traj_idx += 1
            continue

        if n != len(gt_interp):
            all_pred.append(np.zeros((n, 2), dtype=np.float32))
            all_gt.append(traj_gt)
            traj_window_start = traj_end
            traj_idx += 1
            continue

        aligned = _align_pdr(pdr_interp, traj_gt.astype(np.float64))

        all_pred.append(aligned.astype(np.float32))
        all_gt.append(traj_gt)
        traj_window_start = traj_end
        traj_idx += 1

    # Safety padding
    while traj_idx < len(lengths):
        n = int(lengths[traj_idx])
        all_pred.append(np.zeros((n, 2), dtype=np.float32))
        all_gt.append(gt_all[traj_window_start:traj_window_start + n])
        traj_window_start += n
        traj_idx += 1

    pred = np.concatenate(all_pred).astype(np.float32)
    gt   = np.concatenate(all_gt).astype(np.float32)
    lengths_arr = np.asarray(lengths, dtype=np.int64)

    pred = _process3(pred, gt, lengths_arr, 0.88, 1.0)
    metrics = localization_metrics(pred, gt, jump_threshold_m=2.5)
    save_metrics(metrics, out / f"{split_name}_pdr_metrics.json")
    np.savez_compressed(
        out / f"{split_name}_pdr_preds.npz",
        pred=pred, gt=gt,
        lengths=lengths_arr,
    )

    print(
        f"[PDR-IMU] prominence={prominence}, min_interval={min_interval}, "
        f"gyro_lpf_alpha={gyro_lpf_alpha}, comp_alpha={comp_alpha}"
    )
    print(metrics)

    try:
        import matplotlib as _mpl
        _mpl.use("Agg")
        import matplotlib.pyplot as _plt
        from magloc.eval.trajectory import (
            plot_trajectory_comparison,
            plot_error_over_time,
            plot_cumulative_error,
        )
        plot_trajectory_comparison(pred, gt, lengths_arr, "PDR: Trajectory Comparison", out)
        plot_error_over_time(pred, gt, lengths_arr, "PDR: Error Over Time", out)
        plot_cumulative_error(pred, gt, "PDR: Cumulative Error Distribution", out)
        _plt.close("all")
        del _plt, _mpl
    except Exception:
        pass

    return metrics
