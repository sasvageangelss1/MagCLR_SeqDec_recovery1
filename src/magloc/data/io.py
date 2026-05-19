from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


def infer_device_name(path: str | Path) -> str:
    """Infer device name from file name/path used in the early project.

    Examples: 12-25-信息12-25-Huawei P60004.npy -> Huawei P60
    """
    s = str(path)
    candidates = [
        "Honor 200", "Huawei P60", "Huawei P70", "MEIZU 20", "OPPO Find X", "XiaoMi 14", "Xiaomi 14",
    ]
    low = s.lower()
    for c in candidates:
        if c.lower() in low:
            return c.replace("Xiaomi", "XiaoMi")
    # fallback: chunk between final 12-25- and trailing digits
    m = re.search(r"12-25-([^/\\]+?)(\d{3})?\.npy$", s)
    return m.group(1) if m else "unknown"


def list_npy_files(root: str | Path, pattern: str = "*.npy", scene_filter: Optional[str] = None) -> List[Path]:
    files = [Path(p) for p in glob.glob(str(Path(root) / pattern))]
    if scene_filter:
        files = [p for p in files if scene_filter in p.name or scene_filter in str(p.parent)]
    return sorted(files)


def load_npy_trajectory(path: str | Path) -> np.ndarray:
    """Load a .npy trajectory as a (T, >=5) array and return the first 5 columns.

    Supports two formats:
      - Old:  (T, 5)  → [mx, my, mz, pos_x, pos_y]
      - New:  (T, 15) → [mx,my,mz, pos_x,pos_y, epoch, accX..Z, gyroX..Z, gravX..Z]

    Only the first 5 columns (magnetic + position) are returned, so existing
    pipeline code is unaffected.
    """
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} has shape {arr.shape}; expected 2D array.")
    if arr.shape[1] < 5:
        raise ValueError(
            f"{path} has shape {arr.shape}; expected at least 5 columns "
            "[mx,my,mz,x,y] or 15 columns with full IMU."
        )
    return arr[:, :5].astype(np.float32)


def load_npy_trajectory_full(path: str | Path) -> dict[str, np.ndarray]:
    """Load a raw .npy trajectory returning all IMU fields.

    Column layout (15 total):
      0-2  : magX, magY, magZ
      3-4  : pos_x, pos_y
      5    : epochMillis
      6-8  : accX, accY, accZ
      9-11 : gyroX, gyroY, gyroZ
      12-14: gravityX, gravityY, gravityZ
    """
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} has shape {arr.shape}; expected 2D array.")
    if arr.shape[1] < 15:
        raise ValueError(
            f"{path} has shape {arr.shape}; expected at least 15 columns "
            "[mx,my,mz,x,y,epoch,ax,ay,az,gx,gy,gz,gvX,gvY,gvZ]."
        )
    arr = arr[:, :15].astype(np.float64)
    return dict(
        magX=arr[:, 0], magY=arr[:, 1], magZ=arr[:, 2],
        pos_x=arr[:, 3], pos_y=arr[:, 4],
        epochMillis=arr[:, 5],
        accX=arr[:, 6], accY=arr[:, 7], accZ=arr[:, 8],
        gyroX=arr[:, 9], gyroY=arr[:, 10], gyroZ=arr[:, 11],
        gravX=arr[:, 12], gravY=arr[:, 13], gravZ=arr[:, 14],
    )


def load_split(root: str | Path, split_name: str, pattern: str = "*.npy", scene_filter: Optional[str] = None) -> Tuple[List[np.ndarray], List[Path]]:
    split_dir = Path(root) / split_name
    files = list_npy_files(split_dir, pattern=pattern, scene_filter=scene_filter)
    if not files:
        raise FileNotFoundError(f"No npy files found in {split_dir} with pattern={pattern}, scene_filter={scene_filter!r}")
    arrays = [load_npy_trajectory(p) for p in files]
    return arrays, files


def load_split_full(root: str | Path, split_name: str, pattern: str = "*.npy", scene_filter: Optional[str] = None) -> Tuple[List[dict[str, np.ndarray]], List[Path]]:
    """Load all IMU fields for every trajectory in a split.

    Returns List of dicts from load_npy_trajectory_full(), plus the file paths.
    Used by PDR and other IMU-intensive baselines.
    """
    split_dir = Path(root) / split_name
    files = list_npy_files(split_dir, pattern=pattern, scene_filter=scene_filter)
    if not files:
        raise FileNotFoundError(f"No npy files found in {split_dir}")
    arrays = [load_npy_trajectory_full(p) for p in files]
    return arrays, files
