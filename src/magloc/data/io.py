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
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"{path} has shape {arr.shape}; expected at least 5 columns [mx,my,mz,x,y].")
    return arr[:, :5].astype(np.float32)


def load_split(root: str | Path, split_name: str, pattern: str = "*.npy", scene_filter: Optional[str] = None) -> Tuple[List[np.ndarray], List[Path]]:
    split_dir = Path(root) / split_name
    files = list_npy_files(split_dir, pattern=pattern, scene_filter=scene_filter)
    if not files:
        raise FileNotFoundError(f"No npy files found in {split_dir} with pattern={pattern}, scene_filter={scene_filter!r}")
    arrays = [load_npy_trajectory(p) for p in files]
    return arrays, files
