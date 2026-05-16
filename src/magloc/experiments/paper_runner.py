from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from magloc.data.io import list_npy_files
from magloc.experiments.backbone_compare import run_backbone_compare
from magloc.experiments.evaluate import evaluate_regression, evaluate_retrieval, evaluate_seqdec
from magloc.experiments.finetune import run_finetune
from magloc.experiments.paper_csv import (
    make_error_rows,
    read_errors_from_npz,
    summarize_rows,
    write_error_csv,
    write_summary_csv,
    write_summary_json,
)
from magloc.experiments.pretrain import run_pretrain
from magloc.utils import ensure_dir, load_yaml, save_yaml


E1_METHODS = {
    "rnn": ("RNN", "RNN", "rnn"),
    "lstm": ("LSTM", "LSTM", "lstm"),
    "cnn_tcn": ("CNN+TCN", "CNNTCN", "cnn_tcn"),
    "convnext_lite_1d": ("ConvNeXt-Lite-1D", "CONV", "convnext_lite_1d"),
}


def infer_scene_label_and_code(cfg: Dict, scene_label: str | None = None, scene_code: str | None = None) -> tuple[str, str]:
    if scene_label and scene_code:
        return scene_label, scene_code.upper()
    name = str(cfg.get("scene", {}).get("name", "scenario_1"))
    filt = str(cfg.get("scene", {}).get("scene_filter", ""))
    is_s2 = "2" in name or "文管" in filt
    label = scene_label or ("场景2" if is_s2 else "场景1")
    code = (scene_code or ("S2" if is_s2 else "S1")).upper()
    return label, code


def _out_root(cfg: Dict) -> Path:
    return Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"]


def _pretrain_best(cfg: Dict) -> Path:
    return _out_root(cfg) / "pretrain" / "pretrain_best.pth"


def _finetune_best(cfg: Dict, scratch: bool = False) -> Path:
    return _out_root(cfg) / ("finetune_scratch" if scratch else "finetune") / "regression_best.pth"


def _eval_reg_npz(cfg: Dict, split: str = "test", scratch: bool = False) -> Path:
    sub = "eval_regression_scratch" if scratch else "eval_regression"
    return _out_root(cfg) / sub / f"{split}_regression_preds.npz"


def _eval_ret_npz(cfg: Dict, split: str = "test") -> Path:
    return _out_root(cfg) / "eval_retrieval" / f"{split}_retrieval_candidates.npz"


def _eval_seq_npz(cfg: Dict, split: str = "test", variant: str | None = None) -> Path:
    sub = "eval_seqdec" if not variant else f"eval_seqdec_{variant}"
    return _out_root(cfg) / sub / f"{split}_seqdec_preds.npz"


def _deep_update(base: Dict, updates: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _variant_config(base_config_path: str, tag: str, updates: Dict, work_dir: str | Path) -> tuple[Dict, Path]:
    base = load_yaml(base_config_path)
    base_scene = base["scene"]["name"]
    cfg = _deep_update(base, updates)
    cfg["scene"]["name"] = f"{base_scene}_{tag}"
    path = Path(work_dir) / "configs" / f"{cfg['scene']['name']}.yaml"
    ensure_dir(path.parent)
    save_yaml(cfg, path)
    return cfg, path


def _variant_config_exclude_device(
    base_config_path: str,
    held_out: str,
    work_dir: str | Path,
) -> tuple[Dict, Path]:
    """Variant that excludes `held_out` device from train/val via scene_filter on the directory name."""
    base = load_yaml(base_config_path)
    base_scene = base["scene"]["name"]
    cfg = _deep_update(base, {"scene": {"name": f"{base_scene}_lodo_exclude_{held_out}", "scene_filter": held_out}})
    path = Path(work_dir) / "configs" / f"{cfg['scene']['name']}.yaml"
    ensure_dir(path.parent)
    save_yaml(cfg, path)
    return cfg, path


def _discover_devices(cfg: Dict) -> List[str]:
    """Infer available device names from base config's train split directory via file names."""
    data_root = Path(cfg["paths"]["data_root"])
    train_dir = data_root / cfg["split"].get("train_dir", "train")
    if not train_dir.exists():
        return []
    from magloc.data.io import infer_device_name
    files = list_npy_files(train_dir, pattern=cfg["split"].get("file_pattern", "*.npy"))
    devices = sorted({infer_device_name(f) for f in files})
    return devices


class PaperExperimentRunner:
    """One-stop runner for thesis experiments and the uploaded-CSV-compatible error table."""

    def __init__(
        self,
        config_path: str,
        csv_path: str | Path | None = None,
        summary_csv: str | Path | None = None,
        scene_label: str | None = None,
        scene_code: str | None = None,
        encoding: str = "gbk",
        force: bool = False,
        collect_only: bool = False,
    ):
        self.config_path = str(config_path)
        self.cfg = load_yaml(config_path)
        self.scene_label, self.scene_code = infer_scene_label_and_code(self.cfg, scene_label, scene_code)
        root = _out_root(self.cfg)
        self.result_root = ensure_dir(root / "paper_csv")
        self.csv_path = Path(csv_path) if csv_path else self.result_root / "paper_error_curves.csv"
        self.summary_csv = Path(summary_csv) if summary_csv else self.result_root / "paper_summary_metrics.csv"
        self.summary_json = self.summary_csv.with_suffix(".json")
        self.encoding = encoding
        self.force = force
        self.collect_only = collect_only
        self.rows: List[Dict[str, object]] = []

    def _add_npz(self, npz_path: str | Path, figure_id: str, chapter_section: str, method: str, curve_key: str) -> bool:
        p = Path(npz_path)
        if not p.exists():
            print(f"[csv] skip missing result: {p}")
            return False
        errors = read_errors_from_npz(p)
        self.rows.extend(make_error_rows(errors, figure_id, chapter_section, self.scene_label, method, curve_key))
        print(f"[csv] added {len(errors)} rows: {figure_id} / {method} / {curve_key}")
        return True

    def _add_errors(self, errors: Sequence[float], figure_id: str, chapter_section: str, method: str, curve_key: str) -> None:
        self.rows.extend(make_error_rows(errors, figure_id, chapter_section, self.scene_label, method, curve_key))
        print(f"[csv] added {len(errors)} rows: {figure_id} / {method} / {curve_key}")

    def _ensure_pretrain(self, cfg_path: str, cfg: Dict) -> Path:
        p = _pretrain_best(cfg)
        if self.collect_only:
            return p
        if self.force or not p.exists():
            run_pretrain(cfg_path)
        else:
            print(f"[reuse] pretrain ckpt: {p}")
        return p

    def _ensure_finetune(self, cfg_path: str, cfg: Dict, pretrained: Path | None = None, scratch: bool = False) -> Path:
        p = _finetune_best(cfg, scratch=scratch)
        if self.collect_only:
            return p
        if self.force or not p.exists():
            if scratch:
                run_finetune(cfg_path, pretrained_ckpt=None, scratch=True)
            else:
                run_finetune(cfg_path, pretrained_ckpt=str(pretrained or _pretrain_best(cfg)), scratch=False)
        else:
            print(f"[reuse] finetune ckpt: {p}")
        return p

    def _ensure_eval_regression(self, cfg_path: str, cfg: Dict, ckpt: Path, scratch: bool = False) -> Path:
        p = _eval_reg_npz(cfg, scratch=scratch)
        if self.collect_only:
            return p
        if self.force or not p.exists():
            out = _out_root(cfg) / ("eval_regression_scratch" if scratch else "eval_regression")
            evaluate_regression(cfg_path, str(ckpt), output_dir=str(out))
        else:
            print(f"[reuse] regression pred: {p}")
        return p

    def _ensure_eval_retrieval(self, cfg_path: str, cfg: Dict, encoder_ckpt: Path) -> Path:
        p = _eval_ret_npz(cfg)
        if self.collect_only:
            return p
        if self.force or not p.exists():
            evaluate_retrieval(cfg_path, str(encoder_ckpt))
        else:
            print(f"[reuse] retrieval candidates: {p}")
        return p

    def _ensure_eval_seqdec(self, cfg_path: str, cfg: Dict, encoder_ckpt: Path, variant: str | None = None) -> Path:
        p = _eval_seq_npz(cfg, variant=variant)
        if self.collect_only:
            return p
        if self.force or not p.exists():
            out = _out_root(cfg) / ("eval_seqdec" if not variant else f"eval_seqdec_{variant}")
            evaluate_seqdec(cfg_path, str(encoder_ckpt), output_dir=str(out))
        else:
            print(f"[reuse] seqdec pred: {p}")
        return p

    def ensure_base_pretrain_mlp_retrieval_seqdec(self) -> Dict[str, Path]:
        pre = self._ensure_pretrain(self.config_path, self.cfg)
        ft = self._ensure_finetune(self.config_path, self.cfg, pre, scratch=False)
        reg = self._ensure_eval_regression(self.config_path, self.cfg, ft, scratch=False)
        ret = self._ensure_eval_retrieval(self.config_path, self.cfg, pre)
        seq = self._ensure_eval_seqdec(self.config_path, self.cfg, pre)
        return {"pretrain": pre, "finetune": ft, "regression_npz": reg, "retrieval_npz": ret, "seqdec_npz": seq}

    def run_e1_backbone(self) -> None:
        out = _out_root(self.cfg) / "backbone_compare"
        missing = [safe for _, _, safe in E1_METHODS.values() if not (out / safe / "test_preds.npz").exists()]
        if not self.collect_only and (self.force or missing):
            run_backbone_compare(self.config_path)
        fig = "3.7.2_E1_backbone"
        sec = "3.7.2 backbone监督定位对比"
        for key, (method, curve_suffix, safe) in E1_METHODS.items():
            self._add_npz(out / safe / "test_preds.npz", fig, sec, method, f"E1_{self.scene_code}_{curve_suffix}")

    def run_e2_pretrain(self) -> None:
        pre = self._ensure_pretrain(self.config_path, self.cfg)
        scratch_ckpt = self._ensure_finetune(self.config_path, self.cfg, None, scratch=True)
        scratch_npz = self._ensure_eval_regression(self.config_path, self.cfg, scratch_ckpt, scratch=True)
        pt_ckpt = self._ensure_finetune(self.config_path, self.cfg, pre, scratch=False)
        pt_npz = self._ensure_eval_regression(self.config_path, self.cfg, pt_ckpt, scratch=False)
        fig = "3.7.3_E2_pretrain"
        sec = "3.7.3 对比学习预训练有效性验证"
        self._add_npz(scratch_npz, fig, sec, "Scratch + MLP", f"SCRATCH_MLP_{self.scene_code}")
        self._add_npz(pt_npz, fig, sec, "Pretrain + MLP", f"PRETRAIN_MLP_{self.scene_code}")

    def _run_lodo_fold(
        self,
        held_out: str,
        curve_suffix: str,
        fig: str,
        sec: str,
    ) -> tuple[list[float], list[float], list[float]]:
        """Run one LODO fold: train on all-but-held_out, test on held_out device only."""
        work = self.result_root
        lcfg, lcfg_path = _variant_config_exclude_device(self.config_path, held_out, work)
        lpre = self._ensure_pretrain(str(lcfg_path), lcfg)
        lft = self._ensure_finetune(str(lcfg_path), lcfg, lpre, scratch=False)
        lreg = self._ensure_eval_regression(str(lcfg_path), lcfg, lft, scratch=False)
        lret = self._ensure_eval_retrieval(str(lcfg_path), lcfg, lpre)
        lseq = self._ensure_eval_seqdec(str(lcfg_path), lcfg, lpre)

        faiss_errs, mlp_errs, seq_errs = [], [], []
        if Path(lret).exists():
            faiss_errs = read_errors_from_npz(lret).tolist()
            self._add_errors(faiss_errs, fig, sec, "LODO Pretrain + FAISS", f"FAISS_{curve_suffix}")
        if Path(lreg).exists():
            mlp_errs = read_errors_from_npz(lreg).tolist()
            self._add_errors(mlp_errs, fig, sec, "LODO Pretrain + MLP", f"MLP_{curve_suffix}")
        if Path(lseq).exists():
            seq_errs = read_errors_from_npz(lseq).tolist()
            self._add_errors(seq_errs, fig, sec, "LODO Pretrain + SeqDec", f"SEQ_{curve_suffix}")
        return faiss_errs, mlp_errs, seq_errs

    def run_e3_downstream(self) -> None:
        devices = _discover_devices(self.cfg)
        if not devices:
            print("[lodo] no devices found; skipping LODO experiments.")
            return
        fig = "3.7.5_downstream_cross_device"
        sec = "3.7.5 下游定位方式与跨设备泛化综合实验"
        self.ensure_base_pretrain_mlp_retrieval_seqdec()
        self._add_npz(Path(self.cfg["paths"]["output_root"]) / self.cfg["scene"]["name"] / "eval_retrieval" / "test_retrieval_candidates.npz",
                      fig, sec, "In-Device Pretrain + FAISS", f"FAISS_ID_{self.scene_code}")
        self._add_npz(Path(self.cfg["paths"]["output_root"]) / self.cfg["scene"]["name"] / "eval_regression" / "test_regression_preds.npz",
                      fig, sec, "In-Device Pretrain + MLP", f"MLP_ID_{self.scene_code}")
        all_faiss, all_mlp = [], []
        for device in devices:
            code = device.upper()[:6].replace(" ", "")
            curve_suffix = f"LODO_{code}_{self.scene_code}"
            faiss_errs, mlp_errs, _ = self._run_lodo_fold(device, curve_suffix, fig, sec)
            all_faiss.extend(faiss_errs)
            all_mlp.extend(mlp_errs)
        if all_faiss:
            self._add_errors(all_faiss, fig, sec, "LODO Pretrain + FAISS (all)", f"FAISS_LODO_ALL_{self.scene_code}")
        if all_mlp:
            self._add_errors(all_mlp, fig, sec, "LODO Pretrain + MLP (all)", f"MLP_LODO_ALL_{self.scene_code}")

    def run_a1_equal_distance(self) -> None:
        base = self.ensure_base_pretrain_mlp_retrieval_seqdec()
        work = self.result_root
        wo_cfg, wo_path = _variant_config(
            self.config_path,
            "a1_wo_equal_distance",
            {"preprocess": {"window_mode": "fixed_time"}},
            work,
        )
        pre = self._ensure_pretrain(str(wo_path), wo_cfg)
        ret = self._ensure_eval_retrieval(str(wo_path), wo_cfg, pre)
        fig = "3.7.6_A1_equal_distance"
        sec = "3.7.6 A1 等空间窗口构造消融"
        self._add_npz(base["retrieval_npz"], fig, sec, "Full Model", f"A1_FULL_{self.scene_code}")
        self._add_npz(ret, fig, sec, "w/o Equal-Distance", f"A1_WO_ED_{self.scene_code}")

    def run_a2_representation_ablation(self) -> None:
        base = self.ensure_base_pretrain_mlp_retrieval_seqdec()
        work = self.result_root
        no_aug_cfg, no_aug_path = _variant_config(
            self.config_path,
            "a2_wo_augmentation",
            {
                "augmentation": {
                    "rotation_max_deg": 0, "noise_sigma": 0, "crop_ratio_min": 1.0, "crop_ratio_max": 1.0,
                    "grid_jitter_prob": 0.0, "grid_jitter_std": 0.0, "channel_dropout_prob": 0.0, "channel_shuffle_prob": 0.0,
                }
            },
            work,
        )
        no_lvf_cfg, no_lvf_path = _variant_config(
            self.config_path,
            "a2_wo_local_variation",
            {"preprocess": {"msfe": False}, "model": {"in_channels": 3}},
            work,
        )
        no_aug_pre = self._ensure_pretrain(str(no_aug_path), no_aug_cfg)
        no_aug_ret = self._ensure_eval_retrieval(str(no_aug_path), no_aug_cfg, no_aug_pre)
        no_lvf_pre = self._ensure_pretrain(str(no_lvf_path), no_lvf_cfg)
        no_lvf_ret = self._ensure_eval_retrieval(str(no_lvf_path), no_lvf_cfg, no_lvf_pre)
        fig = "3.7.6_A2_representation_ablation"
        sec = "3.7.6 A2 表征增强机制综合消融"
        self._add_npz(base["retrieval_npz"], fig, sec, "Full Model", f"A2_FULL_{self.scene_code}")
        self._add_npz(no_aug_ret, fig, sec, "w/o Augmentation", f"A2_WO_AUG_{self.scene_code}")
        self._add_npz(no_lvf_ret, fig, sec, "w/o Local-Variation Features", f"A2_WO_LVF_{self.scene_code}")

    def run_continuous(self) -> None:
        devices = _discover_devices(self.cfg)
        base = self.ensure_base_pretrain_mlp_retrieval_seqdec()
        fig = "4.7.3_continuous"
        sec = "4.7.3 整体连续定位性能对比"
        self._add_npz(base["regression_npz"], fig, sec, "MagCLR + Regression (In-Device)", f"MLP_ID_{self.scene_code}")
        self._add_npz(base["retrieval_npz"], fig, sec, "MagCLR + Top-K Weighted Fusion (In-Device)", f"FAISS_ID_{self.scene_code}")
        self._add_npz(base["seqdec_npz"], fig, sec, "SeqDec (In-Device)", f"SEQ_ID_{self.scene_code}")
        if not devices:
            print("[continuous] no devices found; skipping LODO continuous variants.")
            return
        all_faiss, all_mlp, all_seq = [], [], []
        for device in devices:
            code = device.upper()[:6].replace(" ", "")
            curve_suffix = f"LODO_{code}_{self.scene_code}"
            faiss_errs, mlp_errs, seq_errs = self._run_lodo_fold(device, curve_suffix, fig, sec)
            all_faiss.extend(faiss_errs)
            all_mlp.extend(mlp_errs)
            all_seq.extend(seq_errs)
        if all_faiss:
            self._add_errors(all_faiss, fig, sec, "MagCLR + Top-K Weighted Fusion (LODO)", f"FAISS_LODO_{self.scene_code}")
        if all_mlp:
            self._add_errors(all_mlp, fig, sec, "MagCLR + Regression (LODO)", f"MLP_LODO_{self.scene_code}")
        if all_seq:
            self._add_errors(all_seq, fig, sec, "SeqDec (LODO)", f"SEQ_LODO_{self.scene_code}")

    def run_transition(self) -> None:
        base = self.ensure_base_pretrain_mlp_retrieval_seqdec()
        work = self.result_root
        no_disp_cfg, no_disp_path = _variant_config(
            self.config_path,
            "seq_wo_displacement",
            {"seqdec": {"use_displacement": False, "use_jump_suppression": True, "use_confidence": True}},
            work,
        )
        no_jump_cfg, no_jump_path = _variant_config(
            self.config_path,
            "seq_wo_jump",
            {"seqdec": {"use_displacement": True, "use_jump_suppression": False, "use_confidence": True}},
            work,
        )
        # Reuse the same base pretrained encoder for transition ablations; only decoding rules change.
        pre = base["pretrain"]
        no_disp_npz = self._ensure_eval_seqdec(str(no_disp_path), no_disp_cfg, pre, variant="wo_displacement")
        no_jump_npz = self._ensure_eval_seqdec(str(no_jump_path), no_jump_cfg, pre, variant="wo_jump")
        fig = "4.7.5_transition_support"
        sec = "4.7.5 状态转移约束消融实验"
        self._add_npz(base["seqdec_npz"], fig, sec, "SeqDec (Full)", f"SEQ_FULL_{self.scene_code}")
        self._add_npz(no_disp_npz, fig, sec, "SeqDec (w/o Displacement Consistency)", f"SEQ_WO_DISP_{self.scene_code}")
        self._add_npz(no_jump_npz, fig, sec, "SeqDec (w/o Jump Suppression)", f"SEQ_WO_JUMP_{self.scene_code}")

    def run(self, experiments: Sequence[str]) -> tuple[Path, Path]:
        exp_set = [e.strip().lower() for e in experiments]
        all_names = ["e1", "e2", "e3", "a1", "a2", "continuous", "transition"]
        if not exp_set or "all" in exp_set:
            exp_set = all_names
        for exp in exp_set:
            if exp == "e1":
                self.run_e1_backbone()
            elif exp == "e2":
                self.run_e2_pretrain()
            elif exp == "e3":
                self.run_e3_downstream()
            elif exp == "a1":
                self.run_a1_equal_distance()
            elif exp == "a2":
                self.run_a2_representation_ablation()
            elif exp in {"continuous", "seq", "4.7.3"}:
                self.run_continuous()
            elif exp in {"transition", "4.7.5"}:
                self.run_transition()
            else:
                raise ValueError(f"unknown experiment {exp!r}")
        write_error_csv(self.rows, self.csv_path, encoding=self.encoding)
        summary = summarize_rows(self.rows)
        write_summary_csv(summary, self.summary_csv, encoding=self.encoding)
        write_summary_json(summary, self.summary_json)
        print(f"[done] paper error csv: {self.csv_path}")
        print(f"[done] summary csv: {self.summary_csv}")
        return self.csv_path, self.summary_csv


def run_paper_experiments(
    config_path: str,
    experiments: Sequence[str],
    csv_path: str | Path | None = None,
    summary_csv: str | Path | None = None,
    scene_label: str | None = None,
    scene_code: str | None = None,
    encoding: str = "gbk",
    force: bool = False,
    collect_only: bool = False,
) -> tuple[Path, Path]:
    runner = PaperExperimentRunner(
        config_path,
        csv_path=csv_path,
        summary_csv=summary_csv,
        scene_label=scene_label,
        scene_code=scene_code,
        encoding=encoding,
        force=force,
        collect_only=collect_only,
    )
    return runner.run(experiments)


def collect_existing_results_to_csv(
    config_path: str,
    csv_path: str | Path | None = None,
    summary_csv: str | Path | None = None,
    scene_label: str | None = None,
    scene_code: str | None = None,
    encoding: str = "gbk",
) -> tuple[Path, Path]:
    return run_paper_experiments(
        config_path,
        experiments=["all"],
        csv_path=csv_path,
        summary_csv=summary_csv,
        scene_label=scene_label,
        scene_code=scene_code,
        encoding=encoding,
        collect_only=True,
    )
