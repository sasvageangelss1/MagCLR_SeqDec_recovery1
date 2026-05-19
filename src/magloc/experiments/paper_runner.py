"""
论文实验一键运行器（PaperExperimentRunner）。

功能：
  1. 统一管理所有论文实验（E1~E3 性能实验、A1~A2 消融实验、continuous 连续定位、transition 转移约束）
  2. 自动跳过已完成的步骤（复用已有检查点/NPZ 结果）
  3. 将所有误差结果统一导出为与论文上传格式兼容的 CSV 文件

实验说明：
  E1  - 主干网络对比：对比 RNN / LSTM / CNN+TCN / ConvNeXt 等有监督定位基线
  E2  - 预训练有效性：Scratch（无预训练）vs Pretrain（有预训练），验证对比学习的作用
  E3  - 下游定位方式综合实验 + 跨设备 Leave-One-Device-Out（LODO）泛化评估
  A1  - 等空间窗口构造消融：对比使用/不使用等距采样窗口的检索性能
  A2  - 表征增强机制消融：对比使用/不使用数据增强（旋转/噪声）、使用/不使用局部变异特征（MSFE）
  continuous - 整体连续定位性能：WKNN 指纹、PDR 步行航位推算、MagCLR+Regression、MagCLR+FAISS、SeqDec
  transition - 状态转移约束消融：位移一致性约束、跳点抑制机制各自的重要性

LODO（Leave-One-Device-Out）策略：
  每个折叠留出一个设备的数据作为测试集，其余设备数据合并为训练集，
  用于评估模型对未见设备的泛化能力。
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from magloc.data.io import list_npy_files, load_split
from magloc.experiments.backbone_compare import run_backbone_compare
from magloc.experiments.evaluate import evaluate_regression, evaluate_retrieval, evaluate_seqdec
from magloc.experiments.evaluate_baselines import evaluate_wknn, evaluate_pdr
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


# E1 主干网络对比实验中各方法的元数据：(展示名称, 曲线后缀, 目录安全名称)
E1_METHODS = {
    "rnn": ("RNN", "RNN", "rnn"),
    "lstm": ("LSTM", "LSTM", "lstm"),
    "cnn_tcn": ("CNN+TCN", "CNNTCN", "cnn_tcn"),
    "convnext_lite_1d": ("ConvNeXt-Lite-1D", "CONV", "convnext_lite_1d"),
}


def infer_scene_label_and_code(
    cfg: Dict,
    scene_label: str | None = None,
    scene_code: str | None = None,
) -> tuple[str, str]:
    """
    从配置文件自动推断场景标签和代码。

    规则：
      - 若用户通过命令行指定了 scene_label / scene_code，直接返回
      - 否则根据 scene.name 或 scene.scene_filter 中的关键词判断：
          含 "2" 或 "文管" → 场景2 / S2
          其他 → 场景1 / S1
    """
    if scene_label and scene_code:
        return scene_label, scene_code.upper()
    name = str(cfg.get("scene", {}).get("name", "scenario_1"))
    filt = str(cfg.get("scene", {}).get("scene_filter", ""))
    is_s2 = "2" in name or "文管" in filt
    label = scene_label or ("场景2" if is_s2 else "场景1")
    code = (scene_code or ("S2" if is_s2 else "S1")).upper()
    return label, code


def _out_root(cfg: Dict) -> Path:
    """获取实验输出根目录：{output_root}/{scene_name}/"""
    return Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"]


def _pretrain_best(cfg: Dict) -> Path:
    """预训练最佳检查点的默认路径。"""
    return _out_root(cfg) / "pretrain" / "pretrain_best.pth"


def _finetune_best(cfg: Dict, scratch: bool = False) -> Path:
    """微调最佳检查点的默认路径；scratch=True 对应从头训练目录。"""
    return _out_root(cfg) / ("finetune_scratch" if scratch else "finetune") / "regression_best.pth"


def _eval_reg_npz(cfg: Dict, split: str = "test", scratch: bool = False) -> Path:
    """回归预测 NPZ 文件路径（由 evaluate_regression 产生）。"""
    sub = "eval_regression_scratch" if scratch else "eval_regression"
    return _out_root(cfg) / sub / f"{split}_regression_preds.npz"


def _eval_ret_npz(cfg: Dict, split: str = "test") -> Path:
    """检索候选 NPZ 文件路径（由 evaluate_retrieval 产生）。"""
    return _out_root(cfg) / "eval_retrieval" / f"{split}_retrieval_candidates.npz"


def _eval_seq_npz(cfg: Dict, split: str = "test", variant: str | None = None) -> Path:
    """SeqDec 轨迹预测 NPZ 文件路径；variant 用于区分不同解码策略的子目录。"""
    sub = "eval_seqdec" if not variant else f"eval_seqdec_{variant}"
    return _out_root(cfg) / sub / f"{split}_seqdec_preds.npz"


def _deep_update(base: Dict, updates: Dict) -> Dict:
    """
    递归深拷贝并合并字典，用于从 base config 生成 variant config。
    若 updates 中的键也是字典，则递归合并；否则直接覆盖。
    """
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _variant_config(
    base_config_path: str,
    tag: str,
    updates: Dict,
    work_dir: str | Path,
) -> tuple[Dict, Path]:
    """
    基于基础配置 + 指定参数更新，生成一个实验变体配置文件。

    参数:
        tag: 标识本次变体的字符串，会拼接到 scene.name 中（如 "a1_wo_equal_distance"）
        updates: 覆盖参数的字典（如 {"preprocess": {"window_mode": "fixed_time"}}）
        work_dir: 变体配置文件的保存目录
    返回:
        (cfg, cfg_path): 解析后的配置字典和配置文件路径
    """
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
    """
    生成 LODO（Leave-One-Device-Out）实验的变体配置：
    在 scene_filter 中指定 held_out 设备名称，使得训练/验证数据中排除该设备的数据。
    这样可以保证测试集完全来自未见设备，公平评估跨设备泛化性能。
    """
    base = load_yaml(base_config_path)
    base_scene = base["scene"]["name"]
    cfg = _deep_update(
        base,
        {"scene": {"name": f"{base_scene}_lodo_exclude_{held_out}", "scene_filter": held_out}},
    )
    path = Path(work_dir) / "configs" / f"{cfg['scene']['name']}.yaml"
    ensure_dir(path.parent)
    save_yaml(cfg, path)
    return cfg, path


def _discover_devices(cfg: Dict) -> List[str]:
    """
    自动从训练集文件列表中推断可用的设备名称列表。
    用于 LODO 实验：遍历每个设备，将其作为 held-out 测试集。
    """
    data_root = Path(cfg["paths"]["data_root"])
    train_dir = data_root / cfg["split"].get("train_dir", "train")
    if not train_dir.exists():
        return []
    from magloc.data.io import infer_device_name
    files = list_npy_files(train_dir, pattern=cfg["split"].get("file_pattern", "*.npy"))
    devices = sorted({infer_device_name(f) for f in files})
    return devices


class PaperExperimentRunner:
    """
    论文实验统一运行器。

    核心逻辑：
      - 每个实验方法（如 run_e1_backbone）按需调用 _ensure_* 系列方法
      - _ensure_* 方法会检查目标文件是否已存在：
          若 --force=True 或文件不存在 → 执行实际的训练/评估
          否则 → 跳过并复用已有结果（节省时间）
      - 所有误差数据通过 _add_npz / _add_errors 收集到 self.rows 列表
      - run() 结束时统一导出为 paper_error_curves.csv 和 paper_summary_metrics.csv
    """

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
        self.scene_label, self.scene_code = infer_scene_label_and_code(
            self.cfg, scene_label, scene_code
        )
        root = _out_root(self.cfg)
        # 所有 CSV 输出统一保存在 paper_csv/ 子目录
        self.result_root = ensure_dir(root / "paper_csv")
        self.csv_path = Path(csv_path) if csv_path else self.result_root / "paper_error_curves.csv"
        self.summary_csv = Path(summary_csv) if summary_csv else self.result_root / "paper_summary_metrics.csv"
        self.summary_json = self.summary_csv.with_suffix(".json")
        self.encoding = encoding
        self.force = force
        self.collect_only = collect_only
        # 收集所有误差行，最后统一导出 CSV
        self.rows: List[Dict[str, object]] = []

    # ------------------------------------------------------------------ #
    # 辅助方法：将 NPZ 误差数据 / 原始误差列表转换为 CSV 行并加入 rows
    # ------------------------------------------------------------------ #

    def _add_npz(
        self,
        npz_path: str | Path,
        figure_id: str,
        chapter_section: str,
        method: str,
        curve_key: str,
    ) -> bool:
        """
        从 NPZ 文件读取误差序列，生成 CSV 行并加入 rows 列表。

        返回:
            True  if 文件存在且成功添加
            False if 文件不存在（跳过）
        """
        p = Path(npz_path)
        if not p.exists():
            print(f"[csv] skip missing result: {p}")
            return False
        errors = read_errors_from_npz(p)
        self.rows.extend(make_error_rows(
            errors, figure_id, chapter_section, self.scene_label, method, curve_key
        ))
        print(f"[csv] added {len(errors)} rows: {figure_id} / {method} / {curve_key}")
        return True

    def _add_errors(
        self,
        errors: Sequence[float],
        figure_id: str,
        chapter_section: str,
        method: str,
        curve_key: str,
    ) -> None:
        """直接将一列误差值（通常来自多个 LODO 折叠的拼接）加入 rows 列表。"""
        self.rows.extend(make_error_rows(
            errors, figure_id, chapter_section, self.scene_label, method, curve_key
        ))
        print(f"[csv] added {len(errors)} rows: {figure_id} / {method} / {curve_key}")

    # ------------------------------------------------------------------ #
    # _ensure_* 系列方法：按需执行训练/评估，复用已有结果
    # ------------------------------------------------------------------ #

    def _ensure_pretrain(self, cfg_path: str, cfg: Dict) -> Path:
        """确保预训练完成：若检查点不存在则运行 run_pretrain。"""
        p = _pretrain_best(cfg)
        if self.collect_only:
            return p
        if self.force or not p.exists():
            run_pretrain(cfg_path)
        else:
            print(f"[reuse] pretrain ckpt: {p}")
        return p

    def _ensure_finetune(
        self,
        cfg_path: str,
        cfg: Dict,
        pretrained: Path | None = None,
        scratch: bool = False,
    ) -> Path:
        """
        确保微调完成：若检查点不存在则运行 run_finetune。
        - scratch=True：传入 pretrained_ckpt=None，从头训练
        - scratch=False：传入预训练编码器路径，进行微调
        """
        p = _finetune_best(cfg, scratch=scratch)
        if self.collect_only:
            return p
        if self.force or not p.exists():
            if scratch:
                run_finetune(cfg_path, pretrained_ckpt=None, scratch=True)
            else:
                run_finetune(
                    cfg_path,
                    pretrained_ckpt=str(pretrained or _pretrain_best(cfg)),
                    scratch=False,
                )
        else:
            print(f"[reuse] finetune ckpt: {p}")
        return p

    def _ensure_eval_regression(
        self,
        cfg_path: str,
        cfg: Dict,
        ckpt: Path,
        scratch: bool = False,
    ) -> Path:
        """确保回归评估完成：若 NPZ 不存在则运行 evaluate_regression。"""
        p = _eval_reg_npz(cfg, scratch=scratch)
        if self.collect_only:
            return p
        if self.force or not p.exists():
            out = _out_root(cfg) / ("eval_regression_scratch" if scratch else "eval_regression")
            evaluate_regression(cfg_path, str(ckpt), output_dir=str(out))
        else:
            print(f"[reuse] regression pred: {p}")
        return p

    def _ensure_eval_retrieval(
        self,
        cfg_path: str,
        cfg: Dict,
        encoder_ckpt: Path,
    ) -> Path:
        """确保检索评估完成：若 NPZ 不存在则运行 evaluate_retrieval。"""
        p = _eval_ret_npz(cfg)
        if self.collect_only:
            return p
        if self.force or not p.exists():
            evaluate_retrieval(cfg_path, str(encoder_ckpt))
        else:
            print(f"[reuse] retrieval candidates: {p}")
        return p

    def _ensure_eval_seqdec(
        self,
        cfg_path: str,
        cfg: Dict,
        encoder_ckpt: Path,
        variant: str | None = None,
    ) -> Path:
        """
        确保 SeqDec 评估完成：若 NPZ 不存在则运行 evaluate_seqdec。
        variant 参数用于区分不同解码策略（如 "wo_displacement" / "wo_jump"）。
        """
        p = _eval_seq_npz(cfg, variant=variant)
        if self.collect_only:
            return p
        if self.force or not p.exists():
            out = _out_root(cfg) / ("eval_seqdec" if not variant else f"eval_seqdec_{variant}")
            evaluate_seqdec(cfg_path, str(encoder_ckpt), output_dir=str(out))
        else:
            print(f"[reuse] seqdec pred: {p}")
        return p

    def _ensure_eval_wknn(self, cfg_path: str, cfg: Dict) -> Path:
        """确保 WKNN（加权 K 近邻）基线评估完成。WKNN 无需训练，直接在测试集上做指纹检索。"""
        p = _out_root(cfg) / "eval_wknn" / f"{cfg['split'].get('test_dir', 'test')}_wknn_preds.npz"
        if self.collect_only:
            return p
        if self.force or not p.exists():
            out = _out_root(cfg) / "eval_wknn"
            evaluate_wknn(cfg_path, output_dir=str(out))
        else:
            print(f"[reuse] wknn pred: {p}")
        return p

    def _ensure_eval_pdr(self, cfg_path: str, cfg: Dict) -> Path:
        """确保 PDR（步行航位推算）基线评估完成。"""
        p = _out_root(cfg) / "eval_pdr" / f"{cfg['split'].get('test_dir', 'test')}_pdr_preds.npz"
        if self.collect_only:
            return p
        if self.force or not p.exists():
            from magloc.experiments.evaluate_baselines import evaluate_pdr
            pdr_cfg = cfg.get("pdr", {})
            evaluate_pdr(
                cfg_path,
                gyro_weight=float(pdr_cfg.get("gyro_weight", 0.97)),
                prominence=float(pdr_cfg.get("prominence", 0.5)),
                min_interval=int(pdr_cfg.get("min_interval", 15)),
                step_length=float(pdr_cfg.get("step_length", 0.65)),
            )
        else:
            print(f"[reuse] pdr pred: {p}")
        return p

    # ------------------------------------------------------------------ #
    # 基础流水线：预训练 → 微调 → 回归/检索/SeqDec 评估（In-Device）
    # ------------------------------------------------------------------ #

    def ensure_base_pretrain_mlp_retrieval_seqdec(self) -> Dict[str, Path]:
        """
        运行 MagCLR 的标准 In-Device（全设备）流水线：
        预训练 → 微调 → 回归评估 + 检索评估 + SeqDec 评估
        所有步骤均使用同一场景的全部设备数据（不做 LODO 划分）。
        返回各结果的 NPZ/检查点路径字典。
        """
        pre = self._ensure_pretrain(self.config_path, self.cfg)
        ft = self._ensure_finetune(self.config_path, self.cfg, pre, scratch=False)
        reg = self._ensure_eval_regression(self.config_path, self.cfg, ft, scratch=False)
        ret = self._ensure_eval_retrieval(self.config_path, self.cfg, pre)
        seq = self._ensure_eval_seqdec(self.config_path, self.cfg, pre)
        return {
            "pretrain": pre,
            "finetune": ft,
            "regression_npz": reg,
            "retrieval_npz": ret,
            "seqdec_npz": seq,
        }

    # ------------------------------------------------------------------ #
    # E1 实验：主干网络对比（有监督基线 vs MagCLR）
    # ------------------------------------------------------------------ #

    def run_e1_backbone(self) -> None:
        """
        E1 主干网络对比实验。

        对比多种有监督定位模型（RNN / LSTM / CNN+TCN / ConvNeXt-Lite-1D）
        与 MagCLR（基于预训练的编码器）的定位性能差异。
        结果导出至 paper_error_curves.csv，曲线 key 格式：E1_{S1/S2}_{方法缩写}
        """
        out = _out_root(self.cfg) / "backbone_compare"
        # 跳过已存在结果的模型（除非 --force）
        missing = [
            safe for _, _, safe in E1_METHODS.values()
            if not (out / safe / "test_preds.npz").exists()
        ]
        if not self.collect_only and (self.force or missing):
            run_backbone_compare(self.config_path)

        fig = "3.7.2_E1_backbone"
        sec = "3.7.2 backbone监督定位对比"
        for key, (method, curve_suffix, safe) in E1_METHODS.items():
            self._add_npz(
                out / safe / "test_preds.npz",
                fig, sec, method,
                f"E1_{self.scene_code}_{curve_suffix}",
            )

    # ------------------------------------------------------------------ #
    # E2 实验：预训练有效性验证
    # ------------------------------------------------------------------ #

    def run_e2_pretrain(self) -> None:
        """
        E2 对比学习预训练有效性验证。

        核心对比：
          Scratch + MLP：编码器随机初始化，不经过任何预训练，直接加回归头
          Pretrain + MLP：编码器先经过对比学习预训练，再加回归头微调

        若 Pretrain 明显优于 Scratch，说明对比学习成功学到了可迁移的磁力信号表征。
        """
        pre = self._ensure_pretrain(self.config_path, self.cfg)
        # Scratch 基线：不加载预训练权重
        scratch_ckpt = self._ensure_finetune(self.config_path, self.cfg, None, scratch=True)
        scratch_npz = self._ensure_eval_regression(self.config_path, self.cfg, scratch_ckpt, scratch=True)
        # Pretrain 基线：加载预训练权重
        pt_ckpt = self._ensure_finetune(self.config_path, self.cfg, pre, scratch=False)
        pt_npz = self._ensure_eval_regression(self.config_path, self.cfg, pt_ckpt, scratch=False)

        fig = "3.7.3_E2_pretrain"
        sec = "3.7.3 对比学习预训练有效性验证"
        self._add_npz(scratch_npz, fig, sec, "Scratch + MLP", f"SCRATCH_MLP_{self.scene_code}")
        self._add_npz(pt_npz,    fig, sec, "Pretrain + MLP", f"PRETRAIN_MLP_{self.scene_code}")

    # ------------------------------------------------------------------ #
    # LODO 单折叠运行（内部方法，供 E3 / continuous / transition 调用）
    # ------------------------------------------------------------------ #

    def _run_lodo_fold(
        self,
        held_out: str,
        curve_suffix: str,
        fig: str,
        sec: str,
    ) -> tuple[list[float], list[float], list[float]]:
        """
        运行一次 LODO（Leave-One-Device-Out）折叠实验。

        参数:
            held_out:    被排除的设备名称（测试集专用）
            curve_suffix: CSV 曲线 key 的后缀部分（包含设备名和场景码）
            fig / sec:   论文图表 ID 和章节名称（用于 CSV 归档）

        流程：
          1. 生成排除 held_out 的变体配置（训练集不含该设备数据）
          2. 在变体配置下运行完整的预训练→微调→评估流水线
          3. 将 FAISS 检索误差、MLP 回归误差、SeqDec 误差分别加入 CSV rows

        返回:
            (faiss_errs, mlp_errs, seq_errs): 三种定位方法的误差列表（用于后续聚合）
        """
        work = self.result_root
        lcfg, lcfg_path = _variant_config_exclude_device(self.config_path, held_out, work)

        # 安全检查：排除 held_out 后，若训练集或验证集为空则跳过该折叠
        try:
            train_arrays, train_files = load_split(
                Path(lcfg["paths"]["data_root"]),
                lcfg["split"].get("train_dir", "train"),
                pattern=lcfg["split"].get("file_pattern", "*.npy"),
                scene_filter=lcfg["scene"].get("scene_filter"),
            )
            val_arrays, val_files = load_split(
                Path(lcfg["paths"]["data_root"]),
                lcfg["split"].get("val_dir", "val"),
                pattern=lcfg["split"].get("file_pattern", "*.npy"),
                scene_filter=lcfg["scene"].get("scene_filter"),
            )
        except FileNotFoundError:
            print(f"[lodo] skip fold {held_out!r}: train or val is empty after exclusion.")
            return [], [], []

        if not train_files:
            print(f"[lodo] skip fold {held_out!r}: no train files remain after exclusion.")
            return [], [], []
        if not val_files:
            print(f"[lodo] skip fold {held_out!r}: no val files remain after exclusion.")
            return [], [], []

        # Verify test split has files before running any training/evaluation.
        # The held-out device may exist only in train/val (not in test), so we use
        # the ORIGINAL config (no scene_filter) to check the test split.
        try:
            test_root = Path(self.cfg["paths"]["data_root"])
            test_split_dir = self.cfg["split"].get("test_dir", "test")
            test_pattern = self.cfg["split"].get("file_pattern", "*.npy")
            test_scene_filter = self.cfg.get("scene", {}).get("scene_filter") or None
            test_files_check = list_npy_files(
                test_root / test_split_dir,
                pattern=test_pattern,
                scene_filter=test_scene_filter,
            )
        except Exception:
            test_files_check = []
        if not test_files_check:
            print(f"[lodo] skip fold {held_out!r}: test split is empty (device not present in test set).")
            return [], [], []

        # 在排除 held_out 的配置下执行完整流水线
        lpre = self._ensure_pretrain(str(lcfg_path), lcfg)
        lft = self._ensure_finetune(str(lcfg_path), lcfg, lpre, scratch=False)
        try:
            lreg = self._ensure_eval_regression(str(lcfg_path), lcfg, lft, scratch=False)
            lret = self._ensure_eval_retrieval(str(lcfg_path), lcfg, lpre)
            lseq = self._ensure_eval_seqdec(str(lcfg_path), lcfg, lpre)
        except FileNotFoundError as e:
            print(f"[lodo] skip fold {held_out!r}: test split not found after exclusion. ({e})")
            return [], [], []

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

    # ------------------------------------------------------------------ #
    # E3 实验：下游定位方式与跨设备泛化综合实验
    # ------------------------------------------------------------------ #

    def run_e3_downstream(self) -> None:
        """
        E3 下游定位方式与跨设备泛化综合实验。

        两部分：
          1. In-Device 基准：使用全部设备数据训练/测试，评估三种定位方式
          2. LODO 跨设备泛化：对每个设备执行一次 LODO，汇总所有折叠误差

        对比 In-Device vs LODO 的性能差距，可量化 MagCLR 对未见设备的泛化能力。
        """
        devices = _discover_devices(self.cfg)
        if not devices:
            print("[lodo] no devices found; skipping LODO experiments.")
            return

        fig = "3.7.5_downstream_cross_device"
        sec = "3.7.5 下游定位方式与跨设备泛化综合实验"

        # In-Device 基准（使用全部设备数据训练/测试）
        self.ensure_base_pretrain_mlp_retrieval_seqdec()
        self._add_npz(
            Path(self.cfg["paths"]["output_root"]) / self.cfg["scene"]["name"]
            / "eval_retrieval" / "test_retrieval_candidates.npz",
            fig, sec, "In-Device Pretrain + FAISS",
            f"FAISS_ID_{self.scene_code}",
        )
        self._add_npz(
            Path(self.cfg["paths"]["output_root"]) / self.cfg["scene"]["name"]
            / "eval_regression" / "test_regression_preds.npz",
            fig, sec, "In-Device Pretrain + MLP",
            f"MLP_ID_{self.scene_code}",
        )

        # LODO 跨设备泛化：遍历每个设备，评估泛化误差
        all_faiss, all_mlp = [], []
        for device in devices:
            code = device.upper()[:6].replace(" ", "")   # 设备名截取前6字符作为代号
            curve_suffix = f"LODO_{code}_{self.scene_code}"
            faiss_errs, mlp_errs, _ = self._run_lodo_fold(device, curve_suffix, fig, sec)
            all_faiss.extend(faiss_errs)
            all_mlp.extend(mlp_errs)

        # 汇总所有 LODO 折叠的误差（等效于在未见设备上的整体误差）
        if all_faiss:
            self._add_errors(all_faiss, fig, sec, "LODO Pretrain + FAISS (all)",
                             f"FAISS_LODO_ALL_{self.scene_code}")
        if all_mlp:
            self._add_errors(all_mlp, fig, sec, "LODO Pretrain + MLP (all)",
                             f"MLP_LODO_ALL_{self.scene_code}")

    # ------------------------------------------------------------------ #
    # A1 实验：等空间窗口构造消融
    # ------------------------------------------------------------------ #

    def run_a1_equal_distance(self) -> None:
        """
        A1 等空间窗口构造消融实验。

        对比：
          Full Model：使用等距采样窗口（equal-distance sampling）构造训练数据
          w/o Equal-Distance：不使用等距采样，使用原始时间窗口

        等距采样确保训练窗口在空间上均匀分布，有助于提高检索召回率。
        """
        base = self.ensure_base_pretrain_mlp_retrieval_seqdec()
        work = self.result_root
        # 生成变体配置：将 window_mode 改为 fixed_time（不等距）
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
        self._add_npz(base["retrieval_npz"], fig, sec, "Full Model",
                      f"A1_FULL_{self.scene_code}")
        self._add_npz(ret, fig, sec, "w/o Equal-Distance",
                      f"A1_WO_ED_{self.scene_code}")

    # ------------------------------------------------------------------ #
    # A2 实验：表征增强机制综合消融
    # ------------------------------------------------------------------ #

    def run_a2_representation_ablation(self) -> None:
        """
        A2 表征增强机制综合消融实验。

        两种消融维度：
          w/o Augmentation：关闭数据增强（旋转=0°, 噪声sigma=0）
            → 验证随机增强（旋转、噪声）对对比学习表征质量的提升作用
          w/o Local-Variation Features（MSFE）：关闭局部变异特征
            → 验证 MSFE 特征（捕捉局部信号变化趋势）对定位的贡献

        MSFE 关闭时模型输入通道数需从 5 通道降回 3 通道（Bxyz + Acc）。
        """
        base = self.ensure_base_pretrain_mlp_retrieval_seqdec()
        work = self.result_root

        # 变体1：关闭数据增强（旋转+噪声置零）
        no_aug_cfg, no_aug_path = _variant_config(
            self.config_path,
            "a2_wo_augmentation",
            {"augmentation": {"rotation_max_deg": 0, "noise_sigma": 0}},
            work,
        )
        # 变体2：关闭 MSFE 局部变异特征（输入降维）
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
        self._add_npz(base["retrieval_npz"], fig, sec, "Full Model",
                      f"A2_FULL_{self.scene_code}")
        self._add_npz(no_aug_ret, fig, sec, "w/o Augmentation",
                      f"A2_WO_AUG_{self.scene_code}")
        self._add_npz(no_lvf_ret, fig, sec, "w/o Local-Variation Features",
                      f"A2_WO_LVF_{self.scene_code}")

    # ------------------------------------------------------------------ #
    # continuous 实验：整体连续定位性能对比
    # ------------------------------------------------------------------ #

    def run_continuous(self) -> None:
        """
        continuous 整体连续定位性能对比。

        包含以下方法的测试集定位误差：
          传统方法（无需训练）：
            WKNN（Weighted K-Nearest Neighbors）：指纹匹配定位
            PDR（Pedestrian Dead Reckoning）：基于加速度积分的步数+航向角推算
          MagCLR 系列：
            In-Device: Regression（MLP）/ Top-K FAISS / SeqDec
            LODO: 同上三种方法在每个未见设备上的泛化误差

        此实验综合评估 MagCLR 在连续定位场景下相比传统方法的精度优势。
        """
        devices = _discover_devices(self.cfg)
        base = self.ensure_base_pretrain_mlp_retrieval_seqdec()

        # 基线方法（无需训练，直接在测试集上评估）
        wknn_npz = self._ensure_eval_wknn(self.config_path, self.cfg)
        pdr_npz  = self._ensure_eval_pdr(self.config_path, self.cfg)

        fig = "4.7.3_continuous"
        sec = "4.7.3 整体连续定位性能对比"

        # 传统基线（无论是否发现设备都添加）
        self._add_npz(wknn_npz, fig, sec, "WKNN (Fingerprint)",
                      f"WKNN_{self.scene_code}")
        self._add_npz(pdr_npz,  fig, sec, "PDR (Step-Heading)",
                      f"PDR_{self.scene_code}")

        # MagCLR In-Device 结果
        self._add_npz(base["regression_npz"], fig, sec,
                      "MagCLR + Regression (In-Device)",
                      f"MLP_ID_{self.scene_code}")
        self._add_npz(base["retrieval_npz"], fig, sec,
                      "MagCLR + Top-K Weighted Fusion (In-Device)",
                      f"FAISS_ID_{self.scene_code}")
        self._add_npz(base["seqdec_npz"], fig, sec,
                      "SeqDec (In-Device)",
                      f"SEQ_ID_{self.scene_code}")

        if not devices:
            print("[continuous] no devices found; skipping LODO continuous variants.")
            return

        # LODO 跨设备泛化误差（各方法在未见设备上的表现）
        all_faiss, all_mlp, all_seq = [], [], []
        for device in devices:
            code = device.upper()[:6].replace(" ", "")
            curve_suffix = f"LODO_{code}_{self.scene_code}"
            faiss_errs, mlp_errs, seq_errs = self._run_lodo_fold(
                device, curve_suffix, fig, sec
            )
            all_faiss.extend(faiss_errs)
            all_mlp.extend(mlp_errs)
            all_seq.extend(seq_errs)

        if all_faiss:
            self._add_errors(all_faiss, fig, sec,
                             "MagCLR + Top-K Weighted Fusion (LODO)",
                             f"FAISS_LODO_{self.scene_code}")
        if all_mlp:
            self._add_errors(all_mlp, fig, sec,
                             "MagCLR + Regression (LODO)",
                             f"MLP_LODO_{self.scene_code}")
        if all_seq:
            self._add_errors(all_seq, fig, sec,
                             "SeqDec (LODO)",
                             f"SEQ_LODO_{self.scene_code}")

    # ------------------------------------------------------------------ #
    # transition 实验：状态转移约束消融
    # ------------------------------------------------------------------ #

    def run_transition(self) -> None:
        """
        transition 状态转移约束消融实验。

        SeqDec 的两条核心约束：
          1. 位移一致性约束（Displacement Consistency）：
              当前帧的估计坐标 = 上一帧坐标 + 预测位移
              保证相邻帧之间物理上连续（不会瞬移）
          2. 跳点抑制（Jump Suppression）：
              若预测位移超过 jump_threshold_m（默认2.5m），则抑制该跳变
              防止异常磁干扰导致的大幅偏移

        三种配置对比：
          SeqDec (Full):          两条约束均启用（默认）
          SeqDec (w/o Displacement): 仅启用跳点抑制
          SeqDec (w/o Jump):          仅启用位移一致性约束

        注意：所有配置使用相同的预训练编码器，仅解码策略不同。
        """
        base = self.ensure_base_pretrain_mlp_retrieval_seqdec()
        work = self.result_root

        # 变体1：关闭位移一致性约束（只保留跳点抑制）
        no_disp_cfg, no_disp_path = _variant_config(
            self.config_path,
            "seq_wo_displacement",
            {"seqdec": {"use_displacement": False,
                        "use_jump_suppression": True,
                        "use_confidence": True}},
            work,
        )
        # 变体2：关闭跳点抑制（只保留位移一致性约束）
        no_jump_cfg, no_jump_path = _variant_config(
            self.config_path,
            "seq_wo_jump",
            {"seqdec": {"use_displacement": True,
                        "use_jump_suppression": False,
                        "use_confidence": True}},
            work,
        )

        # 复用 base 中已训练好的预训练编码器，只重新运行 SeqDec 评估（避免重复训练）
        pre = base["pretrain"]
        no_disp_npz = self._ensure_eval_seqdec(
            str(no_disp_path), no_disp_cfg, pre, variant="wo_displacement"
        )
        no_jump_npz = self._ensure_eval_seqdec(
            str(no_jump_path), no_jump_cfg, pre, variant="wo_jump"
        )

        fig = "4.7.5_transition_support"
        sec = "4.7.5 状态转移约束消融实验"
        self._add_npz(base["seqdec_npz"], fig, sec, "SeqDec (Full)",
                      f"SEQ_FULL_{self.scene_code}")
        self._add_npz(no_disp_npz, fig, sec,
                      "SeqDec (w/o Displacement Consistency)",
                      f"SEQ_WO_DISP_{self.scene_code}")
        self._add_npz(no_jump_npz, fig, sec,
                      "SeqDec (w/o Jump Suppression)",
                      f"SEQ_WO_JUMP_{self.scene_code}")

    # ------------------------------------------------------------------ #
    # 统一入口：运行指定实验集合，导出 CSV
    # ------------------------------------------------------------------ #

    def run(self, experiments: Sequence[str]) -> tuple[Path, Path]:
        """
        执行指定的实验集合，统一收集误差数据并导出 CSV。

        参数:
            experiments: 实验名称列表，支持 ["e1","e2","e3","a1","a2","continuous","transition"]
                         传入 ["all"] 则运行全部实验
        返回:
            (csv_path, summary_csv_path): 误差曲线 CSV 和汇总指标 CSV 的路径
        """
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

        # 统一导出误差曲线 CSV 和汇总指标 CSV
        write_error_csv(self.rows, self.csv_path, encoding=self.encoding)
        summary = summarize_rows(self.rows)
        write_summary_csv(summary, self.summary_csv, encoding=self.encoding)
        write_summary_json(summary, self.summary_json)
        print(f"[done] paper error csv: {self.csv_path}")
        print(f"[done] summary csv: {self.summary_csv}")
        return self.csv_path, self.summary_csv


# --------------------------------------------------------------------------
# 顶层入口函数（供 scripts/run_paper_experiments.py 调用）
# --------------------------------------------------------------------------

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
    """
    论文实验一键运行入口。

    便捷封装：只需传入 config_path 和 experiments 列表，
    自动创建 PaperExperimentRunner 并执行 run()。
    """
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
    """
    纯收集模式：不执行任何训练/评估，
    只扫描已存在的 NPZ 结果文件，将其汇总导出为 CSV。
    适用于结果已全部生成，只需重新组织 CSV 格式的场景。
    """
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
