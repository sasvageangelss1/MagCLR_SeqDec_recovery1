# MagCLR-Net + SeqDec 恢复项目说明

这个目录是按论文第 3 章 MagCLR-Net 与第 4 章 SeqDec 重新整理出的“代码审查友好版”项目骨架。它不替换你的原始数据，只提供可覆盖到旧项目中的核心代码、配置和复现实验入口。

## 1. 推荐目录结构

```text
configs/base.yaml                  # 统一参数入口
src/magloc/data/                   # 读取 npy、等空间窗口、标准化、局部变化特征、增强
src/magloc/models/                 # ConvNeXt-Lite-1D、MagCLRNet、RegressionHead
src/magloc/train/                  # InfoNCE 损失等训练组件
src/magloc/eval/                   # 指标、检索、SeqDec/Viterbi
src/magloc/experiments/            # pretrain / finetune / evaluate 编排
scripts/                           # 命令行入口
run_all.sh                         # 单场景复现流水线
```

## 2. 数据格式

默认读取：

```text
data/npy_output/train/*.npy
data/npy_output/val/*.npy
data/npy_output/test/*.npy
```

每个 `.npy` 应为 `(T,5)`，列顺序为 `[mx, my, mz, x, y]`。如果你使用旧项目解出的 `data/npy_output`，可以直接把本目录覆盖到旧项目根目录，然后运行。

## 3. 快速自检

```bash
set PYTHONPATH=$PWD/src
python scripts/smoke_test.py
```

## 4. 单场景复现顺序

```bash
set PYTHONPATH=$PWD/src
python scripts/train_pretrain.py --config configs/base.yaml
python scripts/train_regression.py --config configs/base.yaml --pretrained-ckpt experiments/scenario_1/pretrain/pretrain_best.pth
python scripts/evaluate_regression.py --config configs/base.yaml --ckpt experiments/scenario_1/finetune/regression_best.pth
python scripts/evaluate_retrieval.py --config configs/base.yaml --encoder-ckpt experiments/scenario_1/pretrain/pretrain_best.pth
python scripts/evaluate_seqdec.py --config configs/base.yaml --encoder-ckpt experiments/scenario_1/pretrain/pretrain_best.pth
```

## 5. 与论文口径一致的核心默认值

- 输入：Z-score + 等空间窗口 + 3 轴原始磁场 + 3 轴 k 步差分 + 1 维梯度能量，共 7 通道。
- ConvNeXt-Lite-1D：Stem stride=4，depths=[2,2,4,2]，dims=[64,128,256,256]，embedding=256。
- 对比学习：InfoNCE，temperature=0.1，batch size=128，epochs=100。
- 检索定位：Top-K=3，cosine similarity，softmax 加权。
- SeqDec：Top-K=3，tau=0.30，spatial_sigma=1.20m，confidence_alpha=0.75，displacement_sigma=0.80m，max_jump=2.50m，beta=0.45。

## 6. 必须保留的材料

每次实验都保存：配置 YAML、checkpoint、metrics JSON、候选 npz、预测轨迹 npz、运行命令和随机种子。。

## 7. Backbone 对比实验 E1

该实验对应论文第 3.7.2 节，不使用对比学习预训练，四种网络共享同一输入构造、监督回归头、训练策略和评价指标。

```bash
export PYTHONPATH=$PWD/src
python scripts/run_backbone_compare.py --config configs/base.yaml
```

快速检查某一个 backbone：

```bash
python scripts/run_backbone_compare.py --config configs/base.yaml --backbones rnn --epochs 2
```

输出目录默认为：

```text
experiments/<scene.name>/backbone_compare/
  data_meta.json
  used_config.yaml
  summary.md
  summary.csv
  summary.json
  rnn/best.pth, val_metrics.json, test_metrics.json, test_preds.npz
  lstm/...
  cnn_tcn/...
  convnext_lite_1d/...
```

`summary.md` 可以直接改写成论文表 3.3 的数据表。若要复现场景 2，只需要复制一份配置，把 `scene.name` 改成 `scenario_2`，并把 `scene.scene_filter` 改成你的第二场景文件名关键词，例如 `文管`。

## 论文实验统一 CSV 入口

新增脚本：

```bash
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments all
```

它会按论文实验顺序检查已有 checkpoint 和 `.npz` 结果：

- 如果已有 `pretrain_best.pth`、`regression_best.pth` 或评估 `.npz`，默认直接复用；
- 如果缺失，则自动运行对应训练/评估；
- 如果加 `--force`，则强制重新运行；
- 如果加 `--collect-only`，则只从已有 `.npz` 结果导出 CSV，不进行训练和评估。

默认输出：

```text
experiments/<scene_name>/paper_csv/paper_error_curves.csv
experiments/<scene_name>/paper_csv/paper_summary_metrics.csv
experiments/<scene_name>/paper_csv/paper_summary_metrics.json
```

`paper_error_curves.csv` 的列与上传的 CSV 保持一致：

```text
figure_id, chapter_section, scene, method, curve_key, sample_id, error_m
```

### 分实验运行命令

```bash
# E1：Backbone 对比
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments e1

# E2：Scratch + MLP vs Pretrain + MLP
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments e2

# E3：下游定位方式综合实验，In-Device 部分自动生成；LODO 需额外传入 LODO 配置
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments e3

# A1：等空间窗口消融
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments a1

# A2：增强机制与局部变化特征消融
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments a2

# 第4章整体连续定位对比
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments continuous

# 第4章状态转移约束消融
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments transition
```

### 只汇总已有结果，不重复训练

```bash
python scripts/run_paper_experiments.py \
  --config configs/base.yaml \
  --experiments all \
  --collect-only
```

也可使用别名脚本：

```bash
python scripts/export_existing_results_to_csv.py --config configs/base.yaml
```

### 输出到指定 CSV

```bash
python scripts/run_paper_experiments.py \
  --config configs/base.yaml \
  --experiments all \
  --csv experiments/paper_results/all_error_curves.csv \
  --summary-csv experiments/paper_results/all_summary_metrics.csv
```

### 场景2运行方式

复制 `configs/base.yaml` 为 `configs/scenario_2.yaml`，修改：

```yaml
scene:
  name: scenario_2
  scene_filter: "文管"
```

然后运行：

```bash
python scripts/run_paper_experiments.py --config configs/scenario_2.yaml --experiments all --scene-label 场景2 --scene-code S2
```

### LODO 跨设备运行方式

先为每个留一设备准备独立配置，例如：

```text
configs/lodo_huawei_p70.yaml
configs/lodo_xiaomi14.yaml
...
```

每个配置的 `paths.data_root` 指向对应的 LODO 数据目录。然后运行：

```bash
python scripts/run_paper_experiments.py \
  --config configs/base.yaml \
  --experiments e3 \
  --lodo-configs configs/lodo_huawei_p70.yaml,configs/lodo_xiaomi14.yaml
```

脚本会把多个 LODO 配置的误差合并为同一条 `LODO Pretrain + FAISS` / `LODO Pretrain + MLP` 曲线。

运行记录
Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized. OpenMP 运行时冲突问题
set KMP_DUPLICATE_LIB_OK=TRUE