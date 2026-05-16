# 论文实验复现命令与统一 CSV 输出

## 0. 基础准备

```bash
export PYTHONPATH=$PWD/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python scripts/smoke_test.py
```

Windows PowerShell：

```powershell
$env:PYTHONPATH="$PWD/src"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
python scripts/smoke_test.py
```

## 1. 一键运行全部论文实验并输出统一 CSV

```bash
python scripts/run_paper_experiments.py \
  --config configs/base.yaml \
  --experiments all
```

默认输出：

```text
experiments/scenario_1/paper_csv/paper_error_curves.csv
experiments/scenario_1/paper_csv/paper_summary_metrics.csv
experiments/scenario_1/paper_csv/paper_summary_metrics.json
```

其中 `paper_error_curves.csv` 与上传 CSV 格式一致：

```text
figure_id,chapter_section,scene,method,curve_key,sample_id,error_m
```

## 2. 只汇总已有结果，不重复训练

```bash
python scripts/run_paper_experiments.py \
  --config configs/base.yaml \
  --experiments all \
  --collect-only
```

或：

```bash
python scripts/export_existing_results_to_csv.py --config configs/base.yaml
```

## 3. 分实验复现命令

### E1：3.7.2 Backbone 监督定位对比

```bash
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments e1
```

包含曲线/方法：

```text
RNN / LSTM / CNN+TCN / ConvNeXt-Lite-1D
```

### E2：3.7.3 对比学习预训练有效性验证

```bash
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments e2
```

包含曲线/方法：

```text
Scratch + MLP
Pretrain + MLP
```

### E3：3.7.5 下游定位方式与跨设备泛化综合实验

In-Device：

```bash
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments e3
```

如果需要 LODO，请准备多个 LODO 配置后运行：

```bash
python scripts/run_paper_experiments.py \
  --config configs/base.yaml \
  --experiments e3 \
  --lodo-configs configs/lodo_huawei_p70.yaml,configs/lodo_xiaomi14.yaml
```

包含曲线/方法：

```text
In-Device Pretrain + FAISS
In-Device Pretrain + MLP
LODO Pretrain + FAISS  # 需传入 --lodo-configs
LODO Pretrain + MLP    # 需传入 --lodo-configs
```

### A1：3.7.6 等空间窗口构造消融

```bash
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments a1
```

包含曲线/方法：

```text
Full Model
w/o Equal-Distance
```

### A2：3.7.6 表征增强机制综合消融

```bash
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments a2
```

包含曲线/方法：

```text
Full Model
w/o Augmentation
w/o Local-Variation Features
```

### 4.7.3 整体连续定位性能对比

```bash
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments continuous
```

包含曲线/方法：

```text
MagCLR + Regression
MagCLR + Top-K Weighted Fusion
SeqDec (Full)
```

### 4.7.5 状态转移约束消融实验

```bash
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments transition
```

包含曲线/方法：

```text
SeqDec (Full)
SeqDec (w/o Displacement Consistency)
SeqDec (w/o Jump Suppression)
```

## 4. 场景2复现

复制配置：

```bash
cp configs/base.yaml configs/scenario_2.yaml
```

修改：

```yaml
scene:
  name: scenario_2
  scene_filter: "文管"
```

运行：

```bash
python scripts/run_paper_experiments.py \
  --config configs/scenario_2.yaml \
  --experiments all \
  --scene-label 场景2 \
  --scene-code S2
```

输出：

```text
experiments/scenario_2/paper_csv/paper_error_curves.csv
experiments/scenario_2/paper_csv/paper_summary_metrics.csv
```

## 5. 强制重跑或指定输出路径

强制重跑：

```bash
python scripts/run_paper_experiments.py --config configs/base.yaml --experiments all --force
```

指定输出 CSV：

```bash
python scripts/run_paper_experiments.py \
  --config configs/base.yaml \
  --experiments all \
  --csv experiments/paper_results/all_error_curves.csv \
  --summary-csv experiments/paper_results/all_summary_metrics.csv
```

## 6. 结果文件对应关系

```text
paper_error_curves.csv   # 逐样本误差，供 CDF / 箱线图使用
paper_summary_metrics.csv # 自动聚合平均误差、中位误差、P90、RMSE、最大误差
```

`paper_error_curves.csv` 每一行是一个样本误差；`figure_id + method + curve_key` 可以唯一定位其属于论文哪个实验、哪条曲线。
