# 论文项目恢复与代码审查准备方案

## 0. 当前判断

你上传的早期项目不是完全不可用。压缩包中能看到如下核心内容：

- `models/convnext1d.py`：已有 ConvNeXt-Lite-1D 雏形。
- `models/magclr.py`：已有 MagCLRNet + Projection Head。
- `train/pretrain_contrastive.py`：已有 InfoNCE 预训练和部分 memory bank 实验代码。
- `train/finetune_regression.py`：已有回归微调代码。
- `eval/faiss_index.py`：已有检索索引代码。
- `data/prepare_data.py`：已有 Z-score、等空间窗口、插值、局部变化特征相关代码。

但它目前还不适合作为代码审查版本，因为第四章 SeqDec 基本缺失，实验入口不完整，部分配置和训练逻辑与论文口径不一致。

## 1. 旧项目主要风险点

### 1.1 论文第四章缺失

旧项目只有 FAISS 加权检索，没有完整的 SeqDec：

- 没有候选得分尖锐度；
- 没有候选空间集中度；
- 没有观测置信度调节发射概率；
- 没有位移一致性转移项；
- 没有最大跳变硬筛除；
- 没有 Viterbi 路径回溯；
- 没有跳变比例指标。

这会导致审查时“第四章方法是否真的实现”的风险很高。

### 1.2 配置问题

旧 `configs/config.yaml` 中存在明显拼写问题：

```yaml
label_norm: ture
```

应改为：

```yaml
label_norm: true
```

旧配置还使用 `window_size: 256`，而论文实验设置写的是统一重采样为 128 个采样点。因此恢复论文结果时建议以 `128` 作为主实验，`256` 只能作为额外消融或旧版记录。

### 1.3 预训练与微调逻辑不稳定

旧 `main.py` 中 `finetune_main` 最后固定传入：

```python
train_mode="scratch_full"
```

这会让“Pretrain + MLP”的实验变成从头训练，无法真实验证对比学习预训练有效性。应至少支持三种模式：

1. `scratch_full`：从头监督训练，对应 Scratch + MLP；
2. `pretrained_linear_probe`：冻结编码器，只训练回归头；
3. `pretrained_finetune`：加载预训练，再微调高层与回归头，对应 Pretrain + MLP。

### 1.4 检索依赖 FAISS，不利于审查复现

审查环境不一定安装 FAISS。建议代码中保留 FAISS 可选加速，但必须提供 NumPy/Sklearn fallback。否则老师或审查人员一换电脑就跑不通。

### 1.5 实验结果没有体系化保存

旧项目主要打印 mean L2 / median L2，缺少论文中统一使用的：

- 平均定位误差；
- 中位定位误差；
- P90 误差；
- 跳变比例；
- 同设备与 LODO 分组；
- 按轨迹保存预测结果；
- 按实验保存 config + metrics + npz。

代码审查时，不仅要能跑，还要能追溯每个表格怎么来的。

## 2. 推荐恢复后的项目结构

```text
MagCLR_SeqDec_recovery/
  configs/
    base.yaml
  src/magloc/
    data/
      io.py
      preprocessing.py
      augment.py
      datasets.py
    models/
      convnext1d.py
      magclr.py
    train/
      losses.py
    eval/
      metrics.py
      retrieval.py
      seqdec.py
    experiments/
      common.py
      pretrain.py
      finetune.py
      evaluate.py
  scripts/
    smoke_test.py
    train_pretrain.py
    train_regression.py
    evaluate_regression.py
    evaluate_retrieval.py
    evaluate_seqdec.py
  docs/
    code_audit_recovery_plan.md
  README_RESTORE.md
  run_all.sh
```

## 3. 复现实验总路线

### 阶段 A：先保证数据和标签可靠

必须先跑数据统计，而不是直接训练。检查：

- 每个 `.npy` 是否为 `(T,5)`；
- 坐标范围是否与场景尺寸相符；
- 训练/验证/测试是否按轨迹划分，而不是窗口随机划分；
- 同一条轨迹切出来的窗口不能同时出现在训练和测试中；
- 每个设备在 train/val/test 中的数量；
- LODO 时测试设备绝不能出现在训练集中。

### 阶段 B：先跑最小可用链路

先不追求论文结果，只验证链路：

```bash
export PYTHONPATH=$PWD/src
python scripts/smoke_test.py
python scripts/train_pretrain.py --config configs/base.yaml
python scripts/evaluate_retrieval.py --config configs/base.yaml --encoder-ckpt experiments/scenario_1/pretrain/pretrain_best.pth
python scripts/evaluate_seqdec.py --config configs/base.yaml --encoder-ckpt experiments/scenario_1/pretrain/pretrain_best.pth
```

若这一步跑不通，不要进入大规模实验。

### 阶段 C：恢复第三章主结果

建议按以下实验顺序恢复：

1. E1：Backbone 监督对比  
   RNN / LSTM / CNN+TCN / ConvNeXt-Lite-1D。  
   目的：证明 ConvNeXt-Lite-1D 合理。

2. E2：有无预训练  
   Scratch + MLP vs Pretrain + MLP。  
   目的：证明 InfoNCE 表征学习有效。

3. K 敏感性  
   K = 1,2,3,4,5。  
   目的：确定检索默认 K=3。

4. E3：下游定位与跨设备泛化  
   In-Device / LODO 下比较 Pretrain + MLP 和 Pretrain + Weighted Retrieval。  
   目的：证明检索式定位更适合鲁棒嵌入。

5. A1/A2 消融  
   - w/o Equal-Distance；
   - w/o Augmentation；
   - w/o Local-Variation Features。

### 阶段 D：恢复第四章主结果

第四章不重新训练编码器，只复用第三章的 encoder + train 指纹库。实验顺序：

1. 整体性能：
   - MagCLR + Regression；
   - MagCLR + Top-K Weighted Fusion；
   - SeqDec Full。

2. 观测置信度机制：
   - SeqDec w/o Confidence；
   - SeqDec Full；
   - 按置信度三分位分组统计。

3. 状态转移消融：
   - w/o Displacement Consistency；
   - w/o Jump Suppression；
   - Full。

4. 参数敏感性：
   - confidence_alpha = 0.25, 0.50, 0.75, 1.00；
   - max_jump_m = 1.5, 2.0, 2.5, 3.0；
   - displacement_sigma_m = 0.5, 0.8, 1.2。

## 4. 建议参数范围

### 主实验默认参数

```yaml
preprocess:
  window_size: 128
  window_length_m: 2.0
  stride_m: 1.0
  zscore: true
  msfe: true
  diff_k: 1

pretrain:
  batch_size: 128
  epochs: 100
  lr: 0.001
  temperature: 0.1

finetune:
  epochs: 80
  lr: 0.001
  huber_beta: 1.0
  label_norm: true

retrieval:
  k: 3
  tau: 0.30

seqdec:
  k: 3
  tau: 0.30
  spatial_sigma_m: 1.20
  confidence_alpha: 0.75
  expected_step_m: 1.0
  displacement_sigma_m: 0.80
  max_jump_m: 2.50
  beta: 0.45
```

### 可调优先级

1. 优先调 `window_length_m` 与 `stride_m`，因为它们直接影响标签语义与相邻窗口位移。
2. 其次调 `K` 与检索 `tau`，因为它们影响候选质量。
3. 再调 SeqDec 参数，尤其是 `max_jump_m` 和 `displacement_sigma_m`。
4. 最后调模型深度、学习率、batch size。

## 5. 代码审查前自检清单

- [ ] 运行 `python scripts/smoke_test.py` 通过。
- [ ] 每个实验目录保存 `config.yaml`。
- [ ] 每个 checkpoint 保存 epoch、model_state_dict、config、metrics。
- [ ] 每个评估保存 `metrics.json` 和 `preds.npz`。
- [ ] 能说明第三章 encoder 如何输出 embedding。
- [ ] 能说明检索库由 train split 构建，test split 只作为 query。
- [ ] 能说明 SeqDec 的输入是每个时间步 Top-K 候选，而不是重新预测坐标。
- [ ] 能说明 `jump_ratio` 的阈值和转移约束中的 `max_jump_m` 保持一致。
- [ ] 能跑通一个最小场景，不依赖 FAISS。
- [ ] LODO 实验能打印被留出的设备名。

## 6. 最应该优先修的旧代码点

1. 修 `label_norm: ture`。
2. 把 `window_size` 主实验改为 128；保留 256 作为旧版对照。
3. `finetune_main` 不要固定 `scratch_full`。
4. 增加 `eval/seqdec.py`。
5. 增加 `metrics.py`，统一 mean/median/P90/jump_ratio。
6. 让检索不强依赖 FAISS。
7. 增加实验脚本，而不是把所有逻辑堆在 `main.py`。
8. 每次实验输出配置、模型、指标、预测结果。

## 7. 结果恢复策略

如果时间紧，建议优先恢复能支撑论文主线的 6 组结果：

1. ConvNeXt-Lite-1D backbone 监督结果；
2. Scratch + MLP vs Pretrain + MLP；
3. Pretrain + MLP vs Pretrain + Weighted Retrieval；
4. K=1~5 敏感性；
5. MagCLR + Weighted Retrieval vs SeqDec Full；
6. SeqDec 三个消融：w/o Confidence、w/o Displacement、w/o Jump。

这 6 组能覆盖论文第 3 章和第 4 章的主贡献，代码审查时也最容易讲清楚。
