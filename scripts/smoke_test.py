from __future__ import annotations
import numpy as np
import torch
torch.set_num_threads(1)
from magloc.data.preprocessing import build_equal_distance_windows, local_variation_features
from magloc.models import MagCLRNet
from magloc.eval.retrieval import NumpyRetriever, softmax_weighted_position
from magloc.eval.seqdec import SeqDecConfig, viterbi_decode
from magloc.eval.metrics import localization_metrics

np.random.seed(0)
T = 200
pos = np.stack([np.linspace(0, 10, T), np.zeros(T)], axis=1).astype(np.float32)
mag = np.stack([np.sin(pos[:, 0]), np.cos(pos[:, 0]), 0.1 * pos[:, 0]], axis=1).astype(np.float32)
arr = np.concatenate([mag, pos], axis=1)
windows, labels = build_equal_distance_windows(arr, window_size=128, window_length_m=2.0, stride_m=1.0)
assert windows.shape[1:] == (128, 3)
feat = np.stack([local_variation_features(w).T for w in windows])
model = MagCLRNet(in_channels=7, embed_dim=256, proj_dim=128)
with torch.no_grad():
    h, z = model(torch.from_numpy(feat).float())
assert h.shape[1] == 256 and z.shape[1] == 128
retriever = NumpyRetriever('cosine').fit(h.numpy(), labels)
res = retriever.query(h.numpy(), k=3)
pred = softmax_weighted_position(res.scores, res.positions)
seq = viterbi_decode(res.scores, res.positions, SeqDecConfig(expected_step_m=1.0))
metrics = localization_metrics(seq['pred'], labels, jump_threshold_m=2.5)
print('smoke ok', windows.shape, h.shape, metrics)
