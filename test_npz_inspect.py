import numpy as np

files = [
    "C:/Users/hp/Desktop/zz/proj/MagCLR_SeqDec_recovery1/experiments/scenario_1/eval_retrieval/test_retrieval_candidates.npz",
    "C:/Users/hp/Desktop/zz/proj/MagCLR_SeqDec_recovery1/experiments/scenario_1/eval_regression/test_regression_preds.npz",
    "C:/Users/hp/Desktop/zz/proj/MagCLR_SeqDec_recovery1/experiments/scenario_1/eval_seqdec/test_seqdec_preds.npz",
]

for f in files:
    data = np.load(f)
    name = f.split("/")[-1]
    print(f"=== {name} ===")
    for k in data.files:
        arr = data[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.ndim == 1 and len(arr) <= 10:
            print(f"    values: {arr}")
        elif arr.ndim == 2 and arr.shape[1] == 2 and len(arr) <= 5:
            print(f"    first few:\n{arr[:5]}")
    print()
