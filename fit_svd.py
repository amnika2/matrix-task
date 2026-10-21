"""实验一、二：仅在训练图文联合矩阵上做不中心化的 float64 SVD。"""
import argparse
import time
from pathlib import Path

import numpy as np

from common import DIMS, load_features, save_npz, sha256, write_csv, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--spectral", action="store_true", help="额外实测残差谱范数，增加计算量")
    args = p.parse_args()
    root = Path(args.root)
    images, texts, _, _ = load_features(root, "train")
    x = np.concatenate([images, texts]).astype(np.float64)
    start = time.perf_counter()
    _, s, vh = np.linalg.svd(x, full_matrices=False)
    fit_seconds = time.perf_counter() - start
    v = vh.T.copy()
    save_npz(root / "models" / "svd.npz", v=v, s=s,
             train_sha256=np.array(sha256(root / "features" / "train.npz")))
    write_json(root / "models" / "svd_meta.json",
               {"shape": list(x.shape), "centered": False, "dtype": "float64",
                "svd_fit_seconds": fit_seconds, "numpy": np.__version__,
                "orthogonality_max_abs": float(np.max(np.abs(v.T @ v - np.eye(512)))),
                "rank_numeric": int(np.sum(s > s[0] * max(x.shape) * np.finfo(float).eps))})
    total_energy = np.sum(s ** 2)
    write_csv(root / "results" / "singular_values.csv",
              [{"index": j + 1, "singular_value": float(s[j]),
                "cumulative_energy": float(np.sum(s[:j + 1] ** 2) / total_energy)} for j in range(len(s))])
    norm = np.linalg.norm(x, "fro")
    rows = []
    for k in DIMS:
        vk = v[:, :k]
        residual = x - (x @ vk) @ vk.T
        actual = float(np.linalg.norm(residual, "fro"))
        theory = float(np.sqrt(np.sum(s[k:] ** 2)))
        if abs(actual - theory) > 1e-8 * norm:
            raise AssertionError(f"k={k} 理论与实测 Frobenius 误差不一致。")
        row = {"dimension": k, "energy_retained": float(np.sum(s[:k] ** 2) / total_energy),
               "fro_theory": theory, "fro_measured": actual,
               "relative_fro_theory": theory / norm, "relative_fro_measured": actual / norm,
               "fro_absolute_gap": abs(actual - theory),
               "spectral_theory": float(s[k]) if k < len(s) else 0.0,
               "spectral_measured": ""}
        if args.spectral:
            row["spectral_measured"] = float(np.linalg.svd(residual, compute_uv=False)[0])
        rows.append(row)
        print(f"k={k}: energy={row['energy_retained']:.6f}, relative_error={actual / norm:.6f}", flush=True)
    write_csv(root / "results" / "reconstruction.csv", rows)
    print(f"SVD 拟合完成：{fit_seconds:.3f} 秒。图表由 plot_results.py 生成。")


if __name__ == "__main__":
    main()
