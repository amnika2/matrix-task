"""实验三、四及随机投影扩展；验证集选维度，测试集独立评价。"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (DIMS, benchmark, check_same_encoder, configure, environment,
                    load_features, load_svd, normalize, project, read_json, recalls,
                    save_npz, sha256, write_csv, write_json)


def random_basis(seed, dimension=512):
    # 高斯矩阵 QR 得到随机正交基；同一 seed 的各维度使用嵌套子空间。
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, dimension)))
    q *= np.where(np.diag(r) < 0, -1.0, 1.0)
    return q


@torch.inference_mode()
def efficiency(images, texts, wi, wt, device, repeats):
    i = torch.as_tensor(images, device=device)
    t = torch.as_tensor(texts, device=device)
    if wi is None:
        zi, zt = i, t
        projection_ms, projection_iqr = 0.0, 0.0
        matrix_count = 0
    else:
        matrix_count = 1 if wt is None else 2
        a = torch.tensor(wi, dtype=torch.float32, device=device)
        b = a if wt is None else torch.tensor(wt, dtype=torch.float32, device=device)

        def mapping():
            return F.normalize(i @ a, dim=1), F.normalize(t @ b, dim=1)

        projection_ms, projection_iqr = benchmark(mapping, device, repeats)
        zi, zt = mapping()
    # 完整矩阵乘法；不包含排序、投影、编码、数据加载和 CPU/GPU 传输。
    t2i_ms, t2i_iqr = benchmark(lambda: zt @ zi.T, device, repeats)
    i2t_ms, i2t_iqr = benchmark(lambda: zi @ zt.T, device, repeats)
    k = zi.shape[1]
    image_bytes, text_bytes = len(i) * k * 4, len(t) * k * 4
    raw = (len(i) + len(t)) * 512 * 4
    projection_bytes = matrix_count * 512 * k * 4
    return {"image_feature_bytes": image_bytes, "text_feature_bytes": text_bytes,
            "both_feature_bytes": image_bytes + text_bytes,
            "projection_bytes_float32": projection_bytes,
            "both_plus_projection_bytes": image_bytes + text_bytes + projection_bytes,
            "feature_saving_percent": 100.0 * (1 - k / 512),
            "saving_including_projection_percent": 100.0 * (1 - (image_bytes + text_bytes + projection_bytes) / raw),
            "projection_both_ms_median": projection_ms, "projection_both_ms_iqr": projection_iqr,
            "T2I_matmul_ms_median": t2i_ms, "T2I_matmul_ms_iqr": t2i_iqr,
            "I2T_matmul_ms_median": i2t_ms, "I2T_matmul_ms_iqr": i2t_iqr}


def full_dimension_check(root, split, images, texts, owners, v, device, baseline):
    # 分别检查 float64 数学等价性与实际 float32 实现。
    i64, t64 = normalize(images.astype(np.float64)), normalize(texts.astype(np.float64))
    ip64, tp64 = normalize(i64 @ v), normalize(t64 @ v)
    ip32, tp32 = project(images, texts, v.astype(np.float32))
    max64, max32 = 0.0, 0.0
    for begin in range(0, len(texts), 256):
        ref64 = t64[begin:begin + 256] @ i64.T
        max64 = max(max64, float(np.max(np.abs(tp64[begin:begin + 256] @ ip64.T - ref64))))
        ref32 = texts[begin:begin + 256] @ images.T
        max32 = max(max32, float(np.max(np.abs(tp32[begin:begin + 256] @ ip32.T - ref32))))
    metrics = recalls(ip32, tp32, owners, device)
    delta = {key: metrics[key] - baseline[key] for key in metrics}
    write_json(root / "results" / f"full_dimension_check_{split}.json",
               {"max_cosine_abs_error_float64": max64, "max_cosine_abs_error_float32": max32,
                "recall_delta_percentage_points": delta, "raw": baseline, "svd512": metrics,
                "passed_similarity_tolerance": max64 < 1e-10 and max32 < 2e-5,
                "note": "微小 Recall 差别可能来自近似同分结果的浮点排序；明显差别需要排查。"})
    if max64 >= 1e-10 or max32 >= 2e-5:
        raise AssertionError("512 维完整投影相似度校验失败。")
    if max(abs(value) for value in delta.values()) > 0.2:
        raise AssertionError("512 维投影 Recall 差异超过 0.2 个百分点，请检查结果。")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--split", choices=["validation", "test"], required=True)
    p.add_argument("--device", default="cpu", help="各次对照保持相同设备")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--random-seeds", nargs="+", type=int, help="可选：如 11 22 33")
    p.add_argument("--max-drop", type=float, default=1.0, help="验证集相对原始特征允许的 mean Recall 下降，百分点")
    args = p.parse_args()
    if args.repeats < 1 or args.max_drop < 0:
        p.error("repeats 必须为正；max-drop 不能为负。")
    if args.random_seeds and len(set(args.random_seeds)) != len(args.random_seeds):
        p.error("随机种子不能重复。")
    configure(threads=args.threads)
    device = torch.device(args.device)
    root = Path(args.root)
    selection_path = root / "results" / "selection.json"
    if args.split == "test":
        selection = read_json(selection_path)
        if selection["svd_sha256"] != sha256(root / "models" / "svd.npz"):
            raise ValueError("SVD 已改变；请重新运行 validation。")
        if selection["validation_features_sha256"] != sha256(root / "features" / "validation.npz"):
            raise ValueError("验证特征已改变；请重新运行 validation。")
    images, texts, owners, meta = load_features(root, args.split)
    train_i, train_t, _, train_meta = load_features(root, "train")
    check_same_encoder(train_meta, meta)
    v, _ = load_svd(root)
    original = recalls(images, texts, owners, device)
    rows = [{"method": "original", "dimension": 512, "seed": "", **original}]
    timings = [{"method": "original", "dimension": 512, "seed": "",
                **efficiency(images, texts, None, None, device, args.repeats)}]
    full = full_dimension_check(root, args.split, images, texts, owners, v, device, original)
    for k in DIMS:
        wi = v[:, :k].astype(np.float32)
        zi, zt = project(images, texts, wi)
        score = full if k == 512 else recalls(zi, zt, owners, device)
        rows.append({"method": "svd", "dimension": k, "seed": "", **score})
        timings.append({"method": "svd", "dimension": k, "seed": "",
                        **efficiency(images, texts, wi, None, device, args.repeats)})
        print(f"{args.split}: svd k={k}, mean Recall={score['mean_recall']:.3f}", flush=True)
    random_reconstruction = []
    if args.random_seeds:
        x = np.concatenate([train_i, train_t]).astype(np.float64)
        norm = np.linalg.norm(x, "fro")
        for seed in args.random_seeds:
            q = random_basis(seed)
            for k in [128, 256]:
                wi = q[:, :k].astype(np.float32)
                # 重建误差使用 float64 正交基；检索和效率使用 float32。
                qk = q[:, :k]
                error = float(np.linalg.norm(x - (x @ qk) @ qk.T, "fro") / norm)
                zi, zt = project(images, texts, wi)
                rows.append({"method": "random", "dimension": k, "seed": seed,
                             **recalls(zi, zt, owners, device)})
                timings.append({"method": "random", "dimension": k, "seed": seed,
                                **efficiency(images, texts, wi, None, device, args.repeats)})
                random_reconstruction.append({"method": "random", "dimension": k, "seed": seed,
                                              "relative_fro_measured": error})
                save_npz(root / "models" / f"random_seed{seed}_k{k}.npz", w=wi)
        write_csv(root / "results" / f"random_reconstruction_{args.split}.csv", random_reconstruction)
        summary = []
        for k in [128, 256]:
            subset = [r for r in rows if r["method"] == "random" and r["dimension"] == k]
            row = {"method": "random", "dimension": k, "num_seeds": len(subset)}
            for metric in original:
                values = [r[metric] for r in subset]
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else ""
            errors = [r["relative_fro_measured"] for r in random_reconstruction if r["dimension"] == k]
            row["relative_fro_mean"] = float(np.mean(errors))
            row["relative_fro_std"] = float(np.std(errors, ddof=1)) if len(errors) > 1 else ""
            summary.append(row)
        write_csv(root / "results" / f"random_summary_{args.split}.csv", summary)
    write_csv(root / "results" / f"retrieval_{args.split}.csv", rows)
    write_csv(root / "results" / f"efficiency_{args.split}.csv", timings)
    write_json(root / "results" / f"evaluation_{args.split}_meta.json",
               {**environment(device), "repeats": args.repeats, "warmups": 5,
                "random_seeds": args.random_seeds, "split": args.split,
                "timing_scope": "in-memory float32 matmul; synchronized; excludes ranking/transfer/encoding",
                "feature_sha256": sha256(root / "features" / f"{args.split}.npz"),
                "svd_sha256": sha256(root / "models" / "svd.npz")})
    if args.split == "validation":
        eligible = [r for r in rows if r["method"] == "svd" and
                    r["mean_recall"] >= original["mean_recall"] - args.max_drop]
        # 无候选满足时回退原始特征，避免强行选择性能下降的压缩方案。
        chosen = min(eligible, key=lambda r: r["dimension"]) if eligible else rows[0]
        write_json(selection_path,
                   {"rule": "smallest SVD dimension within allowed mean Recall drop; else original",
                    "max_drop_percentage_points": args.max_drop, "method": chosen["method"],
                    "dimension": chosen["dimension"], "validation_mean_recall": chosen["mean_recall"],
                    "svd_sha256": sha256(root / "models" / "svd.npz"),
                    "validation_features_sha256": sha256(root / "features" / "validation.npz")})
        print(f"验证集固定配置：{chosen['method']} k={chosen['dimension']}")
    else:
        chosen = next(r for r in rows if r["method"] == selection["method"] and r["dimension"] == selection["dimension"])
        write_json(root / "results" / "selected_test.json", {"selected_on_validation": selection, "test": chosen})


if __name__ == "__main__":
    main()
