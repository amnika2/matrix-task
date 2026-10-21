"""可选实验六：在冻结、已缓存的 CLIP 特征上训练两个线性矩阵。"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (check_same_encoder, configure, environment, load_features,
                    load_svd, project, recalls, save_npz, sha256, sync,
                    write_csv, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dimension", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--test-only", action="store_true", help="只评价已由验证集选定的最佳 checkpoint")
    args = p.parse_args()
    if not 1 <= args.dimension <= 512 or args.epochs < 1 or args.batch_size < 2 or args.temperature <= 0 or args.lr <= 0:
        p.error("检查 dimension / epochs / batch-size / temperature / lr 的范围。")
    configure(args.seed, args.threads)
    root = Path(args.root)
    device = torch.device(args.device)
    tag = f"linear_k{args.dimension}_seed{args.seed}"
    checkpoint = root / "models" / f"{tag}.npz"
    train_path = root / "features" / "train.npz"
    val_path = root / "features" / "validation.npz"
    train_i, train_t, _, train_meta = load_features(root, "train")
    vi, vt, vo, val_meta = load_features(root, "validation")
    check_same_encoder(train_meta, val_meta)
    v, _ = load_svd(root)
    if not args.test_only:
        if checkpoint.exists():
            raise FileExistsError(f"已有 {checkpoint}。用 --test-only 评价；重新训练请选新 seed 或新 root。")
        initial = v[:, :args.dimension].astype(np.float32)
        wi = torch.nn.Parameter(torch.tensor(initial, device=device))
        wt = torch.nn.Parameter(torch.tensor(initial, device=device))
        optimizer = torch.optim.AdamW([wi, wt], lr=args.lr, weight_decay=1e-4)
        x = torch.tensor(train_i, device=device)
        y = torch.tensor(train_t, device=device)
        init_i, init_t = project(vi, vt, initial)
        best = recalls(init_i, init_t, vo, device)["mean_recall"]
        history = [{"epoch": 0, "loss": "", "validation_mean_recall": best, "train_seconds": 0.0}]
        best_epoch, best_wi, best_wt = 0, initial.copy(), initial.copy()
        total_training_seconds = 0.0
        for epoch in range(1, args.epochs + 1):
            sync(device)
            start = time.perf_counter()
            perm = torch.randperm(len(x), device=device)
            loss_sum, count = 0.0, 0
            for begin in range(0, len(x), args.batch_size):
                ids = perm[begin:begin + args.batch_size]
                if len(ids) < 2:
                    continue
                # 每张图像仅有一个训练描述，batch 内没有同图像多正例冲突。
                zi, zt = F.normalize(x[ids] @ wi, dim=1), F.normalize(y[ids] @ wt, dim=1)
                logits = zi @ zt.T / args.temperature
                target = torch.arange(len(ids), device=device)
                loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(ids)
                count += len(ids)
            sync(device)
            elapsed = time.perf_counter() - start
            total_training_seconds += elapsed
            a, b = wi.detach().cpu().numpy().copy(), wt.detach().cpu().numpy().copy()
            zi, zt = project(vi, vt, a, b)
            score = recalls(zi, zt, vo, device)["mean_recall"]
            history.append({"epoch": epoch, "loss": loss_sum / count,
                            "validation_mean_recall": score, "train_seconds": elapsed})
            if score > best:
                best, best_epoch, best_wi, best_wt = score, epoch, a.copy(), b.copy()
            print(f"epoch={epoch}: loss={loss_sum / count:.5f}, val mean Recall={score:.3f}", flush=True)
        save_npz(checkpoint, wi=best_wi, wt=best_wt,
                 train_sha256=np.array(sha256(train_path)), validation_sha256=np.array(sha256(val_path)))
        write_csv(root / "results" / f"{tag}_history.csv", history)
        write_json(root / "results" / f"{tag}_meta.json",
                   {"args": vars(args), "best_epoch": best_epoch, "validation_mean_recall": best,
                    "initialization": "training-only SVD", "train_captions_per_image": 1,
                    "train_seconds_excluding_validation": total_training_seconds,
                    "environment": environment(device), "test_used_for_selection": False})
        print(f"保存最佳验证 checkpoint：epoch {best_epoch}。如为 0，说明训练未超过 SVD 初始化。")
        print("下一步显式使用 --test-only 评价；本次训练没有读取测试集。")
        return
    with np.load(checkpoint, allow_pickle=False) as z:
        if str(z["train_sha256"]) != sha256(train_path) or str(z["validation_sha256"]) != sha256(val_path):
            raise ValueError("训练或验证特征已改变，不能复用 checkpoint。")
        wi, wt = z["wi"], z["wt"]
    ti, tt, to, test_meta = load_features(root, "test")
    check_same_encoder(train_meta, test_meta)
    rows = []
    for method, a, b in [("original", None, None), ("svd", v[:, :args.dimension].astype(np.float32), None),
                          ("linear", wi, wt)]:
        zi, zt = project(ti, tt, a, b)
        rows.append({"method": method, "dimension": 512 if a is None else args.dimension,
                     **recalls(zi, zt, to, device)})
    write_csv(root / "results" / f"{tag}_test.csv", rows)
    print(f"测试结果：results/{tag}_test.csv")


if __name__ == "__main__":
    main()
