"""共用的数据校验、检索协议与计时函数。"""
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

SPLITS = {"train": 6000, "validation": 1000, "test": 1000}
DIMS = [512, 384, 256, 128, 64, 32]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez(f, **arrays)
    temp.replace(path)


def normalize(x):
    x = np.asarray(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms < 1e-12):
        raise ValueError("特征含非有限值或零向量。")
    return x / norms


def configure(seed=42, threads=4):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def environment(device):
    from importlib.metadata import version
    packages = {name: version(name)
                for name in ["torch", "torchvision", "numpy", "Pillow", "datasets", "clip"]}
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "device": str(device), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_threads": torch.get_num_threads(), "tf32": False}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn, device, repeats=30, warmups=5):
    """只测传入操作，排除数据传输；返回中位数和四分位距（毫秒）。"""
    for _ in range(warmups):
        result = fn()
    sync(device)
    samples = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        result = fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000)
    # 保持返回值存活至同步结束，确保设备计算实际完成。
    del result
    return float(np.median(samples)), float(np.percentile(samples, 75) - np.percentile(samples, 25))


def load_features(root, split):
    path = Path(root) / "features" / f"{split}.npz"
    with np.load(path, allow_pickle=False) as z:
        images, texts, owners = z["images"], z["texts"], z["owners"]
        meta = json.loads(str(z["meta"]))
    if meta["manifest_sha256"] != sha256(Path(root) / "data" / "manifest.json"):
        raise ValueError("数据清单已改变，请使用新的实验目录重新提取特征。")
    return images, texts, owners, meta


def load_svd(root):
    path = Path(root) / "models" / "svd.npz"
    with np.load(path, allow_pickle=False) as z:
        v, s = z["v"], z["s"]
        signature = str(z["train_sha256"])
    if signature != sha256(Path(root) / "features" / "train.npz"):
        raise ValueError("训练特征已改变，请重新拟合 SVD。")
    return v, s


def check_same_encoder(train_meta, eval_meta):
    for key in ["checkpoint_sha256", "precision", "model"]:
        if train_meta[key] != eval_meta[key]:
            raise ValueError(f"训练集和评价集特征提取设置不一致：{key}")


@torch.inference_mode()
def recalls(images, texts, owners, device, batch_size=256):
    """T2I 一条文本一个正例；I2T 五条文本中任意一个命中即可。

    在完整 split 图库中排序，分批仅限制内存，不缩小候选图库。
    同分时稳定排序，以候选原始行号打破平局。
    """
    i = torch.as_tensor(images, dtype=torch.float32, device=device)
    t = torch.as_tensor(texts, dtype=torch.float32, device=device)
    o = torch.as_tensor(owners, dtype=torch.long, device=device)
    metrics = {}
    for direction, query, gallery in [("T2I", t, i), ("I2T", i, t)]:
        hits = {k: 0 for k in (1, 5, 10)}
        for start in range(0, len(query), batch_size):
            score = query[start:start + batch_size] @ gallery.T
            order = torch.argsort(score, dim=1, descending=True, stable=True)[:, :10]
            if direction == "T2I":
                good = order == o[start:start + len(order), None]
            else:
                qids = torch.arange(start, start + len(order), device=device)
                good = o[order] == qids[:, None]
            for k in hits:
                hits[k] += int(good[:, :k].any(dim=1).sum().item())
        for k in hits:
            metrics[f"{direction}_R@{k}"] = 100.0 * hits[k] / len(query)
    metrics["mean_recall"] = float(np.mean(list(metrics.values())))
    return metrics


def project(images, texts, wi=None, wt=None):
    if wi is None:
        return images, texts
    if wt is None:
        wt = wi
    return (normalize(images @ wi).astype(np.float32),
            normalize(texts @ wt).astype(np.float32))
