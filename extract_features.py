"""冻结 CLIP，分阶段提取并缓存特征；不进行梯度训练。"""
import argparse
import json
import time
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from common import (SPLITS, configure, environment, normalize,
                    read_json, save_npz, sha256, sync, write_json)


class Images(Dataset):
    def __init__(self, base, rows, preprocess):
        self.base, self.rows, self.preprocess = base, rows, preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.base / self.rows[index]["path"]) as image:
            return self.preprocess(image.convert("RGB"))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    p.add_argument("--checkpoint", help="可选：本地官方 ViT-B-32.pt 权重文件")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    configure(threads=args.threads)
    device = torch.device(args.device)
    root = Path(args.root)
    manifest_path = root / "data" / "manifest.json"
    manifest = read_json(manifest_path)
    signature = sha256(manifest_path)
    model_dir = root / "models" / "clip"
    start = time.perf_counter()
    model, preprocess = clip.load(args.checkpoint or "ViT-B/32", device=device,
                                  jit=False, download_root=str(model_dir))
    model.eval().requires_grad_(False)
    # 全程使用 float32，避免 CPU/GPU 提取精度差异混入压缩对照。
    model.float()
    checkpoint = Path(args.checkpoint) if args.checkpoint else model_dir / "ViT-B-32.pt"
    checkpoint_hash = sha256(checkpoint)
    expected_hash = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
    if args.checkpoint and checkpoint_hash != expected_hash:
        raise ValueError("权重不是方案指定的官方 CLIP ViT-B/32 文件（SHA256 不匹配）。")
    sync(device)
    write_json(root / "features" / "environment.json",
               {**environment(device), "model_load_seconds": time.perf_counter() - start})
    for split in args.splits:
        out = root / "features" / f"{split}.npz"
        config = {"manifest_sha256": signature, "checkpoint_sha256": checkpoint_hash,
                  "model": "ViT-B/32", "precision": "float32", "split": split,
                  "train_caption": 0, "token_truncation": True}
        if out.exists():
            with np.load(out, allow_pickle=False) as z:
                old = json.loads(str(z["meta"]))
            if any(old[k] != v for k, v in config.items()):
                raise ValueError(f"已有缓存配置不一致：{out}。请换用新的 --root。")
            print(f"复用 {out}", flush=True)
            continue
        rows = manifest["splits"][split]
        loader = DataLoader(Images(root / "data", rows, preprocess), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
        image_batches = []
        sync(device)
        start = time.perf_counter()
        for batch in tqdm(loader, desc=f"{split} images"):
            feat = model.encode_image(batch.to(device, non_blocking=True)).float().cpu().numpy()
            image_batches.append(feat)
        sync(device)
        image_seconds = time.perf_counter() - start
        captions = [c for row in rows for c in (row["captions"][:1] if split == "train" else row["captions"])]
        text_batches = []
        sync(device)
        start = time.perf_counter()
        for begin in tqdm(range(0, len(captions), args.batch_size), desc=f"{split} texts"):
            tokens = clip.tokenize(captions[begin:begin + args.batch_size], truncate=True).to(device)
            text_batches.append(model.encode_text(tokens).float().cpu().numpy())
        sync(device)
        text_seconds = time.perf_counter() - start
        images = normalize(np.concatenate(image_batches)).astype(np.float32)
        texts = normalize(np.concatenate(text_batches)).astype(np.float32)
        owners = np.arange(len(rows)) if split == "train" else np.repeat(np.arange(len(rows)), 5)
        meta = {**config, "image_extract_seconds": image_seconds, "text_extract_seconds": text_seconds,
                "timing_scope": "image: decode/preprocess/H2D/encode/D2H; text: tokenize/H2D/encode/D2H; excludes model load and final normalization/save",
                "batch_size": args.batch_size, "workers": args.workers, "environment": environment(device)}
        save_npz(out, images=images, texts=texts, owners=owners, meta=np.array(json.dumps(meta)))
        write_json(root / "features" / f"{split}_timing.json", meta)
        print(f"保存 {out}: images={images.shape}, texts={texts.shape}", flush=True)


if __name__ == "__main__":
    main()
