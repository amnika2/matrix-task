"""下载固定版本的 Flickr8k，保留已有划分，建立显式图文 ID。"""
import argparse
import io
from pathlib import Path

from datasets import Image as HFImage, load_dataset
from huggingface_hub import HfApi
from PIL import Image
from tqdm import tqdm

from common import SPLITS, read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--revision", default="main", help="HF 数据集提交 SHA、分支或标签")
    args = p.parse_args()
    out = Path(args.root) / "data"
    manifest = out / "manifest.json"
    if manifest.exists():
        print(f"已有完整数据清单：{manifest}；保持不变。")
        return
    out.mkdir(parents=True, exist_ok=True)
    revision_file = out / "source.json"
    if revision_file.exists():
        revision = read_json(revision_file)["revision"]
    else:
        revision = HfApi().dataset_info("jxie/flickr8k", revision=args.revision).sha
        write_json(revision_file, {"repository": "jxie/flickr8k", "revision": revision})
    dataset = load_dataset("jxie/flickr8k", revision=revision)
    records = {}
    for split, expected in SPLITS.items():
        ds = dataset[split].cast_column("image", HFImage(decode=False))
        if len(ds) != expected:
            raise ValueError(f"{split} 应有 {expected} 张图像，实际 {len(ds)}。")
        folder = out / "images" / split
        folder.mkdir(parents=True, exist_ok=True)
        records[split] = []
        for index, row in enumerate(tqdm(ds, desc=split)):
            field = row["image"]
            source = io.BytesIO(field["bytes"]) if field["bytes"] is not None else field["path"]
            with Image.open(source) as im:
                rgb = im.convert("RGB")
                image_path = folder / f"{index:05d}.png"
                rgb.save(image_path)
            captions = [row[f"caption_{j}"] for j in range(5)]
            if any(not isinstance(c, str) or not c.strip() for c in captions):
                raise ValueError(f"{split}:{index} 描述缺失。")
            records[split].append({"image_id": f"{split}:{index:05d}",
                                   "path": image_path.relative_to(out).as_posix(),
                                   "captions": captions})
    write_json(manifest, {"source": "jxie/flickr8k", "revision": revision, "splits": records})
    print("完成：train 6000/6000 拟合图文对；validation 和 test 各 1000 图 / 5000 文。")


if __name__ == "__main__":
    main()
