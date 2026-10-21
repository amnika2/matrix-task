"""检查运行环境，不下载数据、不加载模型、不开始实验。"""
import argparse
import json

import clip
import torch
import torchvision

from common import configure, environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    configure()
    device = torch.device(args.device)
    if not hasattr(clip, "load") or not hasattr(clip, "tokenize"):
        raise RuntimeError("安装了错误的 clip 包；请按 requirements.txt 安装 OpenAI 官方版本。")
    # 极小矩阵只用于确认 CUDA kernel 能在当前 GPU 架构上实际执行。
    x = torch.ones((2, 2), device=device)
    assert (x @ x).sum().item() == 8
    print(json.dumps(environment(device), ensure_ascii=False, indent=2))
    print(f"torchvision={torchvision.__version__}; CLIP={clip.__file__}")
    print("环境检查通过，可以进行数据准备。")


if __name__ == "__main__":
    main()
