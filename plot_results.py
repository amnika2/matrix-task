"""离线读取 CSV 生成可用于报告的 PNG / PDF；SSH 下无需图形界面。"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    args = p.parse_args()
    root = Path(args.root)
    results, folder = root / "results", root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25})
    path = results / "singular_values.csv"
    if path.exists():
        rows = read_csv(path)
        indices = [int(r["index"]) for r in rows]
        for key, name, label in [("singular_value", "singular_values", "Singular value"),
                                  ("cumulative_energy", "cumulative_energy", "Cumulative energy retained")]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(indices, [float(r[key]) for r in rows])
            ax.set(xlabel="Singular value index", ylabel=label)
            if key == "cumulative_energy":
                ax.set_ylim(0, 1.02)
            save(fig, folder, name)
    path = results / "reconstruction.csv"
    if path.exists():
        rows = sorted(read_csv(path), key=lambda r: int(r["dimension"]))
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, label, style in [("relative_fro_theory", "Theoretical", "-"),
                                  ("relative_fro_measured", "Measured", "--")]:
            ax.plot([int(r["dimension"]) for r in rows], [float(r[key]) for r in rows], style, marker="o", label=label)
        ax.set(xlabel="Retained dimension", ylabel="Relative Frobenius error")
        ax.legend()
        save(fig, folder, "reconstruction")
    for split in ["validation", "test"]:
        path = results / f"retrieval_{split}.csv"
        if path.exists():
            all_rows = read_csv(path)
            baseline = next(r for r in all_rows if r["method"] == "original")
            rows = sorted([r for r in all_rows if r["method"] == "svd"], key=lambda r: int(r["dimension"]))
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            for ax, direction in zip(axes, ["T2I", "I2T"]):
                for k, color in zip([1, 5, 10], ["C0", "C1", "C2"]):
                    key = f"{direction}_R@{k}"
                    ax.plot([int(r["dimension"]) for r in rows], [float(r[key]) for r in rows],
                            marker="o", color=color, label=f"SVD R@{k}")
                    ax.axhline(float(baseline[key]), color=color, linestyle=":", alpha=0.65, label=f"Original R@{k}")
                ax.set(xlabel="Retained dimension", ylabel="Recall (%)", title=f"{direction} ({split})")
                ax.legend(fontsize=8)
            save(fig, folder, f"retrieval_{split}")
        path = results / f"efficiency_{split}.csv"
        if path.exists():
            rows = sorted([r for r in read_csv(path) if r["method"] == "svd"], key=lambda r: int(r["dimension"]))
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            dims = [int(r["dimension"]) for r in rows]
            axes[0].plot(dims, [float(r["both_feature_bytes"]) / 2**20 for r in rows], marker="o", label="Features")
            axes[0].plot(dims, [float(r["both_plus_projection_bytes"]) / 2**20 for r in rows], marker="s", label="Features + projection")
            axes[0].set(xlabel="Retained dimension", ylabel="Float32 storage (MiB)")
            for direction in ["T2I", "I2T"]:
                axes[1].plot(dims, [float(r[f"{direction}_matmul_ms_median"]) for r in rows], marker="o", label=direction)
            axes[1].set(xlabel="Retained dimension", ylabel="Similarity matmul median (ms)")
            for ax in axes:
                ax.legend()
            save(fig, folder, f"efficiency_{split}")
    print(f"图表已输出到 {folder}；英文坐标避免远程服务器缺少中文字体。")


if __name__ == "__main__":
    main()
