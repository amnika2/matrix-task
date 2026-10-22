# Flickr8k + CLIP + 联合 SVD 特征压缩实验

对应题目：**奇异值分解与低秩近似在图文检索特征压缩中的应用研究**。

## 1. 包含哪些实验

| 文件 | 作用 |
| --- | --- |
| `check_env.py` | 检查 PyTorch、CUDA、CLIP 安装，不启动实验 |
| `prepare_data.py` | 下载 Flickr8k，固定数据版本，保留 6000/1000/1000 划分 |
| `extract_features.py` | 冻结官方 CLIP ViT-B/32，提取归一化 float32 特征并按 split 缓存 |
| `fit_svd.py` | 实验一：奇异值和能量；实验二：理论与实测重建误差 |
| `evaluate.py` | 实验三：双向 Recall；实验四：存储与相似度计算耗时；可选随机投影 |
| `plot_results.py` | 生成报告用 PNG 和 PDF 图表，无需桌面环境 |
| `train_projection.py` | 可选：训练两个线性投影矩阵，CLIP 始终冻结 |
| `common.py` | 共用数据校验、检索和计时函数 |

主实验没有梯度训练：CLIP 做实际编码推理，SVD 做数据拟合。若要实际训练，执行第 8 节的可选实验六。

## 2. 上传到远程服务器

将整个 `clip_svd_experiment` 文件夹上传到服务器的个人目录，例如 `~/clip_svd_experiment`。可以使用 VS Code Remote SSH 的文件上传、SFTP，或者在本机 PowerShell 中执行下面命令，将 `USERNAME`、`SERVER` 替换成你的登录信息：

```powershell
scp -r "C:\Users\Lenovo\Documents\ChatGPT\矩阵理论\clip_svd_experiment" USERNAME@SERVER:~/
```

如果 SSH 使用自定义端口，在 `scp` 后加 `-P 端口号`。后面的命令均在远程服务器的 Linux shell 中运行。

```bash
cd ~/clip_svd_experiment
nvidia-smi
python3 --version
```

建议使用独立 Python 3.11 环境；预留至少 10 GB 可用磁盘作为数据、下载缓存、模型权重和输出空间的工作余量，CPU 内存建议至少 8 GB。这些是运行准备建议，实际资源占用需要在服务器确认。

## 3. 创建环境并安装依赖

如果服务器有 Conda：

```bash
conda create -n clip-svd python=3.11 -y
conda activate clip-svd
```

如果没有 Conda，但已有 Python 3.11，替代上面两条命令：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

然后：

```bash
python -m pip install --upgrade pip setuptools wheel
```

**先安装与你的 NVIDIA 驱动、GPU 架构兼容的一对 torch / torchvision。** 可参考 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/) 或 [官方历史版本表](https://pytorch.org/get-started/previous-versions/)。`nvidia-smi` 显示的 CUDA Version 不能替代实际的 GPU 和驱动兼容检查。

下面是一组官方提供的固定版本示例，**仅在服务器适合 CUDA 12.6 发行包时使用**，不要求服务器另行安装相同版本的 CUDA Toolkit：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

若 GPU 架构或驱动不适合这组版本，应改用官方对应组合；不要把安装成功当作 GPU 可实际运行的证明。没有 GPU 时，可以改装 CPU 版本，但图像编码会更慢：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

上面 GPU / CPU 两条安装命令只选一条。继续安装其他依赖：

```bash
python -m pip install -r requirements.txt
python -m pip check
python check_env.py --device cuda
mkdir -p logs
python -m pip freeze > logs/environment_freeze.txt
```

纯 CPU 环境将 `--device cuda` 改成 `--device cpu`。CLIP 使用官方 GitHub 仓库的固定提交，安装时需要 `git` 和 GitHub 连通性。**不要单独执行 `pip install clip`**，该名称可能安装到其他同名项目。

为统一 CPU 线性代数线程数，在运行后续阶段前设置：

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

环境检查会实际做一个极小矩阵运算确认 GPU kernel 可用，不会下载数据或 CLIP 权重。

## 4. 准备数据并提取 CLIP 特征

先下载并整理数据：

```bash
python -u prepare_data.py
```

来源为 [jxie/flickr8k 数据集](https://huggingface.co/datasets/jxie/flickr8k)，其数据卡提供 train 6000、validation 1000、test 1000 的划分，以及 `caption_0` 到 `caption_4` 五列描述。第一次运行解析并记录精确数据提交 SHA，后续重试使用同一提交，不重新随机划分。

完成后应有：

```text
data/source.json
data/manifest.json
data/images/train/...
data/images/validation/...
data/images/test/...
```

图片保存为无损 PNG，避免再次 JPEG 编码改变输入。保留数据与权重用于学习研究时，请遵守其原始使用条款。

再提取特征。为便于逐步确认，建议先只做训练集：

```bash
python -u extract_features.py --device cuda --batch-size 64 --workers 4 --splits train
```

检查最后应显示：

```text
images=(6000, 512), texts=(6000, 512)
```

随后提取验证集与测试集：

```bash
python -u extract_features.py --device cuda --batch-size 64 --workers 4 --splits validation test
```

两者分别应显示：

```text
images=(1000, 512), texts=(5000, 512)
```

说明：

- 第一次提取会下载官方 CLIP ViT-B/32 权重。权重 SHA256 会校验为方案指定版本。
- 训练集只提取每图 `caption_0`，足够构成 `12000 × 512` 联合矩阵；验证集和测试集使用全部五条文本。
- 原始五条训练描述仍完整保存在数据清单；这里减少的是特征提取量。
- 图像与文本编码均采用 float32，以便 CPU/GPU 设置一致；这比半精度编码占用更多显存。
- 提取时间按 split 单独保存，包含图像读取/预处理或文本分词、数据传输和编码；排除模型下载/加载及最终归一化/保存。
- 完成一个 split 后原子保存对应缓存。重新执行会复用配置一致的完整 split；中途打断的 split 需要重新提取，不支持单批次续算。
- 显存不足先将 `--batch-size 64` 改成 `--batch-size 16`。数据读取进程异常时改用 `--workers 0`。
- 如果需要换 GPU，可用 `--device cuda:1` 或设置 `CUDA_VISIBLE_DEVICES`。评价前后必须维持同一套特征缓存。

输出：`features/train.npz`、`features/validation.npz`、`features/test.npz`，以及环境与提取计时 JSON。

## 5. 实验一、二：SVD 与重建误差

```bash
python -u fit_svd.py
python plot_results.py
```

此阶段使用 CPU NumPy 的 float64 直接 SVD。不做均值中心化，不把 `X.T @ X` 的特征分解替代直接 SVD。只使用训练集的 6000 个图像和 6000 个文本向量。

每个维度计算 `X - (X @ V_k) @ V_k.T` 的实测 Frobenius 范数，并与奇异值尾部平方和的平方根比较；实际误差过大时直接报错。这里验证定理在数据上的误差公式，数学证明仍需在报告理论章节给出。

输出：

| 文件 | 内容 |
| --- | --- |
| `models/svd.npz` | float64 右奇异向量、奇异值、训练特征校验值 |
| `models/svd_meta.json` | 拟合时间、矩阵形状、正交性误差、数值秩 |
| `results/singular_values.csv` | 512 个奇异值与累计能量 |
| `results/reconstruction.csv` | 512/384/256/128/64/32 的理论与实测误差 |
| `figures/singular_values.png/.pdf` | 奇异值曲线 |
| `figures/cumulative_energy.png/.pdf` | 能量保留率曲线 |
| `figures/reconstruction.png/.pdf` | 理论与实测相对重建误差 |

若还要实测谱范数，使用 `python -u fit_svd.py --spectral`；这会对每个残差矩阵额外分解，耗时增加。默认只实测 Frobenius 范数，同时输出理论谱误差。

SVD 的最佳近似误差是在**投影后、再次归一化之前**衡量。检索时再次 L2 归一化是另一项操作，两者不能混为一谈。

## 6. 实验三、四：先验证集，后测试集

固定选择规则：在验证集的 6 个 Recall 指标平均值相比原始特征下降不超过 **1.0 个百分点**的候选中，选择最小 SVD 维度。平均值仅用于统一选型，仍输出每项 Recall。若无 SVD 候选满足则回退原始特征。该阈值应在查看测试结果前确定。

```bash
python -u evaluate.py --split validation --device cpu --repeats 30 --max-drop 1.0
```

完成后 `results/selection.json` 固定选中的配置。之后运行：

```bash
python -u evaluate.py --split test --device cpu --repeats 30
python plot_results.py
```

CPU 作为小规模特征检索的默认测量设备；CLIP 特征提取仍使用 GPU。若你希望测 GPU 检索，将两条评价命令的 `--device cpu` 同时改成 `--device cuda`，并在报告中说明。不要把不同设备或不同线程数的计时放到同一个对比中。

检索协议：T2I 使用 split 内全部 1000 张图像作为候选，每条文本对应一个正例；I2T 使用全部 5000 条文本作为候选，前 K 中含本图任意一条描述即命中。Recall@1/5/10 的单位均为百分比。分批只划分查询，不缩小候选图库。同分使用候选行号稳定排序。

输出：

| 文件 | 内容 |
| --- | --- |
| `results/retrieval_validation.csv`、`retrieval_test.csv` | 原始 512 维和各 SVD 维度的 T2I/I2T R@1/5/10 |
| `results/selected_test.json` | 验证集所选配置对应的最终测试指标 |
| `results/efficiency_validation.csv`、`efficiency_test.csv` | 存储字节数、投影及相似度计算耗时 |
| `results/full_dimension_check_*.json` | 完整 512 维投影前后相似度误差与 Recall 差值 |
| `results/evaluation_*_meta.json` | 测量设备、线程、版本、重复次数、缓存校验值 |
| `figures/retrieval_*.png/.pdf` | 双向 Recall 随维度变化曲线 |
| `figures/efficiency_*.png/.pdf` | 特征存储与相似度计算时间 |

512 维完整投影分别做 float64 和实际 float32 校验。float32 近似同分项可能有轻微排序变化；相似度差异明显或 Recall 变化超过 0.2 个百分点时程序停止，便于排查。

存储与计时的口径：

- 存储为 float32 数组有效载荷；分别给出图像、文本、二者之和，以及额外一个 `512 × k` 投影矩阵。每个图像图库或文本图库单独存储时，也需额外计入同一个投影矩阵。
- `models/svd.npz` 为理论验证保留了完整 float64 基；它的文件大小不等于部署时 float32 `V_k` 的有效载荷。CSV 的部署存储估算不包含 NPZ 头、奇异值、图文 ID 或元数据。
- 投影计时包含图像和文本两侧的矩阵乘法及归一化，排除数据传输。
- T2I/I2T 各自计时是完整相似度矩阵的一次矩阵乘法，先预热 5 次，再重复 30 次，输出中位数和四分位距，CUDA 会显式同步。
- 相似度计时不含排序、投影、CLIP 编码、读取和网络下载，不能称为完整系统端到端加速。小矩阵未必在 GPU 或低维下更快，应如实报告。
- 查看全部测试维度是报告预先安排的消融实验；最终配置从 `selection.json` 获取，不能再用测试集挑选维度。

## 7. 可选实验五：随机正交投影

在同样的缓存上比较 128 和 256 维，固定三个不同种子，报告均值与样本标准差：

```bash
python -u evaluate.py --split validation --device cpu --random-seeds 11 22 33
python -u evaluate.py --split test --device cpu --random-seeds 11 22 33
python plot_results.py
```

这两条命令会重算并覆盖同 split 的检索和效率 CSV，重新包含 SVD 与随机投影；数据和 CLIP 特征不需要重新提取。保持选择阈值与主实验一致，默认仍为 1.0。

输出 `random_summary_*.csv` 中的 Recall 均值/标准差和相对重建误差均值/标准差，单次结果保留在 `retrieval_*.csv` 与 `random_reconstruction_*.csv`；投影矩阵保存在 `models/random_seed*_k*.npz`。

随机投影使用高斯矩阵 QR 得到正交基，图文共用同一个矩阵，重建为 `X Q_k Q_k.T`。这是随机**正交子空间**投影，不是分别为图像和文本抽样，也不是使用任意高斯矩阵后误用转置当逆矩阵。

## 8. 可选实验六：实际训练轻量投影层

在训练缓存的 6000 个一对一图文对上，学习两个 `512 × 128` 矩阵。采用双向 InfoNCE，温度 0.07，AdamW，学习率 0.001，5 个 epoch，batch size 256。用训练集 SVD 初始化，两侧训练矩阵可以独立变化。

```bash
python -u train_projection.py --device cuda --dimension 128 --epochs 5 --seed 42
```

每个 epoch 在完整验证集评价 mean Recall，和 epoch 0 的 SVD 初始化比较，仅保留最佳配置；若最佳 epoch 为 0，应如实说明训练没有改善验证指标。训练命令完全不读取测试特征。

然后显式评价最佳 checkpoint：

```bash
python -u train_projection.py --device cuda --dimension 128 --seed 42 --test-only
```

输出 `results/linear_k128_seed42_history.csv`、对应元数据 JSON、`models/linear_k128_seed42.npz`，以及原始特征/SVD/可训练投影的对照表 `results/linear_k128_seed42_test.csv`。

已存在的同名训练 checkpoint 不会被覆盖。要做独立重复请换 seed；要更改同 seed 的训练配置请使用另一个 `--root` 并准备对应缓存。可训练方法需要两份矩阵，部署时额外有效载荷为 `2 × 512 × k × 4` 字节。训练曲线保留 CSV，主绘图脚本目前只自动生成四项必做实验的图。

## 9. SSH 断开、离线运行和重跑

长阶段可放进服务器已有的 tmux 会话：

```bash
tmux new -s clip-svd
```

在会话内正常激活环境并运行命令。按 `Ctrl+B` 后按 `D` 可分离，之后用 `tmux attach -t clip-svd` 恢复。若没有 tmux，可在成功验证命令后使用：

```bash
mkdir -p logs
nohup python -u extract_features.py --device cuda --batch-size 64 --workers 4 > logs/extract.log 2>&1 &
tail -f logs/extract.log
```

`Ctrl+C` 退出 `tail` 不会停止后台任务。不要同时启动多个写入相同 root 的实验。

网络只在安装、下载数据和首次下载权重时需要。GitHub / Hugging Face / 模型下载源必须在你的服务器可达，不会自动配置代理或使用第三方镜像。

如果服务器不能联网，可在另一台可联网电脑安装依赖后运行 `prepare_data.py`，将完整 `data/` 上传；同时从 [OpenAI 官方 CLIP 源码所列地址](https://github.com/openai/CLIP/blob/main/clip/clip.py)下载官方 `ViT-B-32.pt` 并上传，提取时指定：

```bash
python -u extract_features.py --checkpoint /absolute/path/ViT-B-32.pt --device cuda
```

依赖包仍须在服务器预先安装；离线安装还需准备匹配服务器系统和 Python 的 wheel 包。后续 SVD、检索和训练都只读本地缓存。

所有脚本默认在当前目录读写。`--root` 代表**数据和结果根目录**，不是只改变输出位置；如改为 `--root runs/exp1`，必须在每个阶段使用同一参数。建议第一次保持默认。

运行失败时先看第一条异常，不要继续执行后续阶段；保留报错和运行命令。修改数据清单或训练特征后会触发校验，防止混用旧投影。评价 CSV / 图表会覆盖同名结果，若要对比独立环境，请分别保存完整实验目录。

## 10. 实验结束后保留的材料

保留 `data/source.json`、`data/manifest.json`、`features/*_timing.json`、`features/environment.json`、`models/svd_meta.json`、`results/`、`figures/` 和 `logs/environment_freeze.txt`。它们记录数据版本、数据对应、配置、数值校验、实际指标和计时，可用于后续写报告。图文特征缓存也建议保留，便于增加实验而无需再次编码。

实现资源：[OpenAI CLIP](https://github.com/openai/CLIP)、[Flickr8k 数据卡](https://huggingface.co/datasets/jxie/flickr8k/blob/main/README.md)、[Hugging Face 数据加载说明](https://huggingface.co/docs/datasets/loading)、[PyTorch 安装说明](https://pytorch.org/get-started/locally/)。
