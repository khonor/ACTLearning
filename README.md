# ACTLearning

从零理解并复现 **ACT（Action Chunking with Transformers）**，用于机器人操作策略学习。

本仓库当前的核心产物是教学 notebook：

- [`notebooks/ACT_tutorial.ipynb`](notebooks/ACT_tutorial.ipynb) — 中文，面向初学者，覆盖原理、数据管线、从零实现、真实数据训练、LeRobot 官方 ACT 对齐、部署推理与评估排错。

## 这个 notebook 讲什么

1. **数据**：LeRobot v3 数据集、双相机、7 维关节、action chunk、padding、归一化。
2. **原理**：行为克隆的误差累积与多模态问题、action chunking、CVAE、Transformer encoder/decoder、时序集成。
3. **从零实现**：一个约 1.4M 参数的 MiniACT（CNN + CVAE + Transformer），在真实数据上训练并评估。
4. **官方对齐**：LeRobot 0.6.1 `ACTPolicy` 的配置、forward、预/后处理、短训练和正式训练命令。
5. **部署**：动作队列 vs 时序集成、MiniACT/官方 ACT 控制器、回放式控制循环、延迟测量、真机代码骨架与安全检查清单。
6. **评估与排错**：开环 L1、闭环成功率、常见现象与超参速查。

## 环境

服务器实测环境：

| 项 | 值 |
|---|---|
| 主机 | `panjw-System-Product-Name`，用户 `yuntao` |
| 项目根目录 | `/data/home/yuntao/MyProject/4_ACTLearning/ACTLearning` |
| Conda 环境 | `lerobot`（Python 3.12.14，PyTorch 2.11.0+cu128） |
| GPU | 2 × RTX 3090 Ti（24 GB） |
| 数据 | `~/MyProject/2_Alohamini2/datasets/am2_right_arm_test`（30 episodes，30 fps） |

Notebook 需要的额外包已安装到 `lerobot` 环境：`matplotlib`、`nbconvert`、`ipykernel`、`jupyterlab`。
Jupyter 内核名为 `lerobot`，显示名为 **Python (lerobot ACT)**。

## 快速开始

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
cd ~/MyProject/4_ACTLearning/ACTLearning

# 数据软链（如果还没有）
mkdir -p data
ln -sfn ~/MyProject/2_Alohamini2/datasets/am2_right_arm_test data/am2_right_arm_test

# 启动 JupyterLab（本地通过 SSH 隧道访问 8888）
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
```

无界面执行/导出：

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/ACT_tutorial.ipynb
jupyter nbconvert --to html notebooks/ACT_tutorial.ipynb
```

## 产物

```
outputs/
├── cache/mini_forward_64.pt   # 30 条 episode 的 64×64 前视帧缓存（约 192 MB，可重建）
├── mini_act/model.pt          # 教学用 MiniACT
└── official_act/              # 官方 ACT 短训练 checkpoint + stats + history
```

`data/` 和 `outputs/` 已在 `.gitignore` 中，不会入库。

## 结果说明

- MiniACT（600 步、24 条训练演示、6 条 held-out）：验证集归一化 L1 **0.2193**，
  略优于 repeat-state baseline 的 **0.2285**；回放式控制循环动作 MAE 约 **3.0–3.3°**，
  batch=1 推理 p50 **7–10 ms**（随共享 GPU 负载波动）。它只用于验证管线，不是可上真机的策略。
- 官方 ACT 短训练（900 步、dim_model=256）：验证集归一化 L1 **0.2278**，与 repeat-state **0.2218**
  接近但略差。batch=1 推理在空闲 3090 Ti 上 p50 约 **16.5 ms**；服务器上有其他 GPU 任务时本次实测
  约 **97 ms**。900 步不到一个 epoch，远未收敛；正式训练请按 notebook 第 5.4 节跑 50k 步。
- **延迟必须在目标控制机上实测**：同一模型在不同负载下可能差 5–6 倍，部署时看 p95 并留余量。
- 这些数字的意义是 **建立可复现的评估基线**，不是论文级性能。

## 参考

- 论文：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- 原实现：[tonyzhaozh/act](https://github.com/tonyzhaozh/act)
- LeRobot：`lerobot/policies/act/`
