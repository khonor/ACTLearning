# AGENTS.md

本文件面向在此仓库工作的 AI 编码代理（也适用于人类协作者）。开始任何改动前先读完本文件与 `README.md`。

## 1. 项目目标

本仓库是一个**教学型仓库**：用中文材料从零讲清 **ACT（Action Chunking with Transformers）**，并给出可复现的实现与评估基线。

当前核心产物是 `notebooks/ACT_tutorial.ipynb`，覆盖：

- **原理**：行为克隆的误差累积与多模态问题、action chunking、CVAE、Transformer encoder/decoder、时序集成。
- **从零实现**：约 1.4M 参数的 MiniACT（CNN + CVAE + Transformer），在真实数据上训练与评估。
- **官方对齐**：LeRobot 0.6.1 `ACTPolicy` 的配置、forward、预处理/后处理，以及短训练与正式训练命令。
- **部署与评估**：动作队列 vs 时序集成、控制循环骨架、推理延迟实测、开环 L1 与闭环成功率、常见排错。

目标读者既是初学者，也包括后续在此仓库继续迭代的人。因此：**结论要可复现，数字要能追溯到命令**。

## 2. 运行环境（服务器实测信息）

| 项 | 值 |
|---|---|
| 主机 | `panjw-System-Product-Name`，登录用户 `yuntao` |
| 项目根目录 | `/data/home/yuntao/MyProject/4_ACTLearning/ACTLearning` |
| 系统 | Ubuntu 22.04.5 LTS，内核 6.8.0 |
| GPU | 2 × NVIDIA RTX 3090 Ti（各 24 GB），驱动 580.178.04 |
| 磁盘 | `/data` 3.2 T，剩余约 1.2 T |
| 主力环境 | conda `lerobot`（Python 3.12.14，PyTorch 2.11.0+cu128，`lerobot 0.6.1`） |

其他环境：`lerobot_alohamini`（Python 3.12.12，PyTorch 2.10.0+cu128，公司定制 LeRobot）、`rl`（Python 3.10，与本项目无关）。

notebook 额外依赖已装在 `lerobot` 环境：`matplotlib`、`nbconvert`、`ipykernel`、`jupyterlab`。Jupyter 内核名 `lerobot`，显示名 **Python (lerobot ACT)**。

注意：

- **不要**往 `lerobot` 环境里随意 `pip install` 升级已有包，以免破坏已跑通的官方 ACT baseline；需要新依赖先在 issue/提交说明里写清理由。
- 非交互式 shell（如 `ssh host 'cmd'`）不会自动初始化 conda，需要：
  ```bash
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate lerobot
  ```
- 网络受限：HuggingFace 走镜像，且需关闭 xet 加速：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_DISABLE_XET=1
  ```

## 3. 目录结构（当前实际）

```
ACTLearning/
├── AGENTS.md
├── README.md
├── .gitignore
├── notebooks/
│   ├── ACT_tutorial.ipynb      # 主教程，含 MiniACT 实现与全部实验
│   ├── ACT_tutorial.html       # 导出物，便于不装环境直接阅读
│   └── assets/                 # notebook 引用的示意图
├── data/                       # 软链到数据集（gitignore）
└── outputs/                    # cache / checkpoint / 日志（gitignore）
```

约定：

- **notebook 是当前唯一载体**，但可复用的逻辑（模型、数据管线、评估指标）应逐步抽到 `src/act_learning/`，抽出后同步更新本节与 `README.md`。
- `outputs/cache/mini_forward_64.pt`（约 192 MB）是可重建缓存，**绝不入库**；重建方式写在 notebook 对应 cell 里。
- `notebooks/.ipynb_checkpoints/` 不提交（已在 `.gitignore`）。

## 4. 参考实现（只读，不要修改）

LeRobot 的 ACT 实现位于 `~/MyProject/2_Alohamini2/lerobot_alohamini/src/lerobot/policies/act/`：

- `modeling_act.py` — ACT 网络与损失（CVAE + Transformer + chunk）
- `configuration_act.py` — 超参默认值（`chunk_size`、`n_action_steps`、`dim_model`、`latent_dim` 等）
- `processor_act.py` — 输入归一化与动作反归一化

另外 `~/act_dsh/run_act_pusht.sh` 记录了本机跑 LeRobot ACT（pusht 仿真）的实际命令，可作为 CLI 参数参考。

对拍时**只读**这些文件；需要改动时必须改本仓库内的代码。

## 5. 数据

本机数据集：`~/MyProject/2_Alohamini2/datasets/am2_right_arm_test`

- 格式：LeRobot dataset **v3.0**，30 fps，30 条 episode
- `observation.state`：7 维（`arm_right_*` 关节位置）
- `action`：7 维（同关节顺序，最后一维为夹爪）
- 图像：`observation.images.forward`（及后续可扩展的双相机），480×640×3，视频编码
- 目录：`data/`（parquet）、`videos/`、`meta/`

约定：

- 数据与权重**绝不入库**；用 `ln -sfn` 软链到 `data/`，不要复制。
- 读取统一走 notebook 中的数据管线 cell（后续抽为 `src/act_learning/data/`）；不要在别处零散 `pd.read_parquet`。
- 训练/验证划分必须固定（按 episode 划分），归一化统计量必须落盘，保证评估可复现。

## 6. 常用命令

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate lerobot
cd ~/MyProject/4_ACTLearning/ACTLearning

# 数据软链（首次）
mkdir -p data && ln -sfn ~/MyProject/2_Alohamini2/datasets/am2_right_arm_test data/am2_right_arm_test

# 交互式：本地经 SSH 隧道访问 http://localhost:8888
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser

# 无界面执行全部 cell（会重建 outputs/ 缓存，耗时且占用 GPU）
jupyter nbconvert --to notebook --execute --inplace notebooks/ACT_tutorial.ipynb

# 重新导出 HTML
jupyter nbconvert --to html notebooks/ACT_tutorial.ipynb
```

> 注意：`--execute --inplace` 会覆盖 notebook 内已保存的输出，提交前确认这是你想要的结果。

## 7. 代码规范

- Python 3.12，函数带类型标注；公共函数写 docstring。
- 张量操作必须写明 shape 约定，如 `# (B, T, D)`；维度含义统一：`B` batch、`T` 时序长度、`D` 特征维。
- 超参集中放在 notebook 顶部的配置 cell（或后续的 `configs/`），禁止散落魔数；与 LeRobot 对齐的默认值要注明出处。
- 随机性：训练入口固定 seed，并把关键超参打印/写入 `outputs/`。
- 注释与文档用中文，变量名、函数名、API 用英文。
- 目前尚无 `pyproject.toml` / `tests/`；引入格式化（ruff）、测试（pytest）或打包时，一并补上配置并在本节说明。

## 8. Git 约定

- 主分支 `main` 保持可运行；功能开发用 `feat/<topic>`，修复用 `fix/<topic>`。
- 提交信息：`<type>: <简短说明>`，`type` ∈ `feat|fix|refactor|docs|test|chore`。
- 一次提交只做一件事；不要混入无关的格式化噪音。
- **不要** `git push --force`，**不要**擅自改写远端历史；推送前先确认。
- 大文件（`data/`、`outputs/`、`*.pt`、`*.safetensors`、`*.html` 之外的大产物）不入库；`.gitignore` 已覆盖主要项，新增产物类型先补规则。
- 推送走 SSH（`git@github.com:khonor/ACTLearning.git`），本机 HTTPS 无存储凭据。

## 9. 代理协作规则

- 动手前：读 `README.md`、`AGENTS.md`，`git status` 确认工作区状态，弄清未提交改动是谁的。
- 只在 `4_ACTLearning/ACTLearning` 内改动；**不要**碰同级的 `1_RL`、`2_Alohamini2`、`3_DiffusionPolicy`、`~/act_dsh`。
- 不要使用 `sudo`；不要改 `~/.bashrc`、`~/.gitconfig` 等全局配置。
- 训练/评估/notebook 执行等长任务：用 `tmux` 或 `nohup`，日志写 `outputs/logs/`，不要在前台阻塞。
- 双卡机器默认只占 `cuda:0`；确需多卡再显式指定，避免和别人抢卡（延迟数据会因共享负载波动，测量时要注明）。
- 改完至少验证受影响的 cell 或脚本能跑通；无法运行时如实说明未验证项，不要假装通过。
- 结论要有证据：给出命令、日志片段或指标数值，避免「应该没问题」这类判断。

## 10. 已知坑

- 直接 `ssh host 'conda activate ...'` 会失败，原因是非登录 shell 未加载 conda。
- 从 HuggingFace 下载会卡住：必须设 `HF_ENDPOINT` 并禁用 xet。
- `jupyter nbconvert --execute` 默认超时较短，长 cell 需 `--ExecutePreprocessor.timeout=-1`。
- 数据集视频解码依赖 `ffmpeg` / `torchcodec`，换环境后先跑一次数据加载自检。
- 3090 Ti 单卡 24 GB：480×640、batch 32 量级可跑；显存不足优先降 batch，而不是改模型结构。
- 推理延迟随 GPU 共享负载可差 5–6 倍，任何时候报延迟都要同时报测量条件。
