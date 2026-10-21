# AGENTS.md

本文件面向在此仓库工作的 AI 编码代理（也适用于人类协作者）。开始任何改动前先读完本文件与 `README.md`。

## 1. 项目目标

本仓库是一个**教学型仓库**：用中文材料从零讲清 **ACT（Action Chunking with Transformers）**，并给出可复现的实现与仿真基线。

当前核心产物是 `notebooks/ACT_tutorial.ipynb`，覆盖：

- **原理**：行为克隆的误差累积与多模态问题、action chunking、CVAE、Transformer encoder/decoder、时序集成。
- **从零实现**：约 1.4M 参数的 MiniACT（CNN + CVAE + Transformer），在公开数据上训练与评估。
- **官方对齐**：LeRobot 0.6.1 `ACTPolicy` 的配置、forward、预处理/后处理，以及短训练与正式训练命令。
- **仿真验证**：用 `gym_aloha` 在 MuJoCo 里跑完整 episode，给出成功率与 rollout 视频。
- **部署与评估**：动作队列 vs 时序集成、控制循环骨架、推理延迟测量、开环 L1 与闭环成功率、常见排错。

目标读者既是初学者，也包括后续在此仓库继续迭代的人。因此：**结论要可复现，数字要能追溯到命令**。

## 2. 运行环境

| 项 | 值 |
|---|---|
| Python | **3.12**（LeRobot 要求 `>=3.12`；`dm_control → labmaze` 只提供到 `cp312` 的 wheel） |
| 核心依赖 | `lerobot 0.6.x`、PyTorch（CUDA 版）、`gym-aloha`、`matplotlib`、`jupyterlab` |
| 显卡 | 单卡即可：MiniACT 用 64×64 输入；官方 ACT 短训练用 dim_model=256 |

安装：

```bash
conda create -n lerobot python=3.12 -y    # 若已存在 Python 3.13 的 lerobot 环境，请另建（如 lerobot_act）
conda activate lerobot
pip install "lerobot[dataset,training,core-scripts,evaluation,av-dep,matplotlib-dep]"
pip install gym-aloha jupyterlab
```

注意：

- 非交互式 shell 不会自动初始化 conda，需要先 `source "$HOME/miniconda3/etc/profile.d/conda.sh"`。
- 网络受限时给 HuggingFace 配镜像：`export HF_ENDPOINT=https://hf-mirror.com`、`export HF_HUB_DISABLE_XET=1`。
- headless 机器跑 MuJoCo 必须 `export MUJOCO_GL=egl`（没有 GPU 时用 `osmesa`）。

## 3. 目录结构

```
ACTLearning/
├── AGENTS.md
├── README.md
├── .gitignore
├── notebooks/
│   ├── ACT_tutorial.ipynb      # 主教程，含 MiniACT 实现与全部实验
│   ├── ACT_tutorial.html       # 导出物，便于不装环境直接阅读
│   └── assets/                 # 示意图与 rollout 演示（gif/png）
├── data/                       # 数据集缓存（gitignore，可重新下载）
└── outputs/                    # cache / checkpoint / 评测结果（gitignore）
```

约定：

- **notebook 是当前唯一载体**，但可复用的逻辑（模型、数据管线、评估指标）应逐步抽到 `src/act_learning/`，抽出后同步更新本节与 `README.md`。
- `outputs/cache/mini_top_64.pt`（约 300 MB）是可重建缓存，**绝不入库**；重建方式写在 notebook 对应 cell 里。
- `notebooks/.ipynb_checkpoints/` 不提交（已在 `.gitignore`）。
- 演示素材（`notebooks/assets/*.gif`）要控制在 2 MB 以内，避免拖慢 clone。

## 4. 参考实现（只读）

LeRobot 的 ACT 实现随包安装，读源码即可：

- `site-packages/lerobot/policies/act/modeling_act.py` — ACT 网络与损失（CVAE + Transformer + chunk）
- `site-packages/lerobot/policies/act/configuration_act.py` — 超参默认值（`chunk_size`、`n_action_steps`、`dim_model`、`latent_dim` 等）
- `site-packages/lerobot/policies/act/processor_act.py` — 输入归一化与动作反归一化

对拍时**只读**这些文件；需要改动时必须改本仓库内的代码。

## 5. 数据

使用公开数据集 [`lerobot/aloha_sim_insertion_human`](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human)，首次运行自动下载到 `data/aloha_sim_insertion_human`：

- LeRobot dataset v3.0，50 episodes / 25000 帧 / 50 fps
- `observation.state`：14 维（双臂，每臂 6 关节 + 夹爪）
- `action`：14 维绝对关节目标
- 图像：`observation.images.top`，480×640×3

约定：

- 数据与权重**绝不入库**；需要换数据集时用 `ACT_DATASET_ROOT` 指向本地副本，不要复制进仓库。
- 读取统一走 notebook 中的数据管线 cell（后续抽为 `src/act_learning/data/`）；不要在别处零散 `pd.read_parquet`。
- 训练/验证划分必须固定（按 episode 划分），归一化统计量必须落盘，保证评估可复现。

## 6. 常用命令

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate lerobot
cd ACTLearning

# 交互式（浏览器打开 http://localhost:8888/lab，内核选 "Python (lerobot ACT)"）
jupyter lab --ip 127.0.0.1 --port 8888 --no-browser

# 无界面执行全部 cell（会重建缓存并训练，耗时且占 GPU）
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/ACT_tutorial.ipynb

# 重新导出 HTML
jupyter nbconvert --to html notebooks/ACT_tutorial.ipynb

# 仿真评测（旧格式权重先迁移）
python -m lerobot.processor.migrate_policy_normalization \
  --pretrained-path lerobot/act_aloha_sim_insertion_human \
  --output-dir outputs/models/act_aloha_sim_insertion_human
MUJOCO_GL=egl lerobot-eval \
  --policy.path=outputs/models/act_aloha_sim_insertion_human \
  --env.type=aloha --policy.device=cuda \
  --eval.n_episodes=10 --eval.batch_size=1 \
  --output_dir=outputs/sim_eval/pretrained
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
- **不要** `git push --force`，不要擅自改写远端历史；推送前先确认。
- 大文件（`data/`、`outputs/`、`*.pt`、`*.safetensors`）不入库；`.gitignore` 已覆盖主要项，新增产物类型先补规则。

## 9. 代理协作规则

- 动手前：读 `README.md`、`AGENTS.md`，`git status` 确认工作区状态，弄清未提交改动是谁的。
- 只在仓库目录内改动；不要碰同级的其他项目。
- 不要使用 `sudo`；不要改 `~/.bashrc`、`~/.gitconfig` 等全局配置。
- 训练/评估/notebook 执行等长任务：用 `tmux` 或 `nohup`，日志写 `outputs/logs/`，不要在前台阻塞。
- 默认只占 `cuda:0`；确需多卡再显式指定，避免抢卡（延迟数据会因共享负载波动，测量时要注明）。
- 改完至少验证受影响的 cell 或脚本能跑通；无法运行时如实说明未验证项，不要假装通过。
- 结论要有证据：给出命令、日志片段或指标数值，避免「应该没问题」这类判断。

## 10. 已知坑

- 直接 `ssh host 'conda activate ...'` 会失败，原因是非登录 shell 未加载 conda。
- 从 HuggingFace 下载会卡住：需要设 `HF_ENDPOINT` 镜像并禁用 xet。
- `jupyter nbconvert --execute` 默认超时较短，长 cell 需 `--ExecutePreprocessor.timeout=-1`。
- `lerobot-eval` 默认 `use_async_envs=True`，某些环境下建 async vector env 会 `BrokenPipeError`；加 `--eval.batch_size=1`。
- MuJoCo 无显示环境会直接崩：必须 `MUJOCO_GL=egl`（或用 `osmesa`）。
- torchcodec 在部分机器的 FFmpeg 版本下加载失败，LeRobot 会回退到 `pyav`，不影响本仓库的数据读取。
- 官方旧格式 checkpoint 需要 `migrate_policy_normalization` 才能在 0.6.x 里加载；迁移后的归一化统计量应与数据集 `meta/stats.json` 一致。
- 仿真的成功率对 MuJoCo / gym-aloha 版本敏感：同一份权重在不同版本上可能差很多，报结果时必须带环境版本与 episode 数。
- 单卡 8 GB：MiniACT 用 64×64 没问题；官方 ACT 用 dim_model=256、batch 4–8。
