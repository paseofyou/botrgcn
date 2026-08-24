# 项目说明 (给协作者 / AI agent)

社交机器人检测。基线 BotRGCN，本文方法在其上加时序编码器 + 门控融合。
两个数据集：TwiBot-20（本地）、TwiBot-22（autodl 远程）。

## 环境

本地 Windows 用 conda 环境 `torch_pyg`（torch 2.5.1 + PyG 2.6.1 + CUDA 可用）：

```
C:\Users\20996\.conda\envs\torch_pyg\python.exe
```

注意：`python` 不在 PATH 上（只有 Windows Store 的占位 stub），必须用绝对路径。
`botrgcn` 环境的 numpy 已损坏且缺 torch_geometric，不要用。

## 目录约定

```
Twibot_20/
  src/                      # 代码（train.py, model.py, feature_extraction.py）
  saved_data/processed_data/# 图张量 (.pt)
  feature_model_outputs/    # 时序嵌入 (.npz)
  tmp/tmp_v6/               # 特征提取中间产物, 含各 split 的 *_static.csv
  checkpoints/  swanlog/
Twibot_22/src/              # 同上, 但运行在 autodl, 路径由 --work-dir 指定
experiments/
  results.csv               # 所有运行的最终测试指标, 受版本控制
  aggregate.py              # 汇总为 mean±std 表格 (markdown / latex)
  run_twibot20.py           # 批量实验运行器
```

**代码在 `src/` 下，数据在 `src/` 的上一级。** train.py 里用 `BASE_DIR = os.path.dirname(SCRIPT_DIR)`，
不要用 `os.path.dirname(__file__)` 直接拼数据路径。

## TwiBot-20 数据的坑（已核验，别再踩）

- 图共 **229,580** 个节点，但 `label.pt` 只有 **11,826** 条 —— 前 11,826 个节点是有标注用户，
  其余 217,754 个是 support 用户（无标签）。
- 节点顺序为 **train → dev → test → support**，与 `tmp/tmp_v6/*_static.csv` 的拼接顺序、
  以及 `feature_model_outputs/*.npz` 里 `user_ids` 的顺序 **完全一致**（已逐条比对）。
- 官方划分 = 前三段连续区间：**train[0:8278], dev[8278:10643], test[10643:11826]**。
- `saved_data/processed_data/` 下的 `train_idx.pt / val_idx.pt / test_idx.pt` 是早期的
  **随机划分**（8278/1182/2366，索引不连续），**与官方划分不同，一律不要使用**。
- 官方划分 JSON（`Data/Twibot-20/{train,dev,test,support}.json`）**当前不在仓库里**。
  train.py 会自动回退到 `tmp/tmp_v6/*_static.csv` 来确定用户顺序和划分大小。

## 实验协议（论文要求，不要偏离）

- 5 个种子：`42 123 456 789 2024`；所有变体共用同一套超参
- 按 val accuracy 选 checkpoint，只在最后对 test 评估一次
- 报 Accuracy / Precision / Recall / F1 / MCC 的 mean±std
- 每次运行自动追加一行到 `experiments/results.csv`，含 git commit。
  **commit 带 `-dirty` 的结果不可复现，不能写进论文**，aggregate.py 默认排除。

## 常用命令

```powershell
$py = "C:\Users\20996\.conda\envs\torch_pyg\python.exe"

# 单次训练（在 Twibot_20/src 下运行）
& $py train.py --model pure --variant baseline --seed 42 --epochs 150
& $py train.py --model v2 --variant full --ts-mode anchor --seed 42 --epochs 150

# 批量
& $py experiments/run_twibot20.py stage1            # 表1 主对比
& $py experiments/run_twibot20.py stage2            # 表2 消融
& $py experiments/run_twibot20.py stage4            # 图1 T 敏感性
& $py experiments/run_twibot20.py stage1 --dry-run  # 先看命令

# 汇总
& $py experiments/aggregate.py --dataset TwiBot-20
& $py experiments/aggregate.py --format latex --min-seeds 5
& $py experiments/aggregate.py --ttest baseline full --ttest-metric f1
```

### AutoDL / TwiBot-22

```bash
WORK=/root/autodl-tmp/twibot22
DATA=$WORK/data

# 首次：预处理 + 三种时序特征（real / pseudo / flat）
bash experiments/setup_autodl.sh $WORK $DATA

# 表1 主对比 + 表2 消融（real-TS, 5 seeds）
python experiments/run_twibot22.py stage1 --work-dir $WORK --ts real --seeds 42 123 456 789 2024
python experiments/run_twibot22.py stage2 --work-dir $WORK --ts real --seeds 42 123 456 789 2024

# 汇总
python experiments/aggregate.py --dataset TwiBot-22
python experiments/aggregate.py --dataset TwiBot-22 --ttest baseline full --ttest-metric f1
```

AutoDL 目录结构：

```
$WORK/
  code/                   # 本仓库
  data/                   # 原始数据: user.json, split.csv, label.csv, edge.csv, tweet_*.json
  saved_data/twibot22_data/processed_data/   # 预处理产物
  feature_model_outputs/  # 时序嵌入
  tmp_twibot22_real/      # 真实时序矩阵
```

**重要**：`train_twibot22.py` 使用 `--work-dir $WORK`，所有产物都挂在 `$WORK` 下；
`preprocess_twibot22.py` 需要 `--save-dir $WORK/saved_data/twibot22_data`。

## 模型与消融开关

| `--model` | 说明 |
|---|---|
| `pure` | `BotRGCN_Pure`，官方 BotRGCN 复现，无任何时序通路 |
| `v1` | 早期改动版（LayerNorm + late fusion 拼接时序），**不是**官方基线 |
| `v2` | 本文主模型：RGCN + 时序投影 + 门控融合 |
| `v3` | GAT + 残差，仅作 extra 行 |

`--no-temporal` → `fusion_type='none'`（Graph-only）
`--no-graph` → `fusion_type='no_graph'`（Time-only）
`--concat-fusion` → `fusion_type='concat'`
默认 → `fusion_type='gated'`

## 已知待办 / 风险

1. **TwiBot-20 的时序是合成伪序列**（`feature_extraction.py` 的 `pseudo`/`anchor` 从静态特征生成，
   `anchor` 仅用真实 `account_age_days` 控制曲线拐点），不含静态特征之外的新信息。
   论文主结论应建立在 TwiBot-22 的真实时间戳（`feature_extraction_twibot22_real.py`）上；
   `flat_static` 消融是验证"提升是否真的来自时序建模"的关键对照，必须跑。
2. **两个数据集的 `BotRGCN_v2` 架构目前不一致**：TwiBot-20 版用 LeakyReLU + 单个共享 RGCN 层；
   TwiBot-22 版用 GELU + 两个独立 RGCN 层。论文里"Ours"必须是同一个模型，发论文前需要统一。
