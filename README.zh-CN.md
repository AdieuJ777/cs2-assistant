# CS2 决策助手（Dust2 / CT）

一个面向 **CS2 单人排位、Dust2、CT 方**的个人局内决策助手原型。

目标不是做“全自动 AI”，而是把可验证的统计模型、官方 GSI 和本地雷达感知组合起来，在冻结期给出**买/保**建议，在回合中期给出**转/守**建议，并通过 shadow 日志评估建议质量。

> 当前状态：可运行的研究原型。买/保已经可用；转点建议能跑通真实对局，但仍是研究级，不应被视为确定正确的指令。

---

## 现在能做什么

| 功能 | 状态 | 说明 |
|---|---|---|
| 买 / 保 | ✅ 可用 | 冻结期购买建议、购物清单、余额、敌方经济先验 |
| 雷达感知 | ✅ 可用 | 屏幕采集雷达，识别敌人红点、队友点，并用 A/B 标签反解玩家位置 |
| 转 / 守 | 🟡 研究级 | 5 秒两阶段模型，输出 `HOLD` / `ROTATE_A/B/MID/CT` |
| 架 | 🟡 仅有姿态 | 输出 `HOLD_ANGLE`，尚未建模具体架枪角度 |
| Shadow 评估 | ✅ 可用 | 每条建议 5 秒后回填玩家实际动作，可算一致率、误转/漏转 |
| LLM | ⬜ 不参与决策 | 实验结论：LLM 适合解释/复盘，不适合数值推断或动作选择 |

---

## 快速开始

### 前置条件

- Windows + CS2
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Dust2、CT 方
- 雷达保持默认左上角；`cl_radar_always_centered 1`

### 1. 安装 GSI 配置（一次性）

把仓库里的配置复制到 CS2 的 `game/cs2/cfg/`：

```powershell
copy "C:\Users\22896\Desktop\TAGNet-CS2-main\cs2-assistant\live\gsi_config\gamestate_integration_cs2assistant.cfg" "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\cs2\cfg\"
```

配置里的 token 默认是 `change-me`。改完配置后**完全重启 CS2**。

> 如果使用自定义 token，必须同时修改三处：安装后的 GSI cfg、服务端 `--token`、采集端 `--token`。三者不一致会返回 403 或页面空白。

### 2. 启动服务端

```powershell
cd C:\Users\22896\Desktop\TAGNet-CS2-main\cs2-assistant
python live/gsi_server.py --verbose
```

启动后会显示：

```text
Rotation model loaded: ...\live\models\rotation_policy.json
GSI listener: http://127.0.0.1:3000
```

### 3. 启动雷达采集（第二个终端）

```powershell
cd C:\Users\22896\Desktop\TAGNet-CS2-main\cs2-assistant
uv run --with mss --with pillow --with numpy --with scipy python live/radar_capture.py
```

第一次会自动下载 `mss`、`pillow`、`numpy`、`scipy`。

正常输出：

```text
self=(80,1372) source=radar_labels enemies=1 teammates=2
```

### 4. 打开页面

```text
http://127.0.0.1:3000
```

冻结期显示买/保建议；live 回合雷达上有敌人时显示 `保持站位` / `转 A/B/中路/回 CT`。

### 分辨率与标定

已提供两套雷达标定：

- 1280×960
- 2560×1440

采集端会根据屏幕分辨率自动选择。**全屏窗口模式使用桌面分辨率**，不是游戏内设置的分辨率。改分辨率或 HUD scale 后需要重新标定：

```powershell
uv run --with pillow --with numpy --with scipy python live/radar_calibrate.py --screenshot <截图> --output live/models/radar_config_<宽>x<高>.json
```

> CS2 官方 GSI 在正常游玩时**不提供玩家位置**（只有观战模式提供）。本项目用雷达图上的 A/B 包点标签每帧反解玩家位置，因此不需要 `player_position`。

---

## 目录结构

```text
cs2-assistant/
├── benchmarks/
│   ├── economy_inference/       # 敌方经济推断基准
│   ├── action_selection/        # 动作选择基准
│   └── enemy_visibility/        # 固定快照 + 事件驱动雷达数据集
├── live/                        # 局内助手
│   ├── gsi_server.py            # GSI 监听 + /radar + 中文网页 + shadow 日志
│   ├── advisor.py               # 买/保规则
│   ├── rotation_policy.py       # 5 秒两阶段模型的纯 Python 推理
│   ├── radar_calibrate.py       # 雷达截图标定
│   ├── radar_parser.py          # 红点/队友点检测 + A/B 标签定位
│   ├── radar_capture.py         # mss 屏幕采集 -> /radar
│   ├── build_area_grid.py       # 世界坐标 -> 区域
│   ├── analyze_shadow.py        # shadow 日志分析
│   ├── compare_live_features.py # live vs 训练特征分布对比
│   └── models/                  # 模型、标定、区域网格
├── data/                        # 生成的基准数据
├── results/                     # 实验指标和总结
├── README.md                    # 英文详细说明
├── PROJECT_SUMMARY.md           # 项目阶段与结论
└── README.zh-CN.md              # 本文件
```

---

## 关键设计

### 1. 只用官方 GSI + 屏幕读取

- 不读游戏内存
- 不注入输入
- 不修改游戏
- 雷达采集只截取屏幕左上角的雷达区域

### 2. 统计模型负责推断，LLM 负责解释

多次实验表明：

- LLM 做敌方经济推断：35%–67%，不如决策树（74.4%）
- LLM 做动作选择：37%，随机森林 82%，且 LLM 从不选择 HOLD
- “模型提议、LLM 解释”可行，但解释层还需要更强的 grounding

因此当前架构是：

```text
GSI + 雷达 -> 统计模型 -> 建议
                     -> LLM（可选，解释/复盘，不改变建议）
```

### 3. 事件驱动，而不是固定时间快照

固定快照下雷达敌人信息几乎全为 0。项目改为在“敌人首次出现在雷达”时触发决策：

- 5 秒预测窗口
- 1,099 个事件 / 3,792 条决策（9 场 demo）
- 用置换对照验证提升确实来自敌人信息

### 4. Shadow 评估

服务端把每条建议和 5 秒后的实际动作都写进 `live/shadow_log.jsonl`：

- `advice`：建议动作、置信度、转点概率、完整特征
- `outcome`：玩家实际动作、是否匹配、状态（`ok` / `died` / `no_position` / `round_changed`）

分析：

```powershell
python live/analyze_shadow.py live/shadow_log.jsonl
```

---

## 关键结果（简要）

### 敌方经济推断（664 回合）

| 方法 | Accuracy |
|---|---:|
| 多数类 | 47.4% |
| 手写规则 | 55.1% |
| 决策树 depth 6 | **74.4%** |
| LLM zero-shot | 35.0% |

### 动作选择（13,560 个决策）

- 平衡样本随机森林：Top-1 **82%**，Top-2 **93%**
- 自然分布下会退化成只会 HOLD
- 两阶段策略：目的地预测 **92.1%**，瓶颈是“该不该转”

### 事件驱动转点（5 秒窗口）

| 指标 | base | + 敌人信息 |
|---|---:|---:|
| Stage-1 AP | 0.478 | **0.547** |
| Stage-2 准确率 | 0.740 | **0.848** |
| @0.6 精确率 / 召回 | 0.487 / 0.626 | **0.544 / 0.664** |

置换对照：真实敌人信息 AP 0.547，打乱后 0.464/0.473，随机噪声 0.479。

---

## 当前限制

- 只支持 Dust2、CT 方
- 转点模型是研究级：@0.6 仍有约 16% 误转、34% 漏转
- 队友血量、敌人存活数尚未从 HUD 读取，使用默认值并告警
- 雷达标定依赖固定 HUD 布局；改分辨率/HUD scale 需重新标定
- CS2 实战 GSI 不提供玩家坐标，定位依赖雷达 A/B 标签；标签被遮挡时会显示“无法定位”
- 尚未证明胜率提升
- 不是游戏内 overlay，而是第二屏网页

---

## 测试

```powershell
python -m unittest discover -s live/tests -t .
```

- 30 项 live 测试
- 其中 8 项雷达解析测试需要 `pillow / numpy / scipy`，在纯标准库运行时会跳过

单独运行雷达测试：

```powershell
uv run --with pillow --with numpy --with scipy python -m unittest live.tests.test_radar_parser -v
```

---

## 更多文档

- [README.md](README.md)：英文详细说明
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)：阶段、结论、下一步
- [live/README.md](live/README.md)：局内助手的安装、雷达契约和限制
- [benchmarks/enemy_visibility/README.md](benchmarks/enemy_visibility/README.md)：事件驱动雷达实验
- [results/enemy_visibility/summary.md](results/enemy_visibility/summary.md)：实验结果汇总

---

## 免责声明

这是一个个人研究项目，不保证建议正确，也不保证提升胜率。请只在允许的范围内使用官方 GSI 和屏幕读取；不要使用任何注入、内存读取或自动操作功能。
