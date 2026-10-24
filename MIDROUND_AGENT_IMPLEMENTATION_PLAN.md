# Mid-round Agent 实现规划

> 对应设计文档：`MIDROUND_AGENT_DESIGN.md`
>
> 本文只做模块拆分、接口、数据契约、测试顺序和里程碑，不包含代码。

---

## 1. 实现目标

第一版交付：

```text
按热键
  -> 100ms 内显示一行可执行指令
  -> 新鲜 LLM 计划优先
  -> 无缓存/超时 -> 内核模板
```

第一版不做：

- 角度级“架”
- 语音输入
- 自动操作
- T 方 / 其他地图
- 买/保、赛后复盘

---

## 2. 模块拆分

### 新增目录

```text
live/midround/
├── __init__.py
├── state_card.py
├── route_graph.py
├── candidate_builder.py
├── plan_cache.py
├── validator.py
├── llm_planner.py
├── agent.py
├── logging.py
└── hotkey.py
```

### 现有模块

| 模块 | 变化 |
|---|---|
| `live/gsi_server.py` | 接入 `agent`，新增 `/plan`、`/agent/hotkey` |
| `live/rotation_policy.py` | 不变 |
| `live/area_lookup.py` | 不变 |
| `live/radar_parser.py` | 不变 |
| `live/radar_capture.py` | 不变 |
| `live/models/` | 新增 `dust2_ct_routes.json` |

### 依赖原则

- 服务端核心继续只用标准库
- LLM 通过 `urllib` 调本地 Ollama，不引入 SDK
- 热键使用可选的 `pynput`，未安装时降级为网页按钮
- TTS 优先用浏览器 `speechSynthesis`，不引入额外依赖

---

## 3. 数据契约

### 3.1 状态卡

见 `MIDROUND_AGENT_DESIGN.md` 第 3 节。

实现要求：

- `state_card.py` 只负责组装和缺失值标注
- 不包含任何决策逻辑
- 输出必须是纯 JSON 可序列化

### 3.2 候选

```json
{
  "id": "safe",
  "action": "ROTATE_B",
  "target_area": "B",
  "route": ["Mid", "MidDoors", "CTSpawn", "B_Doors", "B_Site"],
  "route_style": "safe",
  "eta_s": 9.0,
  "exposure": "low",
  "call_hint": "转 B，走 CT 家，和队友汇合",
  "grounded_facts": ["B 点有 1 名队友", "剩余 38 秒"],
  "expires_s": 3
}
```

### 3.3 计划

```json
{
  "plan_id": "5:77:safe:llm",
  "created_at": 1760000000.0,
  "fingerprint": {...},
  "source": "llm",
  "selected_id": "safe",
  "call": "转 B，走 CT 家，和队友汇合",
  "reason_short": "中路没有信息，B 点有队友",
  "confidence": "medium",
  "grounded_facts": ["B 点有 1 名队友"],
  "validation": "ok",
  "cache_age_s": 0.0,
  "llm_latency_s": 2.1
}
```

---

## 4. 接口

### 4.1 服务端

新增：

```text
GET  /plan
  返回当前缓存计划 + age + source
  无计划时返回模板计划或 null

POST /agent/hotkey
  标记玩家请求了一次即时指令
  页面据此放大显示 / 触发 TTS

POST /agent/refresh
  调试用：强制刷新 LLM 计划

GET  /agent/status
  返回缓存状态、LLM 队列长度、最近错误
```

### 4.2 热键进程

独立脚本：

```text
live/midround/hotkey.py
```

行为：

- 监听全局热键（默认鼠标侧键，可配置）
- 调用 `POST /agent/hotkey`
- 不注入游戏输入
- 未安装 `pynput` 时打印提示并退出

### 4.3 页面

每 250ms 轮询 `/plan`：

- 热键触发后，放大显示 call
- 3 秒后变灰
- 可选 `speechSynthesis` 朗读

---

## 5. 模块职责

### 5.1 `state_card.py`

输入：

- GSI payload
- radar payload
- round history
- `AreaLookup`
- rotation result

输出：

- 状态卡 dict
- 缺失字段和来源标注

测试：

- 有/无雷达
- 有/无 GSI 位置
- 炸弹状态
- 队友/敌人缺失
- 回合切换

### 5.2 `route_graph.py`

输入：

- `live/models/dust2_ct_routes.json`

职责：

- 加载节点/边
- Dijkstra 最短路径
- 动态权重：
  - `eta_s`
  - `exposure`
  - 敌人可见区域惩罚
  - 队友支援奖励
- 返回 `safe` / `fast` 两条路线

测试：

- 节点合法性
- 路径可达性
- 时间约束
- 敌人区域避让
- 无路线时返回空

### 5.3 `candidate_builder.py`

输入：

- 状态卡
- 模型动作
- 路线图

职责：

- 生成 2–4 条合法候选
- 排序：内核首选在前
- 去重
- 每条候选附 `grounded_facts`
- 不合法候选直接丢弃

测试：

- HOLD 场景
- ROTATE 场景
- 最后一人
- 残血
- 时间不足
- 无敌人信息
- 炸弹已下

### 5.4 `plan_cache.py`

职责：

- 计算 fingerprint
- TTL 2 秒
- 指纹变化失效
- 线程安全读写

测试：

- 新鲜命中
- 过期未命中
- 指纹变化失效
- 并发读写

### 5.5 `validator.py`

职责：

- 校验 LLM 输出 JSON
- `selected_id` 存在
- 长度限制
- 事实子集
- 禁用措辞
- 返回 `ok` / `fallback`

测试：

- 合法输出
- 不存在的 id
- 超长
- 编造事实
- 敌人意图措辞

### 5.6 `llm_planner.py`

职责：

- 构造 prompt
- 调 Ollama `/api/chat`
- `think: false`、`format: json`
- 5 秒超时
- 最多 1 次重试
- 调用 validator
- 失败返回 None

测试：

- mock Ollama 正常返回
- 非法 JSON
- 超时
- 校验失败
- 限流

### 5.7 `agent.py`

职责：

- 快循环：
  - 状态卡
  - 模型结果
  - 候选
  - 缓存查询
- 慢循环：
  - 事件触发
  - LLM 调用
  - 写缓存
- 热键请求：
  - 返回最新计划
  - 绝不等待 LLM
- 线程安全
- 记录 plan/outcome

测试：

- 模板快速返回
- LLM 慢时不影响热键
- 指纹变化
- 事件触发
- 回退

### 5.8 `logging.py`

新增记录：

```text
record_type = "plan"
record_type = "plan_outcome"
```

字段：

- state_card
- candidates
- selected plan
- source
- cache_age
- llm_latency
- validation
- actual action at +3s / +5s
- adopted

---

## 6. 路线图数据

文件：

```text
live/models/dust2_ct_routes.json
```

格式：

```json
{
  "version": 1,
  "map": "de_dust2",
  "nodes": {
    "CTSpawn": {"callout": "警家"},
    "A_Site": {"callout": "A 包点"},
    "Mid": {"callout": "中路"},
    "B_Site": {"callout": "B 包点"}
  },
  "edges": [
    {"from": "CTSpawn", "to": "A_Site", "eta_s": 6.0, "exposure": "low"},
    {"from": "CTSpawn", "to": "B_Doors", "eta_s": 4.0, "exposure": "low"},
    {"from": "Mid", "to": "MidDoors", "eta_s": 3.0, "exposure": "high"}
  ]
}
```

要求：

- 每条边 `bidirectional` 默认 true
- `exposure` 只用 `low/medium/high`
- 第一版手工维护
- 后续用 demo 轨迹校准 ETA 和常见路线

---

## 7. LLM 接入

### 7.1 运行方式

- 本地 Ollama
- 模型 `qwen3:8b`
- 原生 `/api/chat`
- `stream: false`
- `think: false`
- `format: json`

### 7.2 Prompt 约束

system：

```text
你是 CS2 Dust2 CT 中期指挥。
只能从候选里选择一条。
只能使用状态卡和候选中的事实。
不能声称敌人一定在某处。
输出 JSON。
call 不超过 20 个汉字。
```

### 7.3 失败处理

| 情况 | 处理 |
|---|---|
| 连接失败 | 回退模板 |
| 超时 5s | 回退模板 |
| 非法 JSON | 重试 1 次，再失败回退 |
| 校验失败 | 回退模板，记录原因 |
| 限流 | 跳过本次，保留旧缓存 |

---

## 8. 测试顺序

### 阶段 1：纯离线、无 LLM

1. `state_card` 单测
2. `route_graph` 单测
3. `candidate_builder` 单测
4. `plan_cache` 单测
5. 用两张雷达截图 + 模拟 GSI 构造状态卡
6. 验证候选合法性和路线

### 阶段 2：模板计划上线

1. `agent` 快循环
2. `/plan` 接口
3. 页面显示
4. 热键进程
5. shadow 记录 plan

验收：

- 热键 p95 < 100ms
- 无 LLM 也能工作
- 指令 ≤ 20 字

### 阶段 3：LLM 选择

1. mock Ollama 测试
2. 真实 Ollama 本地测试
3. 校验器
4. 超时/失败回退
5. 缓存命中
6. plan/outcome 日志

验收：

- LLM 失败 100% 回退
- 校验失败有记录
- 热键不受 LLM 延迟影响

### 阶段 4：真实对局 shadow

1. 连续 5 局
2. 人工抽检 20 条指令
3. 计算采纳率、误指挥率
4. 对比 LLM 选择与内核首选

---

## 9. 里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M1 | 状态卡 + 路线图 + 候选生成 | 单测通过，候选合法 |
| M2 | 缓存 + agent 快循环 + `/plan` | 模板计划可用，热键 <100ms |
| M3 | LLM 选择 + 校验 + 回退 | LLM 故障不影响可用性 |
| M4 | 热键 + 页面显示 + plan 日志 | 真实对局可以按出指令 |
| M5 | 5 局 shadow + 人工评估 | 采纳率和可执行性达标 |

---

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| 路线图不准 | 先用少量手工边，后用 demo 轨迹校准 |
| 热键与游戏冲突 | 可配置键位，先支持鼠标侧键 |
| 反作弊/合规 | 只监听热键，不注入、不读内存 |
| LLM 太慢 | 异步 + 缓存 + 模板回退 |
| LLM 编造事实 | validator + 禁用措辞 + 事实子集 |
| 感知缺失 | 状态卡显式标注默认值，LLM 不得当精确事实 |
| 指令过长 | 硬限制 20 字，模板兜底 |
| 页面看不到 | 第二屏 + 可选 TTS |

---

## 11. 下一步

确认本规划后，按 M1 开始：

1. 建 `live/midround/` 包
2. 写 `state_card.py`
3. 写 `dust2_ct_routes.json`
4. 写 `route_graph.py`
5. 写 `candidate_builder.py`
6. 配套单测

M1 不接触 LLM、不接触热键，先把“状态 -> 合法候选”跑通。

### M1 状态（2026-10-23）

已完成：

- `live/midround/state_card.py`：GSI + 雷达 -> 状态卡，含缺失值标注
- `live/models/dust2_ct_routes.json`：14 个节点、18 条边的手工拓扑
- `live/midround/route_graph.py`：safe / fast 路线、暴露度与敌人区域惩罚、时间约束
- `live/midround/candidate_builder.py`：2–4 条候选、合法性校验、grounded facts
- 单测：`test_state_card.py`、`test_route_graph.py`、`test_candidate_builder.py`，以及用真实雷达截图跑通的 `test_midround_integration.py`

验收：`python -m unittest discover -s live/tests -t .` 共 64 项全部通过。

### M2 状态（2026-10-23）

已完成：

- `live/midround/plan_cache.py`：状态指纹、2 秒新鲜期、4 秒上限、线程安全
- `live/midround/agent.py`：快循环（状态卡 -> 候选 -> 模板计划）、热键路径只读缓存、过期后立即用模板重建
- `live/midround/logging.py`：`record_type = "plan"` / `"plan_outcome"` 追加日志
- `live/midround/hotkey.py`：可选 `pynput` 全局鼠标侧键，只监听不注入
- `live/gsi_server.py`：新增 `GET /plan`、`POST /agent/hotkey`、`POST /agent/refresh`、`GET /agent/status`
- 页面：250ms 轮询 `/plan`，大字号 call、3 秒后变灰、可选 TTS 与网页按钮

验收：

- 热键路径 p95 < 1ms（本机测量，远低于 100ms 目标）
- 无 LLM 时模板计划可用，call ≤ 20 字
- 91 项 live 测试全部通过

下一步进入 M3：`llm_planner.py` + `validator.py`，LLM 只做选择与措辞，失败 100% 回退模板。

### M3 状态（2026-10-23）

已完成：

- `live/midround/validator.py`：候选 id、20 字 call、30 字理由、grounded facts 子集、未知区域和禁用措辞校验
- `live/midround/llm_planner.py`：
  - 默认 Ollama `/api/chat`（`qwen3:8b`、`think:false`、`format:json`、`stream:false`）
  - OpenAI 兼容后端（`/v1/chat/completions`，适配 LM Studio / vLLM / llama.cpp 等）
  - 5 秒超时、非法 JSON 重试 1 次、连接失败直接回退
- `live/midround/agent.py`：后台慢循环，热键路径不等待；状态变化则丢弃迟到的 LLM 结果
- 服务端：`GET/POST /agent/config`，页面新增“模型接入”面板（后端/模型/地址/API Key，运行时生效）
- 实测本地 `qwen3:8b`：冷启动约 28 秒（首次加载），预热后约 3 秒，5 秒超时下可正常返回

验收：

- LLM 失败 100% 保留模板计划
- 校验失败、非法 JSON、超时均有记录（`/agent/status` 与 shadow log）
- 热键路径 p95 仍远低于 100ms
- 133 项 live 测试全部通过（LLM 测试使用 fake backend）

下一步进入 M4：热键 + 页面显示已具备，补齐 plan_outcome 回填与真实对局 shadow 评估。

### M4 状态（2026-10-23）

已完成：

- 热键触发后登记 pending outcome，在 +3 秒 / +5 秒分别回填
  - `status = ok / died / round_changed / not_live`
  - `adopted`：玩家所处区域是否等于计划目标区域
  - `actual_action`：`HOLD` / `ROTATE_A/B/MID/CT` / `OTHER`
- 同一 plan 重复按热键只记一次样本，队列上限 50
- 每次 LLM 调用写 `record_type="plan_llm"`，记录成功/失败、校验原因、延迟、重试次数
- `live/analyze_shadow.py` 新增计划报告：
  - plan 来源分布、call 长度、LLM 延迟 p50/p95
  - LLM 成功率、错误分布、校验分布
  - 3 秒 / 5 秒采纳率，并按 `llm` / `template` 拆分
- `/agent/status` 增加 pending、已记录、已采纳与采纳率

验收：

- 3 秒 / 5 秒两条 `plan_outcome` 均按预期写入
- 死亡与换回合会正确关闭待回填记录
- 157 项 live 测试全部通过（新增键盘热键、单屏 ticker 与 no_position 统计口径）

下一步进入 M5：连续 5 局真实对局 shadow，人工抽检 20 条指令，统计采纳率、误指挥率，并对比 LLM 与内核首选。
