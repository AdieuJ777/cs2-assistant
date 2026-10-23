# Mid-round Agent 设计

> 目标：玩家按下热键，在 100ms 内得到一条**简短、可执行**的中期指令。
>
> 架构：**快反射 + 慢思考**。内核负责即时反应，LLM 在后台从候选里选择并组织语言，缓存结果供热键读取。

---

## 1. 范围

### 做

- Dust2、CT 方
- 中期（phase=live）
- 宏观动作 + 路线：`守当前区域` / `转 A/B/Mid/CT` / `退回 CT` / `等`
- 热键触发
- 第二屏一行字
- 长按查看一句理由（可选）

### 不做

- 角度级“架哪个点”
- 语音输入
- 自动操作/注入
- T 方、其他地图
- 买/保、赛后复盘（后续阶段）

---

## 2. 总体架构

```text
每 100ms（快循环）
  GSI + 雷达
    -> 状态卡
    -> 内核模型（两阶段 RF）
    -> 候选生成（2-4 条合法指令）
    -> 缓存查找
    -> 热键返回（<100ms）

关键事件 / 每 2s（慢循环）
  状态卡 + 候选
    -> LLM 选择 + 一句话措辞
    -> 校验器
    -> 写入缓存

热键
  新鲜缓存 -> 直接显示
  过期/无缓存 -> 立即返回内核首选（模板）
```

关键原则：

- LLM **只做选择与措辞**，不新造动作/路线/事实
- 热键路径**永远不等待 LLM**
- 所有输出都可追溯到候选和状态卡

---

## 3. 状态卡

状态卡是快循环和慢循环的唯一输入。

```json
{
  "schema_version": 1,
  "ts": 1760000000.0,
  "round": {
    "number": 5,
    "phase": "live",
    "time_left": 38.0,
    "seconds_since_freeze": 77.0,
    "score_ct": 5,
    "score_t": 3,
    "pistol": false
  },
  "self": {
    "area": "Mid",
    "x": -120.0,
    "y": 836.0,
    "health": 100,
    "armor": 100,
    "equip_value": 4200,
    "weapon": "M4A1-S",
    "defuse_kit": true,
    "utility": {
      "smoke": 1,
      "flash": 1,
      "he": 0,
      "molotov": 0
    },
    "alive": true
  },
  "team": {
    "alive_count": 3,
    "members": [
      {"area": "B", "health": 80, "alive": true},
      {"area": "A", "health": 45, "alive": true}
    ],
    "health_source": "default_100"
  },
  "enemies": {
    "alive_count": 5,
    "alive_count_source": "default_5",
    "visible": [
      {"area": "A", "distance": 1256.0, "bearing": -12.0, "age_s": 0.4}
    ],
    "visible_count": 1,
    "unknown_count": 4,
    "nearest_area": "A",
    "nearest_distance": 1256.0
  },
  "bomb": {
    "state": "carried",
    "site": null,
    "timer": null
  },
  "model": {
    "action": "ROTATE_B",
    "stage1_proba": 0.64,
    "destination": "ROTATE_B",
    "destination_proba": 0.72,
    "threshold": 0.6
  },
  "recent_events": [
    {"type": "spot", "area": "A", "t": -0.4},
    {"type": "teammate_death", "area": "Mid", "t": -8.0}
  ],
  "perception": {
    "radar_age_s": 0.1,
    "self_source": "radar_labels",
    "teammates_source": "radar",
    "warnings": []
  }
}
```

### 字段来源

| 字段 | 来源 |
|---|---|
| 自己区域/血量/护甲/装备/道具 | GSI |
| 自己世界坐标 | 雷达 A/B 标签反解 |
| 队友区域/血量 | 雷达（血量暂缺，默认 100） |
| 敌人可见位置 | 雷达红点 |
| 敌人存活数 | 暂缺，默认 5 |
| 炸弹状态 | GSI |
| 模型输出 | 两阶段随机森林 |

### 缺失值处理

缺失字段必须显式标注：

- `alive_count_source = "default_5"`
- `health_source = "default_100"`
- `warnings` 里说明

LLM 不得把缺失值当成精确事实。

---

## 4. 地图拓扑

### 4.1 节点

Dust2 CT 侧第一版节点：

```text
CTSpawn        警家
A_Site         A 包点
A_Short        A 小
A_Long         A 大
A_Ramp         A 坡
Catwalk        猫道
Pit            大坑
Mid            中路
TopMid         中路上
MidDoors       中门
B_Site         B 包点
B_Doors        B 门
B_Tunnels      B 洞
OutsideTunnel  B 洞外
```

### 4.2 边

边的结构：

```json
{
  "from": "Mid",
  "to": "MidDoors",
  "eta_s": 3.0,
  "exposure": "high",
  "bidirectional": true
}
```

- `eta_s`：常规移动时间
- `exposure`：`low` / `medium` / `high`
- 第一版手工维护
- 后续用 demo 轨迹校准（已有 1,500 万条位置样本）

### 4.3 路径评分

```text
cost = eta_s
     + exposure_penalty
     + enemy_risk_penalty
     - teammate_support_bonus
```

- 已知敌人所在区域 → 经过该区域/相邻区域加风险
- 路线经过队友 → 减分
- 时间不足 `eta + 2s` → 该路线不可用

### 4.4 路线类型

每个目的地区域最多生成两条路线：

- `safe`：低暴露，优先 CT 家/队友附近
- `fast`：时间最短，接受更高暴露

---

## 5. 候选生成

### 5.1 候选类型

| 类型 | 触发条件 | 示例 |
|---|---|---|
| `HOLD` | 模型 HOLD | 守 A 大坑，别推 |
| `ROTATE_SAFE` | 模型 ROTATE | 转 B，走 CT 家 |
| `ROTATE_FAST` | 模型 ROTATE | 转 B，走 B 门 |
| `FALL_BACK` | 残血/少打多/时间不足 | 退回 CT，保枪 |
| `REGROUP` | 队友在另一区域且模型不确定 | 去 B 和队友汇合 |
| `WAIT` | 无敌人信息/模型概率接近阈值 | 先别动，等雷达 |

### 5.2 数量

第一版固定 **2–4 条**：

1. 内核首选（模型动作 + safe 路线）
2. 同动作备选（fast 路线）
3. 保守选项（HOLD / FALL_BACK / WAIT）
4. （可选）Top-2 目的地的 safe 路线

### 5.3 候选 schema

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
  "grounded_facts": [
    "B 点有 1 名队友",
    "中路没有可见敌人",
    "剩余 38 秒"
  ],
  "expires_s": 3
}
```

### 5.4 合法性约束

- `action` 必须在允许集合内
- `target_area` 必须在 `A/B/Mid/CT`
- `route` 必须是图上的合法路径
- `eta_s` 必须小于剩余时间 - 2 秒
- 不能包含未知节点
- 不能引用状态卡里不存在的事实

任何一条不满足 → 候选作废。

---

## 6. LLM 选择

### 6.1 输入

```text
system:
  你是 CS2 Dust2 CT 中期指挥。
  只能从候选里选一条。
  只能使用状态卡和候选里的事实。
  不能声称敌人一定在某处。
  输出 JSON，call <= 20 个汉字。

user:
  状态卡 JSON
  候选数组 JSON
```

### 6.2 输出 schema

```json
{
  "selected_id": "safe",
  "call": "转 B，走 CT 家，和队友汇合",
  "reason_short": "中路没有信息，B 点有队友",
  "confidence": "medium",
  "grounded_facts": [
    "B 点有 1 名队友",
    "中路没有可见敌人"
  ]
}
```

### 6.3 校验器

必须全部通过：

1. `selected_id` 存在
2. `call` 长度 ≤ 20 个汉字
3. `call` 不包含候选之外的区域/动作
4. `grounded_facts` 是状态卡/候选事实的子集
5. 不包含禁用措辞：
   - “敌人一定在……”
   - “肯定……”
   - “百分之……”
6. `reason_short` 长度 ≤ 30 个汉字

失败 → 回退到候选第一条，`source=fallback`。

---

## 7. 缓存与新鲜度

### 7.1 指纹

```json
{
  "round": 5,
  "time_bucket": 7,
  "self_area": "Mid",
  "visible_enemies": ["A"],
  "teammates": ["A:45", "B:80"],
  "bomb": "carried",
  "model_action": "ROTATE_B"
}
```

- `time_bucket = floor(seconds_since_freeze / 5)`
- 任意字段变化 → 缓存失效

### 7.2 生命周期

| 年龄 | 行为 |
|---|---|
| 0–2 秒 | 直接返回缓存 |
| 2–4 秒 | 返回模板；后台刷新 |
| >4 秒 | 丢弃缓存 |
| 指纹变化 | 立即失效 |

### 7.3 缓存条目

```json
{
  "created_at": 1760000000.0,
  "fingerprint": {...},
  "source": "llm",
  "selected_id": "safe",
  "call": "转 B，走 CT 家，和队友汇合",
  "reason_short": "中路没有信息，B 点有队友",
  "confidence": "medium",
  "grounded_facts": [...],
  "validation": "ok"
}
```

---

## 8. 触发策略

LLM 只在以下情况调用：

- 新的敌人可见（`newly_visible_count > 0`）
- 可见敌人集合变化
- 队友死亡 / 队友数量变化
- 炸弹下包 / 掉落
- 回合阶段变化
- 模型动作变化
- 距上次调用 ≥ 2 秒（兜底）

限流：

- 最少间隔 1.5 秒
- 同一指纹不重复调用
- 超时 5 秒
- 失败不重试超过 1 次

---

## 9. 热键与显示

### 触发

- 鼠标侧键或键盘键
- 全局热键监听，不注入游戏

### 显示

- 第二屏网页一行大字
- 长按热键：显示 `reason_short`
- 3 秒后变灰
- 无缓存：立即显示模板

### 状态

| 状态 | 显示 |
|---|---|
| 新鲜 LLM 计划 | 正常 |
| 新鲜内核模板 | 正常，标记“快速建议” |
| 过期 | 灰色 |
| 无雷达/无法定位 | 先别动，等信息 |

---

## 10. 日志与评估

### 10.1 计划日志

```json
{
  "record_type": "plan",
  "ts": "...",
  "round": 5,
  "fingerprint": {...},
  "source": "llm",
  "selected_id": "safe",
  "call": "转 B，走 CT 家，和队友汇合",
  "candidates": [...],
  "validation": "ok",
  "cache_age_s": 0.8,
  "llm_latency_s": 2.1,
  "state_card": {...}
}
```

### 10.2 结果回填

按热键后 3 秒 / 5 秒记录：

```json
{
  "record_type": "plan_outcome",
  "plan_id": "...",
  "horizon_s": 3,
  "actual_action": "ROTATE_B",
  "actual_area": "B",
  "adopted": true,
  "status": "ok"
}
```

### 10.3 指标

- 热键响应延迟 p50 / p95
- LLM 延迟 p50 / p95
- 缓存命中率
- 模板回退率
- 校验失败率
- 指令采纳率
- 误指挥 / 漏指挥
- 人工可执行性评分（1–5）

---

## 11. 第一版验收标准

- 热键响应 p95 < 100ms
- 每条指令 ≤ 20 个汉字
- 100% 指令通过 schema/合法性校验
- LLM 失败时 100% 回退模板
- 每条计划都有完整状态卡和候选
- 能记录 plan + outcome
- 连续 5 局无崩溃
- 人工抽检 20 条指令，可执行性平均 ≥ 4/5

---

## 12. 实现模块（下一步，不写代码）

```text
live/
├── state_card.py        # 状态卡构建
├── route_graph.py       # Dust2 CT 拓扑与路径搜索
├── candidate_builder.py # 候选生成
├── plan_cache.py        # 指纹、TTL、缓存
├── llm_planner.py       # prompt、schema、校验、超时
├── hotkey_listener.py   # 全局热键
└── tests/
```

服务端新增：

- `POST /plan` 或热键监听直接调用
- `GET /plan` 供网页显示
- shadow log 增加 plan/outcome 记录

---

## 13. 待确认

1. 路线图的初始版本是否按上面的节点手工维护？
2. `WAIT` 是否允许作为候选（信息不足时）？
3. 长按理由是否需要，还是第一版完全不做解释？
4. LLM 使用本地 Ollama，还是先留接口、第一版只用模板？

默认建议：

- 手工维护路线图，后续用 demo 校准
- 允许 `WAIT`
- 保留长按理由
- 第一版本地 Ollama，超时回退模板
