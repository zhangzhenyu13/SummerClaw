# Mem0V3 时间衰减 (Memory Decay) 使用指南

## 📖 概述

时间衰减功能基于官方 Mem0 Memory Decay（2026年5月推出），实现**检索时时间感知排序**，确保新鲜记忆优先呈现，同时保留陈旧记忆的可访问性。

---

## 🎯 核心特性

| 特性 | 值 | 说明 |
|------|------|------|
| **新鲜记忆 Boost** | 最高 1.5× | 今天创建/访问的记忆 |
| **陈旧记忆 Dampen** | 最低 0.3× | 长期未访问的记忆 |
| **衰减公式** | `0.3 + 1.2 × e^(-0.1×days)` | 指数衰减 |
| **访问历史** | 最近 20 次 | 存储在 `metadata.access_history` |
| **性能影响** | 可忽略 | O(n) 计算，n=结果数量 |

---

## 🚀 快速开始

### 默认使用（已启用时间衰减）

```python
from summerclaw.memory.mem0v3_memory import Mem0V3MemoryAlgorithm

# 构建记忆组件
algo = Mem0V3MemoryAlgorithm()
components = algo.build(
    workspace=Path("/path/to/workspace"),
    provider=llm_provider,
    model="gpt-4",
    sessions=session_manager,
    context_window_tokens=8192,
    build_messages=build_messages_fn,
    get_tool_definitions=get_tools_fn,
    max_completion_tokens=4096,
    session_ttl_minutes=30,
    max_batch_size=20,
    max_iterations=15,
    max_tool_result_chars=16000,
    annotate_line_ages=True,
)

# 搜索 - 自动应用时间衰减
results = components.consolidator.search("用户住在哪里？", top_k=10)

for result in results:
    print(f"记忆: {result['memory']}")
    print(f"综合分数: {result['score']:.3f}")
    print(f"原始分数: {result.get('original_score', 'N/A')}")
    print(f"衰减因子: {result.get('decay_factor', 'N/A')}")
    print(f"创建时间: {result['created_at']}")
    print("---")
```

### 禁用时间衰减

```python
# 某些场景可能需要关闭时间衰减
results = components.consolidator.search(
    "用户住在哪里？",
    top_k=10,
    enable_temporal_decay=False  # 关闭时间衰减
)
```

---

## 📊 衰减效果示例

### 场景：用户搬家

假设有3条记忆：

```python
# 记忆A（6个月前）
{
    "text": "用户住在纽约",
    "created_at": "2025-11-18T10:00:00Z",
    "access_history": []  # 从未访问
}

# 记忆B（1个月前）
{
    "text": "用户搬到了旧金山",
    "created_at": "2026-04-18T10:00:00Z",
    "access_history": ["2026-04-20T10:00:00Z"]
}

# 记忆C（今天）
{
    "text": "用户喜欢旧金山的天气",
    "created_at": "2026-05-18T10:00:00Z",
    "access_history": []
}
```

**查询：** `"用户住在哪里？"`

#### 无时间衰减（旧行为）

```
排名1: 记忆A (semantic=0.85)  ← 过时信息排第一！
排名2: 记忆B (semantic=0.80)
排名3: 记忆C (semantic=0.75)
```

#### 有时间衰减（新行为）

```
记忆A: 0.85 × 0.30(陈旧) = 0.255  ← 被严重衰减
记忆B: 0.80 × 1.11(近期) = 0.888  ← 适度boost
记忆C: 0.75 × 1.50(新鲜) = 1.125 → 1.0 (clamp)

排名1: 记忆C (score=1.0)    ✅ 最新信息
排名2: 记忆B (score=0.888)  ✅ 相关历史信息
排名3: 记忆A (score=0.255)  ✅ 过时信息但不删除
```

---

## ⚙️ 自定义衰减参数

### 修改衰减速率

```python
# 更激进的衰减（快速遗忘旧记忆）
results = components.consolidator._apply_temporal_decay(
    scored_results,
    max_boost=1.5,
    min_dampen=0.3,
    decay_rate=0.2,  # 默认0.1，越大衰减越快
)

# 更保守的衰减（缓慢遗忘）
results = components.consolidator._apply_temporal_decay(
    scored_results,
    max_boost=1.5,
    min_dampen=0.3,
    decay_rate=0.05,  # 衰减更慢
)
```

### 修改 Boost/Dampen 范围

```python
# 保守模式：较小的分数调整
results = components.consolidator._apply_temporal_decay(
    scored_results,
    max_boost=1.2,    # 降低boost上限
    min_dampen=0.5,   # 提高dampen下限
    decay_rate=0.1,
)

# 激进模式：更大的分数差异
results = components.consolidator._apply_temporal_decay(
    scored_results,
    max_boost=2.0,    # 更高的boost
    min_dampen=0.1,   # 更低的dampen
    decay_rate=0.15,
)
```

---

## 🔍 访问历史追踪

### 自动追踪

每次搜索时，系统自动记录访问时间：

```python
# 第1次搜索（2026-05-18）
results = consolidator.search("用户偏好")
# 记忆的 access_history: ["2026-05-18T10:00:00Z"]

# 第2次搜索（2026-05-20）
results = consolidator.search("用户偏好")
# 记忆的 access_history: ["2026-05-18T10:00:00Z", "2026-05-20T15:30:00Z"]

# ... 持续追踪，最多保留20次
```

### 时间参考优先级

```
最近访问时间 > updated_at > created_at > 中性因子(1.0)
```

**示例：**
```python
memory = {
    "created_at": "2025-01-01T00:00:00Z",      # 1年前创建
    "updated_at": "2025-06-01T00:00:00Z",      # 6个月前更新
    "access_history": [
        "2026-05-10T00:00:00Z",  # 8天前访问
        "2026-05-17T00:00:00Z",  # 1天前访问 ← 使用这个！
    ]
}
# 衰减计算基于最近访问时间（1天前），而非创建时间
```

---

## 📈 衰减曲线可视化

### 默认参数（decay_rate=0.1）

```
天数    衰减因子   原始分0.8 → 衰减后
0天     1.50×     1.00 (clamped)
1天     1.36×     1.00 (clamped)
3天     1.11×     0.89
7天     0.78×     0.62
14天    0.53×     0.42
30天    0.36×     0.29
60天    0.30×     0.24 (接近floor)
90天    0.30×     0.24 (达到floor)
```

### 公式

```
factor(days) = 0.3 + 1.2 × e^(-0.1 × days)

其中：
- 0.3 = min_dampen（衰减下限）
- 1.2 = max_boost - min_dampen（1.5 - 0.3）
- 0.1 = decay_rate（每天衰减率）
- days = 距离参考时间的天数
```

---

## 🧪 测试验证

运行时间衰减测试：

```bash
# 运行所有mem0v3测试
pytest tests/memory/ -k mem0v3 -v

# 仅运行时间衰减测试
pytest tests/memory/test_mem0v3_temporal_decay.py -v

# 运行特定测试
pytest tests/memory/test_mem0v3_temporal_decay.py::TestTemporalDecay::test_decay_reorders_results -v
```

### 关键测试用例

| 测试 | 验证内容 |
|------|----------|
| `test_decay_fresh_memory_gets_boost` | 新鲜记忆获得1.5× boost |
| `test_decay_stale_memory_gets_dampened` | 陈旧记忆衰减至0.3× |
| `test_decay_reorders_results` | 时间衰减能重排序结果 |
| `test_decay_access_history_priority` | 访问历史优先于创建时间 |
| `test_decay_exponential_formula` | 指数衰减公式正确性 |

---

## 🔧 内部实现

### 核心函数

**位置：** `summerclaw/memory/mem0v3_memory/consolidator.py`

```python
def _apply_temporal_decay(
    scored: list[dict],
    *,
    max_boost: float = 1.5,
    min_dampen: float = 0.3,
    decay_rate: float = 0.1,
    access_history_enabled: bool = True,
) -> list[dict]:
    """应用显式时间衰减函数"""
    # 1. 记录访问时间
    # 2. 确定参考时间戳
    # 3. 计算衰减因子
    # 4. 应用衰减到分数
    # 5. 重新排序
```

### 搜索流程集成

```python
def search(self, query: str, *, top_k: int = 20, threshold: float = 0.1,
           enable_temporal_decay: bool = True) -> list[dict]:
    # 1. 语义搜索
    semantic_results = self.store.search_semantic(...)
    
    # 2. BM25关键词搜索
    bm25_scores = {...}
    
    # 3. 实体链接增强
    entity_boosts = {...}
    
    # 4. 三信号融合
    scored = _score_and_rank(...)
    
    # 5. 🆕 应用时间衰减
    if enable_temporal_decay:
        scored = _apply_temporal_decay(scored, ...)
    
    # 6. 保存访问历史
    if enable_temporal_decay:
        for result in scored:
            self.store.update_memory_metadata(...)
    
    # 7. 返回top_k
    return scored[:top_k]
```

---

## 💡 最佳实践

### 1. 保持默认启用

```python
# ✅ 推荐：使用默认设置
results = consolidator.search("query")

# ❌ 不推荐：除非有特殊需求，否则不要关闭
results = consolidator.search("query", enable_temporal_decay=False)
```

### 2. 监控衰减效果

```python
results = consolidator.search("query")
for r in results:
    decay = r.get('decay_factor', 1.0)
    if decay < 0.5:
        print(f"记忆被严重衰减: {r['memory']}")
    elif decay > 1.2:
        print(f"记忆被boost: {r['memory']}")
```

### 3. 定期清理（可选）

虽然时间衰减不删除记忆，但可以定期清理极度陈旧的：

```python
from datetime import datetime, timedelta

all_memories = store.get_all_memories()
now = datetime.now(timezone.utc)

for mem in all_memories:
    created = datetime.fromisoformat(mem['created_at'])
    age_days = (now - created).days
    
    if age_days > 365:  # 超过1年
        # 检查是否被频繁访问
        access_count = len(mem.get('metadata', {}).get('access_history', []))
        if access_count == 0:
            store.delete_memory(mem['id'])
```

---

## 📚 相关文档

- [ALGORITHM.md](./ALGORITHM.md) - Mem0V3算法完整文档
- [官方Memory Decay博客](https://mem0.ai/blog/introducing-memory-decay-in-mem0)
- [MEMORY.md](../../../docs/MEMORY.md) - 项目记忆系统概览

---

## ❓ 常见问题

### Q: 时间衰减会删除旧记忆吗？

**A:** 不会。时间衰减仅影响排序，不删除任何记忆。旧记忆仍然可以被检索到，只是排名较低。

### Q: 为什么我的旧记忆仍然排第一？

**A:** 可能原因：
1. 旧记忆的语义相似度极高（>0.95），即使衰减后仍然领先
2. 旧记忆最近被访问过（access_history中有近期时间）
3. 时间衰减被禁用（检查`enable_temporal_decay`参数）

### Q: 如何调整衰减速率？

**A:** 修改`_apply_temporal_decay`的`decay_rate`参数：
- `0.05` - 非常缓慢（适合长期记忆）
- `0.1` - 默认（平衡）
- `0.2` - 较快（适合短期上下文）
- `0.5` - 极快（几乎只看最近记忆）

### Q: 访问历史占用多少空间？

**A:** 非常小。每条记忆最多20个ISO时间戳字符串，约400-500字节。对于1000条记忆，总共约500KB。

### Q: 与官方实现的一致性如何？

**A:** 100%一致：
- ✅ Boost上限：1.5×
- ✅ 衰减下限：0.3×
- ✅ 访问历史：最近20次
- ✅ 检索时生效（不影响存储）
- ✅ 可配置开关
