# Isla Persistent Memory

一个受 mem0 启发的简易长期记忆系统 MVP。当前版本不依赖外部 API，默认使用规则抽取、SQLite 持久化和本地 hash embedding，目的是先把完整链路跑通：

```text
Conversation Input
  -> Memory Extraction
  -> Memory Update / Dedup
  -> Vector Store Persistence
  -> Query-time Memory Retrieval
  -> Prompt Augmentation
  -> LLM Response
```

## 功能

- 从用户对话中抽取长期 memory candidate。
- 保存 memory 到 SQLite。
- 保存 embedding 并支持相似度检索。
- 支持 `ADD`、`UPDATE`、`DELETE`、`NOOP`。
- `DELETE` 是软删除：标记 `invalid_at`，不物理删除。
- 新 query 到来时检索 relevant memories。
- 将 retrieved memories 注入 prompt。
- 提供可直接运行的 demo。

## 目录结构

```text
isla_memory/
  agent.py
  config.py
  embedding_client.py
  llm_client.py
  memory_extractor.py
  memory_retriever.py
  memory_store.py
  memory_updater.py
  models.py
  prompts.py
  utils.py

scripts/
  demo_chat.py
  reset_memory_db.py

tests/
  test_agent_e2e.py
  test_memory_extractor.py
  test_memory_retriever.py
  test_memory_store.py
  test_memory_updater.py
```

## 快速开始

```bash
python scripts/reset_memory_db.py
python scripts/demo_chat.py
```

Demo 会执行三轮对话：

1. 用户要求以后用中文、直接地回答技术问题。
2. 用户说明正在做类似 mem0 的长期记忆系统。
3. 用户询问自己刚才告诉过什么回答风格偏好。

期望结果：

- 第一轮后写入回答风格 memory。
- 第二轮后写入项目背景 memory。
- 第三轮会检索出回答风格 memory，并把它放进 prompt。

## 测试

不安装额外依赖也可以运行：

```bash
python -m unittest discover -s tests
```

如果安装了 dev 依赖，也可以运行：

```bash
pip install -e ".[dev]"
pytest
```

## 配置

复制 `.env.example` 到 `.env` 后可调整阈值：

```text
MEMORY_DB_PATH=./data/memory.sqlite3
MEMORY_TOP_K=5
MEMORY_MIN_SCORE=0.35
MEMORY_DEDUP_SCORE=0.90
MEMORY_UPDATE_SCORE=0.62
MEMORY_MIN_CONFIDENCE=0.65
HASH_EMBEDDING_DIMENSION=256
```

当前默认 embedding 是 `HashEmbeddingClient`，用于离线 demo 和测试。生产化时建议替换为真实 embedding model 和向量数据库，但业务层接口已经拆开：

- `EmbeddingClient`
- `LLMClient`
- `MemoryStore`
- `MemoryRetriever`
- `MemoryUpdater`

## Python 用法

```python
from isla_memory import MemoryAgent

agent = MemoryAgent(user_id="user_123")

agent.chat("以后请用中文回答技术问题，回答直接一点。")
response = agent.chat("我刚才告诉过你我的回答风格偏好吗？")

print(response)
print(agent.list_memories())
```

## Project Plan

详细 milestone、模块边界、验收标准见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。
