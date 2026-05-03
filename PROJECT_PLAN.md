# 简易长期记忆系统 Project Plan

## 1. 项目目标

构建一个受 mem0 思路启发的简易长期记忆系统，支持从对话中抽取用户长期偏好、事实和上下文信息，经过去重、更新和持久化后，在用户后续提问时检索相关记忆并自动注入到 LLM Prompt 中。

最终系统需要能端到端跑通以下流程：

```text
Conversation Input
    ↓
Memory Extraction
    ↓
Memory Update / Dedup
    ↓
Vector Store Persistence
    ↓
Query-time Memory Retrieval
    ↓
Prompt Augmentation
    ↓
LLM Response
```

## 2. MVP 范围

MVP 只做单用户或多用户都可用的最小闭环，不追求复杂分布式架构。

必须支持：

- 从最近对话和当前 user/assistant pair 中抽取候选 memory。
- 将 memory 保存到本地持久化存储。
- 为 memory 生成 embedding，并支持向量相似度检索。
- 对候选 memory 执行 `ADD`、`UPDATE`、`DELETE`、`NOOP`。
- 根据用户新 query 检索相关 memory。
- 将检索出的 memory 拼接进 LLM prompt。
- 提供一个可直接运行的 demo 脚本或 CLI，验证完整链路。
- 对于候选memory 如果触发`DELETE` 不是直接物理删除，而是标记 invalid。这对 temporal reasoning 很重要，因为用户状态是随时间变化的。

暂不支持：

- 多租户权限系统。
- 复杂后台任务队列。
- 生产级可观测性平台。
- 跨设备同步。
- 复杂 memory graph 或实体关系图谱。

## 3. 推荐技术栈

### 3.1 语言与运行环境

- Python 3.11+
- `uv` 或 `pip` 管理依赖
- `.env` 管理模型 API Key

### 3.2 核心依赖

- LLM SDK：优先使用 OpenAI SDK，也可以保留抽象接口方便替换模型。
- Embedding：OpenAI text embedding 或本地 sentence-transformers。
- Vector Store：MVP 使用 ChromaDB 或 SQLite + numpy cosine similarity。
- 数据持久化：SQLite。
- 配置：`pydantic-settings` 或简单 `.env`。
- 测试：`pytest`。

### 3.3 MVP 推荐选型

为了最快跑通，推荐：

- SQLite 保存 memory 结构化字段。
- SQLite 同步保存 embedding JSON 或 BLOB。
- 用 numpy 在本地做 cosine similarity。
- 抽象出 `EmbeddingClient` 和 `LLMClient`，后续可替换为 Chroma、Qdrant、pgvector。

这样可以避免一开始引入过多外部服务，降低启动成本。

## 4. 目标目录结构

```text
isla_memory/
  __init__.py
  agent.py
  config.py
  llm_client.py
  embedding_client.py
  memory_extractor.py
  memory_store.py
  memory_updater.py
  memory_retriever.py
  models.py
  prompts.py
  utils.py

scripts/
  demo_chat.py
  reset_memory_db.py

tests/
  test_memory_extractor.py
  test_memory_store.py
  test_memory_updater.py
  test_memory_retriever.py
  test_agent_e2e.py

.env.example
pyproject.toml
README.md
PROJECT_PLAN.md
```

## 5. 核心数据模型

### 5.1 Message

```python
class Message(BaseModel):
    message_id: str
    user_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime
```

### 5.2 CandidateMemory

```python
class CandidateMemory(BaseModel):
    content: str
    memory_type: Literal["preference", "fact", "goal", "constraint", "profile", "other"]
    confidence: float
    source_message_id: str | None = None
    metadata: dict[str, Any] = {}
```

### 5.3 Memory

```python
class Memory(BaseModel):
    memory_id: str
    user_id: str
    content: str
    embedding: list[float]
    created_at: datetime
    updated_at: datetime
    source_message_id: str | None = None
    metadata: dict[str, Any] = {}
```

### 5.4 MemoryDecision

```python
class MemoryDecision(BaseModel):
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    candidate: CandidateMemory
    target_memory_id: str | None = None
    final_content: str | None = None
    reason: str
```

## 6. 模块设计

### 6.1 `memory_extractor.py`

职责：

- 输入 recent messages 和当前 user/assistant pair。
- 调用 LLM 或规则抽取候选 memories。
- 输出结构化 `CandidateMemory` 列表。

输入：

```python
extract_memories(
    user_id: str,
    recent_messages: list[Message],
    current_user_message: Message,
    current_assistant_message: Message | None,
) -> list[CandidateMemory]
```

抽取原则：

- 只保存长期有用的信息。
- 不保存一次性问题、短期上下文、无意义闲聊。
- 偏好、长期目标、稳定身份信息、长期约束优先保存。
- 避免保存敏感信息，除非用户明确要求长期记住。

示例候选 memory：

```json
{
  "content": "用户偏好用中文进行技术讨论。",
  "memory_type": "preference",
  "confidence": 0.92,
  "source_message_id": "msg_123",
  "metadata": {
    "topic": "communication"
  }
}
```

### 6.2 `memory_store.py`

职责：

- 保存、读取、更新、删除 memory。
- 持久化 embedding。
- 提供按 user_id 过滤的相似度检索底层能力。

需要实现：

```python
class MemoryStore:
    def add_memory(self, memory: Memory) -> Memory: ...
    def update_memory(self, memory_id: str, content: str, embedding: list[float], metadata: dict) -> Memory: ...
    def delete_memory(self, memory_id: str) -> None: ...
    def get_memory(self, memory_id: str) -> Memory | None: ...
    def list_memories(self, user_id: str) -> list[Memory]: ...
    def similarity_search(self, user_id: str, query_embedding: list[float], top_k: int) -> list[tuple[Memory, float]]: ...
```

SQLite 表结构：

```sql
CREATE TABLE memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_message_id TEXT,
    metadata_json TEXT NOT NULL
);

CREATE INDEX idx_memories_user_id ON memories(user_id);
```

### 6.3 `memory_updater.py`

职责：

- 对 candidate memory 做 similarity search。
- 根据相似 memory 和候选 memory 决策 `ADD`、`UPDATE`、`DELETE`、`NOOP`。
- 执行对应 store 操作。

核心流程：

```text
for candidate in candidates:
    candidate_embedding = embed(candidate.content)
    similar_memories = store.similarity_search(user_id, candidate_embedding, top_k=5)
    decision = decide(candidate, similar_memories)
    apply(decision)
```

决策规则 MVP：

- 如果没有相似 memory，执行 `ADD`。
- 如果相似度高于 `0.90` 且内容基本一致，执行 `NOOP`。
- 如果相似度高于 `0.80` 且候选内容更新或更具体，执行 `UPDATE`。
- 如果候选内容表达用户不再需要某项记忆，执行 `DELETE`。
- 如果置信度低于阈值，比如 `0.65`，执行 `NOOP`。

可先使用规则决策，后续再引入 LLM 决策器。

### 6.4 `memory_retriever.py`

职责：

- 对用户 query 生成 embedding。
- 从 memory store 中 top-k 检索。
- 按阈值过滤。
- 可选 rerank。
- 返回 relevant memories。

接口：

```python
class MemoryRetriever:
    def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.72,
    ) -> list[Memory]: ...
```

MVP 策略：

- 先使用 embedding cosine similarity。
- 默认 `top_k=5`。
- 默认 `min_score=0.72`。
- 如果 query 太短，可以降低阈值或结合最近会话。

### 6.5 `agent.py`

职责：

- 接收用户 query。
- 调用 retriever 获取 relevant memories。
- 拼接 prompt。
- 调用 LLM 生成回复。
- 将当前 user/assistant pair 再送入 extractor/updater 更新记忆。

主流程：

```text
user query
    ↓
retrieve relevant memories
    ↓
build prompt
    ↓
LLM response
    ↓
extract candidate memories from current turn
    ↓
update memory store
    ↓
return response
```

Prompt 格式：

```text
You are a helpful assistant.

Relevant user memories:
- 用户偏好用中文回答技术问题。
- 用户正在做一个简易长期记忆系统。

Current conversation:
User: ...

Answer the current question using relevant memories only when useful.
```

## 7. 端到端运行路径

### 7.1 初始化

```bash
python scripts/reset_memory_db.py
```

效果：

- 创建本地 SQLite 数据库。
- 初始化 `memories` 表。

### 7.2 Demo Chat

```bash
python scripts/demo_chat.py
```

示例对话：

```text
User: 我喜欢你以后用中文解释技术问题，回答要直接一点。
Assistant: 好的，我会尽量用中文、直接地解释技术问题。

User: 我现在在做一个类似 mem0 的简易记忆系统。
Assistant: 明白。

User: 我刚才说我偏好什么回答风格？
Assistant: 你偏好我用中文解释技术问题，并且回答直接一点。
```

系统内部应完成：

- 第一轮抽取 memory：`用户偏好用中文、直接地解释技术问题。`
- 第二轮抽取 memory：`用户正在做一个类似 mem0 的简易记忆系统。`
- 第三轮 query-time retrieval 命中回答风格相关 memory。
- Prompt augmentation 后生成正确回复。

## 8. Milestones

### Milestone 1：项目骨架与基础配置

目标：

- 建立项目目录结构。
- 配置依赖、环境变量和基础运行命令。

交付物：

- `pyproject.toml`
- `.env.example`
- `isla_memory/config.py`
- `isla_memory/models.py`
- `README.md`

验收标准：

- 能运行 `python -m isla_memory` 或基础 import。
- `pytest` 能启动，即使只有空测试。
- `.env.example` 明确需要的 API Key 和模型配置。

### Milestone 2：Memory Store 持久化

目标：

- 实现 SQLite memory 存储。
- 支持 CRUD 和 user_id 隔离。

交付物：

- `isla_memory/memory_store.py`
- `scripts/reset_memory_db.py`
- `tests/test_memory_store.py`

验收标准：

- 能新增 memory。
- 能按 memory_id 查询。
- 能按 user_id 列出 memories。
- 能更新 content、embedding、metadata、updated_at。
- 能删除 memory。

### Milestone 3：Embedding 与相似度检索

目标：

- 实现 embedding client。
- 在 store 层支持 cosine similarity top-k search。

交付物：

- `isla_memory/embedding_client.py`
- `isla_memory/utils.py`
- store similarity search 测试

验收标准：

- 相同文本相似度接近 1。
- 明显相关文本能排在无关文本之前。
- 检索结果只返回当前 user_id 的 memories。

### Milestone 4：Memory Extraction

目标：

- 实现从对话中抽取候选 memory。
- 输出严格结构化数据。

交付物：

- `isla_memory/memory_extractor.py`
- `isla_memory/prompts.py`
- `tests/test_memory_extractor.py`

验收标准：

- 对明确偏好能抽取 candidate memory。
- 对一次性问题不抽取 memory。
- 输出字段包含 content、memory_type、confidence、source_message_id、metadata。
- LLM 输出解析失败时有 fallback，不导致主流程崩溃。

### Milestone 5：Memory Update / Dedup

目标：

- 实现候选 memory 的去重和更新逻辑。
- 支持 `ADD`、`UPDATE`、`DELETE`、`NOOP`。

交付物：

- `isla_memory/memory_updater.py`
- `tests/test_memory_updater.py`

验收标准：

- 新事实会 `ADD`。
- 重复事实会 `NOOP`。
- 更具体的新偏好会 `UPDATE`。
- 用户明确取消某个偏好时能 `DELETE`。
- 低置信度 candidate 不写入 store。

### Milestone 6：Query-time Retrieval

目标：

- 实现 query 时 memory 检索。
- 支持 top-k 和 threshold filter。

交付物：

- `isla_memory/memory_retriever.py`
- `tests/test_memory_retriever.py`

验收标准：

- 对相关 query 返回相关 memories。
- 对不相关 query 返回空列表或低数量结果。
- 能调整 `top_k` 和 `min_score`。
- 返回结果按相似度排序。

### Milestone 7：Agent Prompt Augmentation

目标：

- 将 retrieval 和 LLM response 串起来。
- 当前轮对话结束后自动更新 memory。

交付物：

- `isla_memory/agent.py`
- `tests/test_agent_e2e.py`

验收标准：

- agent 接收 query 后能检索 memory。
- prompt 中包含 relevant memories。
- assistant response 能使用相关 memory。
- response 生成后会抽取并更新新 memory。

### Milestone 8：CLI Demo 与文档

目标：

- 提供一个用户可以直接运行的 demo。
- 文档说明如何安装、配置、运行和验证。

交付物：

- `scripts/demo_chat.py`
- `README.md`
- 完整测试用例

验收标准：

- 新机器按 README 执行命令即可跑通。
- demo 能展示 memory 写入、检索和 prompt augmentation。
- `pytest` 通过。

## 9. 实现优先级

建议按以下顺序实现：

1. `models.py`
2. `memory_store.py`
3. `embedding_client.py`
4. `memory_retriever.py`
5. `memory_extractor.py`
6. `memory_updater.py`
7. `agent.py`
8. `scripts/demo_chat.py`
9. tests 和 README

原因：

- 先完成数据模型和存储层，后续模块都依赖它。
- retriever 依赖 embedding 和 store。
- updater 依赖 extractor、embedding 和 store。
- agent 是最后的编排层。

## 10. 关键配置项

```text
MEMORY_DB_PATH=./data/memory.sqlite3
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
MEMORY_TOP_K=5
MEMORY_MIN_SCORE=0.72
MEMORY_DEDUP_SCORE=0.90
MEMORY_UPDATE_SCORE=0.80
MEMORY_MIN_CONFIDENCE=0.65
```

## 11. Memory Extraction Prompt 草案

```text
You extract long-term user memories from conversation.

Only extract information that is likely to be useful in future conversations.
Do not extract one-off tasks, temporary context, or trivial statements.
Avoid sensitive personal data unless the user explicitly asks the assistant to remember it.

Return JSON only:
{
  "memories": [
    {
      "content": "...",
      "memory_type": "preference|fact|goal|constraint|profile|other",
      "confidence": 0.0,
      "source_message_id": "...",
      "metadata": {}
    }
  ]
}

Recent messages:
{recent_messages}

Current user message:
{current_user_message}

Current assistant message:
{current_assistant_message}
```

## 12. Update Decision 策略

### 12.1 规则版 MVP

```text
if candidate.confidence < MEMORY_MIN_CONFIDENCE:
    NOOP
elif candidate expresses deletion or cancellation:
    DELETE matching memory if similarity >= MEMORY_UPDATE_SCORE
elif no similar memory above MEMORY_UPDATE_SCORE:
    ADD
elif top similarity >= MEMORY_DEDUP_SCORE and contents are equivalent:
    NOOP
else:
    UPDATE top similar memory with merged content
```

### 12.2 后续增强版

引入 LLM decision prompt：

- 输入 candidate memory。
- 输入 top similar existing memories。
- 输出 action、target_memory_id、final_content、reason。

适合处理：

- 语义相似但表达差异较大的记忆。
- 新旧偏好冲突。
- 需要合并多条 memory 的情况。

## 13. 测试计划

### 13.1 单元测试

- MemoryStore CRUD。
- Embedding similarity。
- Extractor JSON parsing。
- Updater action decision。
- Retriever top-k 和 threshold。
- Agent prompt builder。

### 13.2 集成测试

端到端测试场景：

```text
Turn 1:
User: 以后请用中文回答我的技术问题。
Assistant: 好的。

Expected:
memory store contains "用户偏好用中文回答技术问题。"

Turn 2:
User: 我喜欢回答短一点，直接给结论。
Assistant: 明白。

Expected:
memory store contains communication preference, possibly updated/merged.

Turn 3:
User: 我偏好什么回答方式？

Expected:
retriever returns communication preference memory.
agent prompt includes retrieved memory.
assistant response mentions Chinese and concise/direct style.
```

### 13.3 回归测试

- 同一句偏好重复出现不会新增多条重复 memory。
- 用户说“不要再记住我喜欢 X”时能删除对应 memory。
- user A 的 query 不会检索到 user B 的 memory。
- 空数据库时 agent 能正常回答。
- LLM extraction 返回非法 JSON 时主流程不中断。

## 14. Demo 验收脚本

`scripts/demo_chat.py` 应展示：

```python
agent = MemoryAgent(user_id="demo_user")

print(agent.chat("以后请用中文回答技术问题，回答直接一点。"))
print(agent.chat("我正在做一个类似 mem0 的简易长期记忆系统。"))
print(agent.chat("我刚才告诉过你我的回答风格偏好吗？"))

print(agent.list_memories())
```

期望输出：

- 第一次对话后，memory store 新增回答风格偏好。
- 第二次对话后，memory store 新增项目上下文。
- 第三次对话时，retriever 命中回答风格偏好。
- `list_memories()` 能看到已保存的 memories。

## 15. 风险与处理

### 15.1 抽取过度

风险：

- 系统把大量短期对话误存成长期 memory。

处理：

- 提高 `MEMORY_MIN_CONFIDENCE`。
- 在 prompt 中强调只抽取长期有用信息。
- 增加“不应抽取”的测试用例。

### 15.2 重复 memory 堆积

风险：

- 语义重复但文本不同的 memories 越存越多。

处理：

- 使用 similarity threshold。
- 对高相似内容做 `NOOP` 或 `UPDATE`。
- 定期增加 memory compaction 任务。

### 15.3 错误更新

风险：

- 新候选 memory 覆盖了原本正确的 memory。

处理：

- MVP 中只更新 top-1 高相似 memory。
- 更新时保留 metadata 的 update history。
- 后续引入 LLM decision reason 方便调试。

### 15.4 隐私和敏感信息

风险：

- 系统保存用户不希望长期保存的信息。

处理：

- 默认不保存敏感个人信息。
- 支持用户显式删除 memory。
- 在 README 中说明数据存储位置和清除方式。

## 16. 后续增强路线

### Phase 2

- 使用 Chroma、Qdrant 或 pgvector 替换本地 numpy 检索。
- 增加 LLM reranker。
- 增加 LLM-based update decision。
- 增加 memory merge / compaction。
- 增加 memory importance score。
- 支持 memory expiration。

### Phase 3

- 多用户 API 服务。
- FastAPI REST endpoints。
- Web UI 查看、编辑、删除 memories。
- 对接真实聊天应用。
- 增加审计日志。
- 增加加密存储。

## 17. API 形态建议

### 17.1 Python SDK

```python
memory_agent = MemoryAgent(user_id="user_123")
response = memory_agent.chat("我偏好什么语言回答？")
```

### 17.2 FastAPI 后续接口

```text
POST /chat
GET /users/{user_id}/memories
POST /users/{user_id}/memories
PATCH /users/{user_id}/memories/{memory_id}
DELETE /users/{user_id}/memories/{memory_id}
```

## 18. Definition of Done

项目完成时应满足：

- 可以通过 `scripts/demo_chat.py` 跑通完整 memory 生命周期。
- 有 SQLite memory 持久化文件。
- 有 memory extraction、update、retrieval、prompt augmentation 完整模块。
- 关键路径有测试覆盖。
- README 包含安装、配置、运行、测试说明。
- 用户能看到 memory 被新增、更新、去重、检索和注入 prompt 的过程。

## 19. 推荐最终运行命令

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python scripts/reset_memory_db.py
python scripts/demo_chat.py
pytest
```

## 20. 最小可运行版本完成标准

最小可运行版本不要求所有增强能力都完美，但必须做到：

- 输入用户对话。
- 抽取至少一条稳定 memory。
- 保存 memory。
- 后续 query 能检索到该 memory。
- prompt 中能看到该 memory。
- assistant 最终回答能利用该 memory。

