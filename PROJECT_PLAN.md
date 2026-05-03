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
    memory_type: Literal["preference", "fact", "goal", "constraint", "profile", "other"]
    embedding: list[float]
    created_at: datetime
    updated_at: datetime
    invalid_at: datetime | None = None
    source_message_id: str | None = None
    metadata: dict[str, Any] = {}

    @property
    def is_valid(self) -> bool:
        return self.invalid_at is None
```

约定：

- `invalid_at is None` 表示 active memory。
- `invalid_at is not None` 表示 soft-deleted / invalid memory。
- `memory_type` 是一等字段，不只放在 metadata 中，避免 candidate 写入 store 后类型信息丢失。

### 5.4 MemoryDecision

```python
class MemoryDecision(BaseModel):
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    candidate: CandidateMemory
    target_memory_id: str | None = None
    final_content: str | None = None
    confidence: float
    reason: str
    metadata_patch: dict[str, Any] = {}
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
- 对“不要再记住”“忘记”“我不想再用 X”“我改变主意了”等否定或取消表达，应标记为 DELETE 意图，而不是抽取成新的正向偏好。

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

DELETE 意图示例：

```json
{
  "content": "用户不再希望使用英文回答技术问题。",
  "memory_type": "preference",
  "confidence": 0.9,
  "source_message_id": "msg_456",
  "metadata": {
    "intent": "delete",
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
    memory_type TEXT NOT NULL DEFAULT 'other',
    embedding_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    invalid_at TEXT,
    source_message_id TEXT,
    metadata_json TEXT NOT NULL
);

CREATE INDEX idx_memories_user_id ON memories(user_id);
CREATE INDEX idx_memories_valid ON memories(user_id, invalid_at);
```

存储约束：

- `similarity_search` 默认只检索 `invalid_at IS NULL` 的 active memories。
- `delete_memory` 执行 soft delete：写入 `invalid_at`，不物理删除。
- `ADD` 时必须从 `CandidateMemory.memory_type` 透传到 `Memory.memory_type`。

### 6.3 `memory_updater.py`

职责：

- 对 candidate memory 做 similarity search。
- 将 candidate memory 与 top-s 相似旧 memories 一起交给 LLM decision module。
- LLM 只能通过 tool call / function calling 选择 `ADD`、`UPDATE`、`DELETE`、`NOOP`。
- 校验 LLM 决策，并在非法输出时回退到规则版决策。
- 执行对应 store 操作。

核心流程：

```text
for candidate_memory ω_i in candidates:
    if ω_i.confidence < MEMORY_MIN_CONFIDENCE:
        decision = NOOP
    else:
        candidate_embedding = embed(ω_i.content)
        similar_memories = store.similarity_search(
            user_id=user_id,
            query_embedding=candidate_embedding,
            top_k=MEMORY_UPDATE_TOP_S,
        )
        llm_decision = decision_llm.tool_call(
            candidate_memory=ω_i,
            retrieved_memories=similar_memories,
            current_user_message=current_user_message,
            source_metadata=ω_i.metadata,
        )
        decision = validate_or_fallback(llm_decision, candidate=ω_i, similar_memories=similar_memories)
    apply(decision)
```

LLM decision 输入：

- `candidate_memory`: 当前候选记忆 `ω_i`。
- `retrieved_memories`: top-s 相似旧 memories，包含 `memory_id`、`content`、`metadata`、`similarity_score`、`created_at`、`updated_at`、`invalid_at`。
- `current_user_message`: 触发本轮更新的用户消息。
- `source_metadata`: `source_message_id`、memory type、confidence、topic 等抽取来源信息。

LLM 可调用的 tool / function schema：

```text
add_memory(
    content: str,
    memory_type: str,
    metadata_patch: dict,
    reason: str,
    confidence: float
)

update_memory(
    memory_id: str,
    content: str,
    metadata_patch: dict,
    reason: str,
    confidence: float
)

delete_memory(
    memory_id: str,
    reason: str,
    confidence: float
)

noop(
    reason: str,
    confidence: float
)
```

执行约束：

- LLM 只返回操作意图，不直接写数据库。
- `UPDATE` / `DELETE` 的 `memory_id` 必须来自本轮 retrieved memories。
- `ADD` / `UPDATE` 必须提供 `final_content`。
- `DELETE` 是软删除：标记 `invalid_at`，不物理删除。
- 在调用 LLM decision 前，updater 先做 negation/delete intent 前置检测；命中“不要再记住”“忘记”“不想再用”“改变主意”等表达时，优先按 DELETE 候选处理。
- 如果 LLM 输出非法、缺少必填字段、target 不在 retrieved memories 中，系统回退到规则版决策。
- 规则版仍保留为 `MEMORY_UPDATE_STRATEGY=rules` 或 LLM 失败时的 fallback。

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
- `Memory` 模型包含 `memory_type` 和 `invalid_at`，并定义 active / soft-deleted 语义。

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
- 能 soft delete memory，并默认从 active list 和 similarity search 中过滤 invalid memories。
- SQLite schema 与 `Memory` 模型一致，包含 `memory_type` 和 `invalid_at`。

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
- LLM tool-call 决策结果会被校验；非法 action、缺失 `memory_id`、target 不在 top-s 中时会 fallback。
- `memory_type` 从 candidate 透传到新增或更新后的 memory。

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
- 默认不返回 soft-deleted / invalid memories。
- 支持 recency decay 排序：`final_score = similarity * MEMORY_SIM_WEIGHT + recency_score * MEMORY_RECENCY_WEIGHT`。

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
- prompt augmentation 遵守 `MEMORY_MAX_CONTEXT_TOKENS`，按检索排序注入 memories，超过预算即截断。

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
4. `memory_extractor.py`
5. `memory_updater.py`
6. `memory_retriever.py`
7. `agent.py`
8. `scripts/demo_chat.py`
9. tests 和 README

原因：

- 先完成数据模型和存储层，后续模块都依赖它。
- extractor 和 updater 先于 retriever，有助于用真实 memory 写入路径生成检索测试数据。
- retriever 依赖 embedding、store 和真实 memory 数据形态。
- agent 是最后的编排层。

## 10. 关键配置项

```text
MEMORY_DB_PATH=./data/memory.sqlite3
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
MEMORY_UPDATE_STRATEGY=llm_tool_call
MEMORY_UPDATE_TOP_S=5
MEMORY_DECISION_MODEL=gpt-4.1-mini
MEMORY_DECISION_MIN_CONFIDENCE=0.65
MEMORY_DECISION_FALLBACK=rules
MEMORY_TOP_K=5
MEMORY_MIN_SCORE=0.72
MEMORY_MAX_CONTEXT_TOKENS=800
MEMORY_RECENCY_HALF_LIFE_DAYS=30
MEMORY_RECENCY_WEIGHT=0.3
MEMORY_SIM_WEIGHT=0.7
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
If the user negates, cancels, changes, or asks to forget a previous preference or fact, mark the candidate as a DELETE intent instead of creating a new positive memory.

DELETE intent examples:
- "不要再记住我喜欢 X" -> metadata.intent = "delete"
- "忘记我的项目偏好" -> metadata.intent = "delete"
- "我不想再用英文回答了" -> metadata.intent = "delete"
- "我改变主意了，不要简短回答" -> metadata.intent = "delete"

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

### 12.1 LLM Tool-Call 决策版

目标：

- 用 embedding 检索 candidate memory `ω_i` 的 top-s 相似旧 memories。
- 把 `ω_i` 与 retrieved memories 一起交给 LLM。
- LLM 通过 tool/function calling 选择唯一操作。
- 系统校验 tool call，并执行数据库写入。

流程：

```text
candidate memory ω_i
    ↓
embed(ω_i.content)
    ↓
top-s similarity search from memory DB
    ↓
LLM decision prompt:
  - candidate memory
  - retrieved similar memories
  - similarity scores
  - source message metadata
    ↓
LLM tool call:
  ADD / UPDATE / DELETE / NOOP
    ↓
validate decision
    ↓
apply database operation
```

决策语义：

- `ADD`: 当前 candidate 是新的长期信息，与 retrieved memories 不重复也不冲突。
- `UPDATE`: 当前 candidate 对某条旧 memory 做补充、修正、合并或状态更新。
- `DELETE`: 当前 candidate 表达用户取消、否定或要求忘记某条旧 memory。
- `NOOP`: 当前 candidate 低价值、重复、临时、无关或不足以安全更新。

校验规则：

- `candidate.confidence < MEMORY_MIN_CONFIDENCE` 时可直接 `NOOP`，避免无意义 LLM 调用。
- 如果用户消息命中 negation/delete intent 前置检测，应优先检索相关旧 memory，并让 LLM decision 在 `DELETE` 与 `NOOP` 之间做受限选择。
- `UPDATE` / `DELETE` 必须指定 `target_memory_id`。
- `target_memory_id` 必须来自本轮 retrieved memories，不能让 LLM 任意指定数据库外 ID。
- `ADD` / `UPDATE` 必须提供最终写入内容 `final_content`。
- `DELETE` 只能执行软删除，标记 `invalid_at`。
- LLM 输出非法、缺少必填字段、置信度低于 `MEMORY_DECISION_MIN_CONFIDENCE` 时，走 fallback。

### 12.2 Tool Call Schema

```text
add_memory(
    content: str,
    memory_type: "preference|fact|goal|constraint|profile|other",
    metadata_patch: dict,
    reason: str,
    confidence: float
)

update_memory(
    memory_id: str,
    content: str,
    metadata_patch: dict,
    reason: str,
    confidence: float
)

delete_memory(
    memory_id: str,
    reason: str,
    confidence: float
)

noop(
    reason: str,
    confidence: float
)
```

系统将 tool call 统一转换为 `MemoryDecision`：

```python
class MemoryDecision(BaseModel):
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    candidate: CandidateMemory
    target_memory_id: str | None = None
    final_content: str | None = None
    confidence: float
    reason: str
    metadata_patch: dict[str, Any] = {}
```

### 12.3 规则版 fallback

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

fallback 触发场景：

- `MEMORY_UPDATE_STRATEGY=rules`。
- LLM decision 调用失败或超时。
- LLM 返回非法 action。
- `UPDATE` / `DELETE` 缺少 `memory_id`。
- `memory_id` 不属于本轮 retrieved memories。
- `ADD` / `UPDATE` 缺少最终内容。
- LLM decision confidence 低于阈值。

### 12.4 Negation / Delete Intent 前置检测

在 LLM decision 前先用轻量规则识别否定、取消或遗忘意图，降低 DELETE 误判风险。

典型触发表达：

- “不要再记住 ...”
- “忘记 ...”
- “我不想再用 X ...”
- “我改变主意了 ...”
- “以后不用 ...”
- “取消 ... 偏好”

处理策略：

- 命中后仍然检索 top-s 相似旧 memories。
- 将用户原文、candidate 和 retrieved memories 一起交给 LLM decision。
- LLM decision 在 `DELETE` 与 `NOOP` 之间选择，避免把否定表达误写成新的正向 memory。
- `DELETE` 仍执行 soft delete，只写入 `invalid_at`。

## 13. 测试计划

### 13.1 单元测试

- MemoryStore CRUD。
- Embedding similarity。
- Extractor JSON parsing，使用 mock LLM 输出，不真实调用 API。
- Updater LLM tool-call action decision，使用 mock tool call，不真实调用 API。
- 无相似 memory 时，LLM tool call 为 `ADD`，系统新增 memory。
- 高相似重复 memory 时，LLM tool call 为 `NOOP`，系统不新增重复项。
- candidate 与旧 memory 冲突或更具体时，LLM tool call 为 `UPDATE`，系统更新目标 memory。
- 用户明确取消偏好时，LLM tool call 为 `DELETE`，系统软删除目标 memory。
- LLM 返回非法 action、缺失 `memory_id`、或 target 不在 top-s 结果中时，触发 fallback。
- soft-deleted memory 不会出现在默认 retrieval 结果中。
- Retriever top-k、threshold 和 recency decay 排序。
- Agent prompt builder 按 `MEMORY_MAX_CONTEXT_TOKENS` 截断 memories。

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

Turn 4:
User: 不要再记住我的回答风格偏好。

Expected:
memory updater marks the matching preference memory invalid.
default retrieval no longer returns that invalid memory.

Turn 5:
User: 我现在偏好什么回答方式？

Expected:
when old and new relevant memories both exist, recency decay ranks the newer valid memory higher.
```

### 13.3 回归测试

- 同一句偏好重复出现不会新增多条重复 memory。
- 用户说“不要再记住我喜欢 X”时能删除对应 memory。
- invalid memory 默认不会被 query-time retrieval 返回。
- user A 的 query 不会检索到 user B 的 memory。
- 空数据库时 agent 能正常回答。
- LLM extraction 返回非法 JSON 时主流程不中断。
- LLM decision tool call 返回非法 JSON 或非法 tool name 时主流程不中断。

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
- MVP 不把 compaction 放在关键路径；先通过 `NOOP`、`UPDATE` 和 soft delete 控制增长。

### 15.3 错误更新

风险：

- 新候选 memory 覆盖了原本正确的 memory。

处理：

- MVP 中只更新 top-1 高相似 memory。
- 更新时保留 metadata 的 update history。
- LLM tool-call decision 必须返回 reason，便于调试。
- contradiction detector 放入 Phase 2，仅在基础链路稳定后增加。

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
- 将 embedding 从 JSON TEXT 优化为 float32 BLOB 存储。
- 增加 Hybrid Retrieval：BM25 + Vector + Recency decay + RRF。
- 增加 LLM reranker。
- 增加 Cross-encoder reranker。
- 增加 contradiction detector。
- 增加 memory merge / compaction。
- 增加 memory importance score。
- 增加 memory max-per-user 容量上限和 eviction 策略。
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
