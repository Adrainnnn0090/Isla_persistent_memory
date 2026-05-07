# Isla Persistent Memory

> 一个受 **mem0** 启发、以《可塑性记忆》中的 Isla 为初心的长期记忆系统实验项目。  
> 目标不是复刻动画里的奇迹，而是认真拆解：一个 AI agent 如何抽取、维护、检索并使用属于用户的长期记忆。

Isla 当前是一个可运行的 memory MVP：它能从对话中抽取候选记忆，写入本地 SQLite，使用 embedding 做相似度检索，并在后续回答前把相关记忆注入 prompt。默认模式可离线运行；也可以切换到 OpenAI API 做真实 LLM 回复、memory extraction 和 embedding。

```text
Conversation Input
  -> Memory Extraction
  -> Memory Update / Dedup
  -> Vector Store Persistence
  -> Query-time Memory Retrieval
  -> Prompt Augmentation
  -> LLM Response
```

## 当前能力

- 从用户对话中抽取长期有用的 candidate memories。
- 使用 SQLite 持久化 `memory_type`、metadata、embedding 和软删除状态。
- 支持 `ADD`、`UPDATE`、`DELETE`、`NOOP` 记忆更新动作。
- 支持 mem0 风格的 mockable LLM tool-call updater：LLM/测试桩只返回 `add_memory`、`update_memory`、`delete_memory`、`noop` 意图，数据库写入由系统校验后执行。
- 保留规则版 updater 作为默认离线策略和 LLM 决策失败 fallback。
- `DELETE` 采用 soft delete：标记 `invalid_at`，不物理删除，方便后续 temporal reasoning。
- query-time 使用 embedding 检索 relevant memories，并支持 recency-aware scoring。
- 将 retrieved memories 注入 prompt 的 `Relevant user memories` 区域，并受 `MEMORY_MAX_CONTEXT_TOKENS` 预算约束。
- 支持离线规则 demo，也支持 OpenAI API demo。
- 提供 LongMemEval hypothesis 生成脚本，用于科研评测链路。

## 项目结构

```text
isla_memory/
  agent.py                # 对话主编排：retrieve -> prompt -> response -> memory update
  config.py               # .env 配置
  embedding_client.py     # hash / OpenAI / BGE-M3 embedding client
  llm_client.py           # rules / OpenAI response client
  memory_extractor.py     # 从对话中抽取 candidate memories
  memory_retriever.py     # query-time memory retrieval
  memory_store.py         # SQLite persistence + similarity search
  memory_updater.py       # ADD / UPDATE / DELETE / NOOP
  models.py               # Message / Memory / CandidateMemory / MemoryDecision
  prompts.py              # prompt augmentation
  utils.py

scripts/
  demo_chat.py            # 离线 MVP demo
  demo_openai_chat.py     # OpenAI API demo
  run_longmemeval.py      # LongMemEval oracle retrieval baseline
  run_longmemeval_memory.py # LongMemEval memory-system evaluation
  reset_memory_db.py

tests/
  test_agent_e2e.py
  test_memory_extractor.py
  test_memory_retriever.py
  test_memory_store.py
  test_memory_updater.py
```

## 快速开始：离线模式

离线模式不需要 API key。它使用规则型抽取、规则型回复和本地 hash embedding，适合先确认完整链路。

```bash
python scripts/reset_memory_db.py
python scripts/demo_chat.py
```

demo 会执行三轮对话：

```text
User: 以后请用中文回答技术问题，回答直接一点。
Assistant: 好的，我会按这个偏好来回答。
Memory decision: ADD | 用户偏好用中文、直接、简洁地回答技术问题。

User: 我正在做一个类似 mem0 的简易长期记忆系统。
Assistant: 明白，我会把这个项目背景作为后续上下文。
Memory decision: ADD | 用户正在做一个类似 mem0 的简易长期记忆系统。

User: 我刚才告诉过你我的回答风格偏好吗？
Assistant: 你告诉过我：用户偏好用中文、直接、简洁地回答技术问题。
```

重点观察：

- 第一轮写入回答风格偏好。
- 第二轮写入项目背景。
- 第三轮通过 query-time retrieval 找回第一轮 memory。
- `agent.last_prompt` 中能看到 `Relevant user memories`。

## 使用 OpenAI API

安装可选依赖：

```bash
pip install -e ".[openai]"
```

复制配置：

```bash
cp .env.example .env
```

填写 `.env`：

```text
OPENAI_API_KEY=your_openai_api_key_here
MEMORY_LLM_PROVIDER=openai
MEMORY_LLM_MODEL=gpt-4.1-mini
MEMORY_EMBEDDING_PROVIDER=openai
MEMORY_EMBEDDING_MODEL=text-embedding-3-small
MEMORY_EXTRACTOR_PROVIDER=openai
MEMORY_EXTRACTOR_MODEL=gpt-4.1-mini
```

运行真实 API demo：

```bash
python scripts/demo_openai_chat.py
```

这个脚本会真实调用 API 完成：

- LLM 回复。
- 对话中的 candidate memory extraction。
- query 和 memory 的 embedding。
- SQLite 持久化。
- 后续 query 的 memory retrieval 和 prompt augmentation。

如果只想让回答走 OpenAI，但 memory extraction 保持本地规则，配置：

```text
MEMORY_EXTRACTOR_PROVIDER=rules
```

## Python 用法

```python
from isla_memory import MemoryAgent

agent = MemoryAgent(user_id="user_123")

agent.chat("以后请用中文回答技术问题，回答直接一点。")
agent.chat("我正在做一个类似 mem0 的简易长期记忆系统。")
response = agent.chat("我刚才告诉过你我的回答风格偏好吗？")

print(response)
print(agent.list_memories())
print(agent.last_prompt)
```

## 配置项

基础配置：

```text
MEMORY_DB_PATH=./data/memory.sqlite3
MEMORY_TOP_K=5
MEMORY_MIN_SCORE=0.35
MEMORY_MAX_CONTEXT_TOKENS=800
MEMORY_RECENCY_HALF_LIFE_DAYS=30
MEMORY_RECENCY_WEIGHT=0.3
MEMORY_SIM_WEIGHT=0.7
MEMORY_UPDATE_STRATEGY=rules
MEMORY_UPDATE_TOP_S=5
MEMORY_DECISION_MODEL=gpt-4.1-mini
MEMORY_DECISION_MIN_CONFIDENCE=0.65
MEMORY_DECISION_FALLBACK=rules
MEMORY_DEDUP_SCORE=0.90
MEMORY_UPDATE_SCORE=0.62
MEMORY_MIN_CONFIDENCE=0.65
HASH_EMBEDDING_DIMENSION=256
MEMORY_BGE_MODEL=BAAI/bge-m3
MEMORY_BGE_DEVICE=auto
MEMORY_BGE_BATCH_SIZE=8
MEMORY_BGE_MAX_LENGTH=8192
MEMORY_BGE_USE_FP16=true
```

Provider 配置：

```text
MEMORY_LLM_PROVIDER=rules|openai
MEMORY_EMBEDDING_PROVIDER=hash|openai|bge_m3
MEMORY_EXTRACTOR_PROVIDER=rules|openai
```

当前默认 embedding 是 `HashEmbeddingClient`，用于离线 demo 和测试。正式实验建议使用 `bge_m3` 或 OpenAI embedding，并为不同 embedding 模型使用不同数据库，避免向量维度和相似度分布混用。

本地 BGE-M3：

```bash
pip install -e ".[local-embedding]"
```

```text
MEMORY_EMBEDDING_PROVIDER=bge_m3
MEMORY_BGE_MODEL=BAAI/bge-m3
MEMORY_BGE_DEVICE=auto
```

## 测试

本项目的默认测试不会调用 OpenAI API，不会产生 API 费用。

```bash
python -m unittest discover -s tests
```

如果安装了 dev 依赖：

```bash
pip install -e ".[dev]"
pytest
```

## LongMemEval 评测

Isla 现在提供两条 LongMemEval 路径，语义不同：

- `scripts/run_longmemeval.py`：oracle retrieval baseline。它直接把 LongMemEval haystack 切成 session/turn 后写入 vector store，只能作为检索上限参考。
- `scripts/run_longmemeval_memory.py`：memory-system evaluation。它按时间 replay user/assistant pairs，经由 Isla extractor/updater 生成 memories，再 retrieval/answer，更适合对标 Mem0。默认每 8 个 pair 合成一个 chunk 调一次 extractor，避免每个 pair 都调用一次 LLM；`add-only` 模式可用 `--extract-concurrency` 对一个 question 内的所有 session/chunk jobs 做全局并发抽取，以降低墙钟时间。

Oracle retrieval baseline 小样本：

```bash
python scripts/run_longmemeval.py \
  --data ../LongMemEval/data/longmemeval_oracle.json \
  --output data/longmemeval_outputs/isla_oracle_10.jsonl \
  --limit 10 \
  --top-k 10 \
  --granularity turn
```

Memory-system evaluation 小样本：

```bash
python scripts/run_longmemeval_memory.py \
  --data ../LongMemEval/data/longmemeval_oracle.json \
  --output data/longmemeval_outputs/isla_memory_1.jsonl \
  --limit 1 \
  --top-k 20 \
  --extraction-granularity chunk \
  --chunk-pairs 8
```

BGE/OpenAI 都可用的 add-only 并发抽取版本：

```bash
python scripts/run_longmemeval_memory.py \
  --data ../LongMemEval/data/longmemeval_oracle.json \
  --output data/longmemeval_outputs/isla_memory_chunk8_async4.jsonl \
  --db-path data/longmemeval_outputs/isla_memory_chunk8_async4.sqlite3 \
  --top-k 20 \
  --ingest-mode add-only \
  --extraction-granularity chunk \
  --chunk-pairs 8 \
  --extract-concurrency 4
```

BGE-M3 本地 embedding：

```bash
MEMORY_EMBEDDING_PROVIDER=bge_m3 \
python scripts/run_longmemeval_memory.py \
  --data ../LongMemEval/data/longmemeval_oracle.json \
  --output data/longmemeval_outputs/isla_memory_bge_1.jsonl \
  --limit 1 \
  --top-k 20 \
  --extraction-granularity chunk \
  --chunk-pairs 8
```

输出格式：

```json
{"question_id": "...", "hypothesis": "..."}
```

再使用 LongMemEval 官方 `src/evaluation/evaluate_qa.py` 做 LLM judge。建议顺序：

```text
oracle --limit 1   # 验证环境和 API
oracle --limit 10  # 验证成本和结果格式
oracle full        # 500 条完整评测
```

如果中途断掉，可以使用：

```bash
python scripts/run_longmemeval_memory.py ... --resume
```

注意：

- `session` 粒度可能触发 embedding 单条输入长度限制。
- `turn` 粒度更适合当前 MVP 跑完整 benchmark。
- memory-system evaluation 默认只 embed 抽取后的短 memories，因此比 oracle session embedding 更不容易触发 OpenAI embedding 长度限制。
- memory-system evaluation 默认 `--extraction-granularity chunk --chunk-pairs 8`；如果要复现每个 user/assistant pair 一次 extractor 调用，可显式使用 `--extraction-granularity pair`。
- `--extract-concurrency N` 只作用于 `--ingest-mode add-only`：并发发起一个 question 内所有 session/chunk 的 extractor 请求，主线程仍按原 job 顺序写入 memory DB。`updater` 模式保持顺序执行，因为 ADD/UPDATE/DELETE/NOOP 依赖当前库状态。
- `top-k` 越大，回答阶段 prompt token 越多，费用也越高。
- LongMemEval 生成答案和官方 judge 都可能产生 API 费用。

## 研究路线

当前 Isla 是第一阶段原型，重点是让“记忆生命周期”跑通：

```text
extract -> decide -> persist -> retrieve -> augment -> respond
```

下一阶段重点：

- 接入真实 OpenAI / 其他 provider 的 updater decision client，而不只使用 mock tool-call 测试桩。
- 在 LongMemEval 上系统比较 oracle baseline、memory-system evaluation、不同 embedding provider 和不同 `top_k` 的效果。
- 后续再引入 hybrid retrieval、reranker、compaction、importance score、contradiction detector 和更严肃的 benchmark。

完整 milestone 和实现计划见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

## 项目初心

Isla 这个名字来自《可塑性记忆》。这部作品讨论的是记忆、人格、时间和告别。本项目借这个名字，不是为了浪漫化工程问题，而是提醒自己：长期记忆系统不是“把所有东西存起来”这么简单。

真正难的是：

- 什么值得记住。
- 什么应该被更新。
- 什么必须被忘记。
- 当记忆互相矛盾时，系统如何保持诚实。
- 当用户改变时，agent 如何跟着改变。

所以 Isla 的目标不是做一个会背数据库的聊天机器人，而是做一个能被观测、能被评测、能被修正的长期记忆研究原型。记忆会被写入，也会被软删除；偏好会被继承，也会被更新。像实验室里安静运行的一台小型 Giftia 维护装置，认真、克制，但始终朝着更接近“理解用户”的方向迭代。
