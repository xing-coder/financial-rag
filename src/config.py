"""集中配置:所有"可调的数字和名字"都放这,别散落在各处。

面试时能一口说出每个参数的值和理由,这个文件就是你的小抄。
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "raw" / "docs"   # 放 PDF 的地方

# --- 向量库 (pgvector / Postgres) ---
# 格式:postgresql+psycopg://<user>@<host>:<port>/<db>
# 用户是 mac 系统用户(brew 的 Postgres 默认拿它当超级用户),本地无密码。
PG_CONNECTION = "postgresql+psycopg://xingshi@localhost:5432/financial_rag"
COLLECTION_NAME = "bank_compliance_docs"            # 一个 collection = 一套知识库

# --- 模型 ---
EMBED_MODEL = "BAAI/bge-base-en-v1.5"               # 本地 embedding,~0.44GB
OLLAMA_MODEL = "qwen2.5:7b"                          # 本地生成 LLM

# --- 切分(以 token 计,用 embedding 模型自己的 tokenizer)---
# bge-base 的上限是 512 token(含 [CLS]/[SEP] 特殊符),设 500 留余量,
# 否则超长的块会被模型悄悄截断 = 丢内容。
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 64

# --- 检索 ---
TOP_K = 5                                            # 召回喂给 LLM 的块数

# --- 接地 / 拒答(Day2)---
# 相似度相关性阈值(0~1,越高越相似)。最高分低于它 → 直接拒答,不调 LLM。
# 这个值是"猜的起点",必须用已知的库内/库外问题实测分布后校准(Day4)。
# 实测校准:库内问题 0.716~0.763,库外 0.426~0.612 → 卡在 0.65。
# margin 仅 ~0.1 偏脆,所以必须搭配第二道 LLM 接地闸。Day4 用完整拒答集再细调。
RELEVANCE_THRESHOLD = 0.65
REFUSAL_MSG = "I cannot answer this based on the available documents."

# --- 输出护栏(Day3)---
GUARDRAIL_BLOCK_MSG = "This response was blocked by the output guardrail."
