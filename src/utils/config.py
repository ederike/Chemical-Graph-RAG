from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml

def resolve_credentials(
    config: "Config",
    stage: Any = None,
    *,
    api_key_attr: str = "api_key",
    base_url_attr: str = "base_url",
) -> tuple:
    """
    Resolve (api_key, base_url) for a pipeline stage.
    Stage fields win when non-empty; otherwise fall back to settings.
    """
    api_key = None
    base_url = None
    if stage is not None:
        api_key = getattr(stage, api_key_attr, None)
        base_url = getattr(stage, base_url_attr, None)
    if not api_key:
        api_key = config.settings.api_key
    if not base_url:
        base_url = config.settings.base_url
    return (api_key or "EMPTY"), (base_url or "")

def _none_to_empty(v):
    """YAML empty keys often load as None; treat as empty string."""
    return "" if v is None else v

class SettingsConfig(BaseModel):
    working_path: str
    debug: bool = False
    api_key: str = "EMPTY"
    base_url: str = ""

    @field_validator("api_key", "base_url", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        if v is None:
            return ""
        return v

class RetryConfig(BaseModel):
    """
    Per-stage @Retry parameters (seconds for wait/timeout).

    wait 支持：
      - 标量：固定间隔，如 0.1
      - 列表：指数退避序列，如 [10, 30, 60]
        （第 1/2/3 次失败后分别等待；超出序列则用最后一项）
    """
    max_attempt: int = 3
    wait: Union[float, List[float]] = 0.1
    timeout: float = 60.0

    @field_validator("max_attempt", mode="before")
    @classmethod
    def _coerce_attempt(cls, v):
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 3
        return max(1, n)

    @field_validator("wait", mode="before")
    @classmethod
    def _coerce_wait(cls, v):
        # 标量
        if v is None:
            return 0.1
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                try:
                    out.append(max(0.0, float(x)))
                except (TypeError, ValueError):
                    continue
            return out if out else [0.1]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return 0.1
            if ',' in s:
                out = []
                for p in s.split(','):
                    p = p.strip()
                    if not p:
                        continue
                    try:
                        out.append(max(0.0, float(p)))
                    except (TypeError, ValueError):
                        continue
                return out if out else [0.1]
            try:
                return max(0.0, float(s))
            except (TypeError, ValueError):
                return 0.1
        try:
            return max(0.0, float(v))
        except (TypeError, ValueError):
            return 0.1

    @field_validator("timeout", mode="before")
    @classmethod
    def _coerce_nonneg_float(cls, v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, x)

class DocRecognitionConfig(BaseModel):
    """
    PDF vision recognition（页切片多图 VLM，切片级多线程）。

    渲染页数超过 max_pages_per_doc 时按该上限切成多段，每段独立识别、
    独立入库：首段保留原文件名，后续段为「原文件名_{n}」（n 从 1 起）。
    """
    api_key: str = ""
    base_url: str = ""
    model_args: dict = Field(default_factory=lambda: {
        'model': 'Qwen3-VL-8B-Instruct',
        'temperature': 0.0,
        'enable_thinking': False,
    })
    use_cache: bool = True
    num_thread: int = 4
    prompt: str = 'pdf_recognize'
    dpi: int = 150
    image_format: str = 'jpeg'
    # 单次 VLM 最多携带的渲染页数；> 该值则切成多个独立文档
    max_pages_per_doc: int = 12
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(
            max_attempt=4, wait=[10.0, 30.0, 60.0], timeout=600.0
        )
    )

    @field_validator("api_key", "base_url", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return _none_to_empty(v)

    @field_validator("max_pages_per_doc", mode="before")
    @classmethod
    def _coerce_max_pages(cls, v):
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 12
        return max(1, n)

def _coerce_flush_every(v, default: int = 1000) -> int:
    """每处理多少条落盘一次；<=0 时回落到 default。"""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return max(1, int(default))
    if n <= 0:
        return max(1, int(default))
    return n

class DocConfig(BaseModel):
    count_token: bool = True
    source_type: str = 'pdf'
    doc_dir: str = 'doc'
    recognition: DocRecognitionConfig = Field(default_factory=DocRecognitionConfig)
    flush_every: int = 1000

    @field_validator("flush_every", mode="before")
    @classmethod
    def _flush(cls, v):
        return _coerce_flush_every(v, 1000)

class ChunkConfig(BaseModel):
    """
    分块：hyperedge 总结 → head（整块）；doc 识别正文 → body_n（按 token + 重叠）。
    """
    count_token: bool = True
    chunk_size_max: int = 512
    chunk_overlap: float = 0.1
    flush_every: int = 1000

    @field_validator("chunk_overlap", mode="before")
    @classmethod
    def _normalize_chunk_overlap(cls, v):
        if v is None or v == "":
            return 0.1
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.1
        if x > 1.0:
            x = x / 100.0
        return max(0.0, min(x, 0.9))

    @field_validator("flush_every", mode="before")
    @classmethod
    def _flush(cls, v):
        return _coerce_flush_every(v, 1000)

class SummaryConfig(BaseModel):
    """
    文档总结（insert 之后、chunk 之前，可独立运行）。
    对 doc.content 全文 LLM 总结 → 写入 hyperedge.content。
    """
    api_key: str = ""
    base_url: str = ""
    model_args: dict = Field(default_factory=lambda: {
        'model': 'qwen3.6-27b',
        'temperature': 0.2,
        'enable_thinking': False,
    })
    prompt: str = 'doc_summary'
    use_cache: bool = True
    num_thread: int = 8
    flush_every: int = 500
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(max_attempt=3, wait=0.1, timeout=120.0)
    )

    @field_validator("api_key", "base_url", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return _none_to_empty(v)

    @field_validator("flush_every", mode="before")
    @classmethod
    def _flush(cls, v):
        return _coerce_flush_every(v, 500)

class ExtractConfig(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model_args: dict = Field(default_factory=lambda: {
        'model': 'qwen3:32b',
        'enable_thinking': False,
        'response_format': {'type': 'json_object'},
    })
    extract_prompt: str = 'extract'
    use_cache: bool = True
    num_thread: int = 1
    flush_every: int = 1000
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(max_attempt=3, wait=0.1, timeout=60.0)
    )

    @field_validator("api_key", "base_url", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return _none_to_empty(v)

    @field_validator("flush_every", mode="before")
    @classmethod
    def _flush(cls, v):
        return _coerce_flush_every(v, 1000)

class BuildConfig(BaseModel):
    """
    构图：超边已在 summary 写入；本步建 node，并可选回写 hyperedge.extra/chunk_id。
    target 可选：hyperedge（更新元数据）、node。
    """
    target: List[str] = Field(default_factory=lambda: ['hyperedge', 'node'])
    flush_every: int = 1000

    @field_validator("flush_every", mode="before")
    @classmethod
    def _flush(cls, v):
        return _coerce_flush_every(v, 1000)

class VectorizationConfig(BaseModel):
    """
    向量化配置。

    日常只需关心：
      - batch_size / num_thread  → API 并发（一次请求几条、几个请求并行）
      - shard_max_vectors       → 进度批大小（拉任务 / 写 FAISS / 落盘封片 共用这一档）

    下面三个是历史/细调项，在设置了 shard_max_vectors 后会自动对齐到它，
    一般不用再写：
      flush_every / index_save_every / task_page_size
    """
    api_key: str = ""
    base_url: str = ""
    model_args: dict = Field(default_factory=dict)
    dim: int = 1024
    default_target: List[str] = Field(default_factory=list)
    use_cache: bool = True
    num_thread: int = 1
    # Texts per embeddings API request (OpenAI input=list). 1 = legacy one-by-one.
    # Throughput ≈ num_thread × batch_size (server still rate-limits concurrent requests).
    batch_size: int = 1
    # --- progress batch (prefer shard_max_vectors alone) ---
    # Max vectors per FAISS shard = one durable batch:
    #   load page → embed → write FAISS → save disk → seal/unload.
    # Peak RAM ≈ N × dim × 4B (fp16: × 2B).
    # None/0 = mono single-file index (legacy; grows unboundedly in RAM).
    shard_max_vectors: Optional[int] = None
    # FAISS index backend: flat_l2 (exact brute-force) | hnsw (approx ANN).
    # HNSW does not support in-place remove_ids; rebuild via vectorization_clear.
    index_type: str = "hnsw"
    # FAISS stored-vector encoding (not the embedding API dtype):
    #   none — float32 (IndexHNSWFlat / IndexFlatL2)
    #   fp16 — ScalarQuantizer QT_fp16 (IndexHNSWSQ / IndexScalarQuantizer)
    # Switching encoding is incompatible with existing .vdb shards;
    # rebuild with vectorization_clear after changing this.
    index_quant: str = "none"
    # HNSW graph degree (M). Higher = better recall, more RAM / build time.
    hnsw_M: int = 32
    # HNSW build-time search depth. Higher = better graph, slower build.
    hnsw_efConstruction: int = 200
    # HNSW query-time search depth. Higher = better recall, slower search.
    hnsw_efSearch: int = 64
    # Legacy fine-tuning (auto-aligned to shard_max_vectors when it is set):
    flush_every: int = 1000
    index_save_every: Optional[int] = None
    task_page_size: Optional[int] = None
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(max_attempt=3, wait=0.1, timeout=60.0)
    )

    @field_validator("flush_every", mode="before")
    @classmethod
    def _flush(cls, v):
        return _coerce_flush_every(v, 1000)

    @field_validator("batch_size", mode="before")
    @classmethod
    def _batch_size(cls, v):
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 1
        return max(1, n)

    @field_validator(
        "index_save_every", "task_page_size", "shard_max_vectors", mode="before"
    )
    @classmethod
    def _optional_positive_int(cls, v):
        if v is None or v == "" or v == 0:
            return None
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

    @field_validator("index_type", mode="before")
    @classmethod
    def _index_type(cls, v):
        s = str(v or "hnsw").strip().lower().replace("-", "_")
        aliases = {
            "l2": "flat_l2",
            "flat": "flat_l2",
            "flatl2": "flat_l2",
            "indexflatl2": "flat_l2",
            "hnswflat": "hnsw",
            "hnsw_l2": "hnsw",
        }
        s = aliases.get(s, s)
        if s not in ("flat_l2", "hnsw"):
            return "hnsw"
        return s

    @field_validator("index_quant", mode="before")
    @classmethod
    def _index_quant(cls, v):
        if isinstance(v, bool):
            return "fp16" if v else "none"
        s = str(v or "none").strip().lower().replace("-", "_")
        aliases_fp16 = {
            "fp16", "float16", "half", "sq_fp16", "sqfp16",
            "qt_fp16", "qtfp16", "sq16", "true", "on", "1", "yes",
        }
        aliases_none = {
            "none", "off", "false", "flat", "fp32", "float32",
            "no", "0", "",
        }
        if s in aliases_fp16:
            return "fp16"
        if s in aliases_none:
            return "none"
        return "none"

    @field_validator("hnsw_M", mode="before")
    @classmethod
    def _hnsw_M(cls, v):
        try:
            return max(2, int(v))
        except (TypeError, ValueError):
            return 32

    @field_validator("hnsw_efConstruction", mode="before")
    @classmethod
    def _hnsw_efc(cls, v):
        try:
            return max(1, int(v))
        except (TypeError, ValueError):
            return 200

    @field_validator("hnsw_efSearch", mode="before")
    @classmethod
    def _hnsw_efs(cls, v):
        try:
            return max(1, int(v))
        except (TypeError, ValueError):
            return 64

    @field_validator("api_key", "base_url", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return _none_to_empty(v)

    @model_validator(mode="after")
    def _align_progress_batch_to_shard(self):
        """
        One progress knob: when shard_max_vectors is set, force
        flush_every = index_save_every = task_page_size = shard_max_vectors.
        """
        sm = self.shard_max_vectors
        if sm is not None and int(sm) > 0:
            sm = int(sm)
            self.shard_max_vectors = sm
            self.flush_every = sm
            self.index_save_every = sm
            self.task_page_size = sm
        return self

    @model_validator(mode="after")
    def _normalize_emb_args_and_dim(self):
        """
        model_args.dimension / dim / dims → dimensions；
        若写了维度，同步到 vectorization.dim（FAISS 索引维）。
        """
        try:
            from .OpenAIAPI import normalize_embedding_model_args
            ma = normalize_embedding_model_args(self.model_args)
        except Exception:
            ma = dict(self.model_args or {})
            for src in ("dimension", "dim", "dims"):
                if "dimensions" not in ma and ma.get(src) is not None:
                    try:
                        ma["dimensions"] = int(ma[src])
                    except (TypeError, ValueError):
                        pass
                ma.pop(src, None)
        self.model_args = ma
        d = ma.get("dimensions")
        if d is not None:
            try:
                d = int(d)
                if d > 0:
                    self.dim = d
            except (TypeError, ValueError):
                pass
        return self

class RetrieveConfig(BaseModel):
    use_cache: bool = True
    # Kept for old yaml compatibility; unused after dual-path merge.
    top_k: Optional[int] = None
    # 0 = skip that path
    chunk_candidate_k: int = 30
    node_candidate_k: int = 20
    api_key: str = ""
    base_url: str = ""
    embedding_model_args: dict = Field(default_factory=dict)
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    model_args: dict = Field(default_factory=lambda: {
        'enable_thinking': False,
    })
    rewrite_model_args: dict = Field(default_factory=lambda: {
        'enable_thinking': False,
        'max_tokens': 128,
        'temperature': 0.0,
    })
    enable_query_rewrite: bool = True
    enable_recommendation_expand: bool = False
    enable_full_body_context: bool = False

    enable_keyword_exact: bool = True
    keyword_candidate_k: int = 50
    keyword_top_k: int = 10
    keyword_extract_model_args: dict = Field(default_factory=lambda: {
        'enable_thinking': False,
        'max_tokens': 256,
        'temperature': 0.0,
        'response_format': {'type': 'json_object'},
    })

    enable_rerank: bool = True
    rerank_top_k: int = 4
    rerank_api_key: str = ""
    rerank_base_url: str = ""
    rerank_model_args: dict = Field(default_factory=lambda: {
        'model': 'Qwen3-Reranker-0.6B',
    })
    rerank_max_chars: int = -1

    @field_validator(
        "api_key", "base_url",
        "embedding_api_key", "embedding_base_url",
        "rerank_api_key", "rerank_base_url",
        mode="before",
    )
    @classmethod
    def _coerce_str(cls, v):
        return _none_to_empty(v)

    @field_validator(
        "chunk_candidate_k",
        "node_candidate_k",
        "keyword_candidate_k",
        "keyword_top_k",
        "rerank_top_k",
        mode="before",
    )
    @classmethod
    def _coerce_topk_allow_zero(cls, v, info):
        """
        各路 topk：>=1 正常截断；0（或负数）= 跳过该路 / 关闭截断。
        非法值回落到字段默认。
        """
        defaults = {
            'chunk_candidate_k': 30,
            'node_candidate_k': 20,
            'keyword_candidate_k': 50,
            'keyword_top_k': 10,
            'rerank_top_k': 4,
        }
        fallback = defaults.get(getattr(info, 'field_name', '') or '', 0)
        try:
            n = int(v)
        except (TypeError, ValueError):
            return fallback
        return max(0, n)

    @model_validator(mode="after")
    def _normalize_embedding_model_args(self):
        """embedding_model_args：dimension/dim/dims → dimensions。"""
        try:
            from .OpenAIAPI import normalize_embedding_model_args
            self.embedding_model_args = normalize_embedding_model_args(
                self.embedding_model_args
            )
        except Exception:
            ma = dict(self.embedding_model_args or {})
            for src in ("dimension", "dim", "dims"):
                if "dimensions" not in ma and ma.get(src) is not None:
                    try:
                        ma["dimensions"] = int(ma[src])
                    except (TypeError, ValueError):
                        pass
                ma.pop(src, None)
            self.embedding_model_args = ma
        return self

class RecommendConfig(BaseModel):
    """Offline similar-hyperedge recommendation (HDBSCAN on node embeddings)."""
    keywords: List[str] = Field(default_factory=lambda: ['用途', '应用', '场景'])
    max_cluster_size: int = 3
    min_cluster_size: int = 2
    min_samples: Optional[int] = None
    metric: str = 'cosine'
    random_seed: int = 42
    skip_missing_embedding: bool = True
    cluster_selection_method: str = 'eom'

class AgentConfig(BaseModel):
    """Multi-hop agent LLM settings (independent of retrieve)."""
    api_key: str = ""
    base_url: str = ""
    model_args: dict = Field(default_factory=lambda: {
        'model': 'qwen3.6-27b',
        'temperature': 0.2,
        'enable_thinking': False,
    })
    use_cache: bool = True
    max_steps: int = 12
    enable_query_rewrite: bool = False
    chunk_candidate_k: Optional[int] = None
    node_candidate_k: Optional[int] = None

    @field_validator("api_key", "base_url", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return _none_to_empty(v)

    @field_validator("max_steps", mode="before")
    @classmethod
    def _coerce_max_steps(cls, v):
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 12
        return max(1, n)

class MysqlConfig(BaseModel):
    """Generic MySQL connection block (e.g. dm_data_mysql)."""
    host: str = ""
    port: Union[int, str] = 3306
    user: str = ""
    password: str = ""
    db: str = ""

class OssDownloadConfig(BaseModel):
    """Defaults for download_from_oss(); function args override when not None."""
    table: str = 'spider_product'
    file_type: Union[str, int] = 'all'
    limit: int = 0
    download_dir: str = 'doc'
    bucket_key: str = 'ky-products-files'

class Config(BaseModel):
    settings: SettingsConfig
    doc: DocConfig = Field(default_factory=DocConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    build: BuildConfig = Field(default_factory=BuildConfig)
    vectorization: VectorizationConfig = Field(default_factory=VectorizationConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    recommend: RecommendConfig = Field(default_factory=RecommendConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    dm_data_mysql: MysqlConfig = Field(default_factory=MysqlConfig)
    ali_oss: dict = Field(default_factory=dict)
    oss_download: OssDownloadConfig = Field(default_factory=OssDownloadConfig)
    extra_env: dict = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        known = {
            'settings', 'doc', 'summary', 'chunk', 'extract', 'build',
            'vectorization', 'retrieve', 'recommend', 'agent',
            'dm_data_mysql', 'ali_oss', 'oss_download',
        }
        pipeline = {k: v for k, v in data.items() if k in known}
        extras = {k: v for k, v in data.items() if k not in known}
        if extras:
            pipeline['extra_env'] = extras
        return cls(**pipeline)
