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
    """Per-stage @Retry parameters (seconds for wait/timeout)."""
    max_attempt: int = 3
    wait: float = 0.1
    timeout: float = 60.0

    @field_validator("max_attempt", mode="before")
    @classmethod
    def _coerce_attempt(cls, v):
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 3
        return max(1, n)

    @field_validator("wait", "timeout", mode="before")
    @classmethod
    def _coerce_nonneg_float(cls, v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, x)


class DocRecognitionConfig(BaseModel):
    """PDF vision recognition settings (multimodal VLM)."""
    api_key: str = ""
    base_url: str = ""
    model_args: dict = Field(default_factory=lambda: {
        'model': 'Qwen3-VL-8B-Instruct',
        'temperature': 0.0,
        'enable_thinking': False,
        'response_format': {'type': 'json_object'},
    })
    use_cache: bool = True
    num_thread: int = 4
    prompt: str = 'pdf_recognize'
    dpi: int = 150
    image_format: str = 'png'
    # 长 PDF / 多图 VL 调用默认给足单次超时
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(max_attempt=3, wait=0.1, timeout=300.0)
    )

    @field_validator("api_key", "base_url", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return _none_to_empty(v)


class DocConfig(BaseModel):
    count_token: bool = True
    source_type: str = 'pdf'
    doc_dir: str = 'doc'
    recognition: DocRecognitionConfig = Field(default_factory=DocRecognitionConfig)


class ChunkConfig(BaseModel):
    count_token: bool = True
    chunk_size_min: int = 300
    chunk_size_max: int = 512
    force_single_chunk: bool = True
    chunk_overlap: float = 0.1

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


class ExtractConfig(BaseModel):
    # Empty → fall back to settings
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
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(max_attempt=3, wait=0.1, timeout=60.0)
    )

    @field_validator("api_key", "base_url", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return _none_to_empty(v)


class BuildConfig(BaseModel):
    target: List[str] = Field(default_factory=list)
    # hyperedge.content uses full chunk text instead of extract.knowledge
    hyperedge_use_full_chunk: bool = True


class VectorizationConfig(BaseModel):
    # Empty → fall back to settings
    api_key: str = ""
    base_url: str = ""
    model_args: dict = Field(default_factory=dict)
    dim: int = 1024
    default_target: List[str] = Field(default_factory=list)
    use_cache: bool = True
    num_thread: int = 1
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(max_attempt=3, wait=0.1, timeout=60.0)
    )

    @field_validator("api_key", "base_url", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return _none_to_empty(v)

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
    # 已废弃：双路合并后不再做全局资料组截断。保留字段仅为兼容旧 yaml。
    top_k: Optional[int] = None
    # 双路各自截断（合并后全部进上下文 / 或再经 rerank 截断）：
    #   chunk 路：query ↔ chunk 向量 Top-K
    #   node 路：query ↔ node 向量 Top-K → 映射到所属块
    chunk_candidate_k: int = 30
    node_candidate_k: int = 20
    # LLM answer / rewrite endpoint (empty → settings)
    api_key: str = ""
    base_url: str = ""
    embedding_model_args: dict = Field(default_factory=dict)
    # Embedding endpoint (empty → vectorization keys, then settings)
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    model_args: dict = Field(default_factory=lambda: {
        'enable_thinking': False,
    })
    # 改写专用覆盖（合并进 model_args）；限制输出长度避免拖慢
    rewrite_model_args: dict = Field(default_factory=lambda: {
        'enable_thinking': False,
        'max_tokens': 128,
        'temperature': 0.0,
    })
    # 双路检索前先 LLM 改写查询
    enable_query_rewrite: bool = True
    # 命中超边后，是否把 hyperedge.recommendation 中的推荐超边 content 一并带入上下文
    enable_recommendation_expand: bool = False
    # 命中某文档任一 chunk（头块或索引块，双路均可）后：
    # 将该文档全部 body 索引块写入上下文（按原文序），且不包含头块。
    # 仅命中头块时同样只扩 body，不写 head。
    enable_full_body_context: bool = False

    # ── 文档级重排（双路合并/扩展/去重之后） ────────────────────────────
    # 将每份完整文档（块按原文序纯文本拼接，无「资料N」标注）送入 Reranker，
    # 只保留 top_k 份进入最终 LLM 上下文。
    enable_rerank: bool = True
    rerank_top_k: int = 4
    # 空 → embedding_api_key / embedding_base_url → vectorization → settings
    rerank_api_key: str = ""
    rerank_base_url: str = ""
    rerank_model_args: dict = Field(default_factory=lambda: {
        'model': 'Qwen3-Reranker-0.6B',
    })
    # 送入重排模型的单文档最大字符；-1 / 0 = 不截断
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

    @field_validator("rerank_top_k", mode="before")
    @classmethod
    def _coerce_rerank_top_k(cls, v):
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 4
        return max(1, n)

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
    """
    相似超边推荐（离线模块，不参与 build 流水线）。
    按节点 name 关键词筛选 → 节点向量 HDBSCAN 聚类 → 写 hyperedge.recommendation。
    """
    # 在 node.name 列做「包含」匹配；多次查找后合并去重
    keywords: List[str] = Field(default_factory=lambda: ['用途', '应用', '场景'])
    # 簇内最大元素数（非簇个数上限）；簇个数不限。
    # HDBSCAN 之后会把超大簇二次切分到不超过该值。
    max_cluster_size: int = 3
    # HDBSCAN：形成簇的最小元素数；1 点簇无意义，默认 2
    min_cluster_size: int = 2
    # HDBSCAN min_samples；None 时与 min_cluster_size 相同
    min_samples: Optional[int] = None
    # 距离度量（文本 embedding 常用 cosine / euclidean）
    metric: str = 'cosine'
    # 随机种子（便于复现；HDBSCAN 本身近似确定，仍固定 numpy 种子）
    random_seed: int = 42
    # 无向量的节点是否跳过（只能跳过，无法聚类）
    skip_missing_embedding: bool = True
    # HDBSCAN 簇选择方式：eom | leaf
    cluster_selection_method: str = 'eom'


class AgentConfig(BaseModel):
    """
    轻量多跳 Agent（src/agent）。
    规划 / 依赖改写 / 单跳作答 / 汇总 共用一套 LLM 配置，与 retrieve 解耦。
    """
    # Empty → fall back to settings
    api_key: str = ""
    base_url: str = ""
    model_args: dict = Field(default_factory=lambda: {
        'model': 'qwen3.6-27b',
        'temperature': 0.2,
        'enable_thinking': False,
    })
    use_cache: bool = True
    # 规划步骤数上限
    max_steps: int = 12
    # 相对 settings.working_path 的临时步骤记事本文件名
    notebook_path: str = 'agent_scratchpad.md'
    # 单跳 skill 检索前是否用 agent LLM 改写（不走 retrieve 改写配置）
    enable_query_rewrite: bool = False
    # 检索宽度；None 时沿用 retrieve.chunk_candidate_k / node_candidate_k
    chunk_candidate_k: Optional[int] = None
    node_candidate_k: Optional[int] = None

    @field_validator("api_key", "base_url", "notebook_path", mode="before")
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


class AliOssBucketConfig(BaseModel):
    ACCESS_KEY: str = ""
    SECRET_KEY: str = ""
    END_POINT: str = ""
    BUCKET_NAME: str = ""


class AliOssConfig(BaseModel):
    # Bucket key as in master_env, e.g. ky-products-files
    ky_products_files: AliOssBucketConfig = Field(
        default_factory=AliOssBucketConfig,
        alias='ky-products-files',
    )

    model_config = {'populate_by_name': True}


class OssDownloadConfig(BaseModel):
    """
    download_from_oss(): query spider_product + download from Aliyun OSS.
    Function args (limit / file_type) override these when not None.
    """
    # Table name in dm_data_mysql
    table: str = 'spider_product'
    # 1=TDS, 2=MSDS, all=both types
    file_type: Union[str, int] = 'all'
    # 0 = no limit (fetch all matching rows)
    limit: int = 0
    # Relative to settings.working_path (same as doc.doc_dir by default)
    download_dir: str = 'doc'
    # Which ali_oss bucket key to use
    bucket_key: str = 'ky-products-files'


class Config(BaseModel):
    settings: SettingsConfig
    doc: DocConfig = Field(default_factory=DocConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    build: BuildConfig = Field(default_factory=BuildConfig)
    vectorization: VectorizationConfig = Field(default_factory=VectorizationConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    recommend: RecommendConfig = Field(default_factory=RecommendConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    # Infrastructure (copied from production master_env; optional for pipeline)
    dm_data_mysql: MysqlConfig = Field(default_factory=MysqlConfig)
    ali_oss: dict = Field(default_factory=dict)
    oss_download: OssDownloadConfig = Field(default_factory=OssDownloadConfig)
    # Extra master_env blocks kept for reference / future use (not validated strictly)
    extra_env: dict = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Known top-level pipeline keys consumed by Config model
        known = {
            'settings', 'doc', 'chunk', 'extract', 'build',
            'vectorization', 'retrieve', 'recommend', 'agent',
            'dm_data_mysql', 'ali_oss', 'oss_download',
        }
        # Preserve remaining master_env-style keys under extra_env if present,
        # or keep them as ignored pass-through via model_config extra allow
        pipeline = {k: v for k, v in data.items() if k in known}
        extras = {k: v for k, v in data.items() if k not in known}
        if extras:
            pipeline['extra_env'] = extras
        return cls(**pipeline)
