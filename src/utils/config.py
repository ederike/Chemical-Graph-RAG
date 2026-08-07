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
    """
    PDF vision recognition（逐页 VLM，文件级多线程）。
    每页单独调用，中间页历史仅带上一页识别文本；输出纯文本（非 JSON）。
    """
    api_key: str = ""
    base_url: str = ""
    model_args: dict = Field(default_factory=lambda: {
        'model': 'Qwen3-VL-8B-Instruct',
        'temperature': 0.0,
        'enable_thinking': False,
    })
    use_cache: bool = True
    # 按文件并行（单文件内页序串行）
    num_thread: int = 4
    prompt: str = 'pdf_recognize'
    dpi: int = 150
    image_format: str = 'png'
    # 单页 VL 调用超时
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(max_attempt=3, wait=0.1, timeout=300.0)
    )

    @field_validator("api_key", "base_url", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return _none_to_empty(v)


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
    # 入库 / 识别分批：每 N 条 doc 写库（防一次性攒爆内存）
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
    # 相邻 body 块重叠比例（相对 chunk_size_max）。0.1 = 10%
    chunk_overlap: float = 0.1
    # 切块分批：chunk 行累计达 N 条则写库
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
    # 抽取分批：chunk 更新累计达 N 条则写库
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
    # 构图分批：node + hyperedge 更新累计达 N 条则写库
    flush_every: int = 1000

    @field_validator("flush_every", mode="before")
    @classmethod
    def _flush(cls, v):
        return _coerce_flush_every(v, 1000)


class VectorizationConfig(BaseModel):
    # Empty → fall back to settings
    api_key: str = ""
    base_url: str = ""
    model_args: dict = Field(default_factory=dict)
    dim: int = 1024
    default_target: List[str] = Field(default_factory=list)
    use_cache: bool = True
    num_thread: int = 1
    # 边嵌边写：每处理 N 条就刷 FAISS + SQLite（防 OOM）；与 embedding 缓存无关
    flush_every: int = 1000
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(max_attempt=3, wait=0.1, timeout=60.0)
    )

    @field_validator("flush_every", mode="before")
    @classmethod
    def _flush(cls, v):
        return _coerce_flush_every(v, 1000)

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
    #   chunk 路：query ↔ chunk 向量 Top-K；0 = 跳过该路
    #   node 路：query ↔ node 向量 Top-K → 映射到所属块；0 = 跳过该路
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

    # ── 关键词精确匹配第三路 ───────────────────────────────────────────
    # LLM 从查询中抽取 minority（少数值标识）/ majority（多数值字段键），
    # 在 chunk.content 上做精确包含匹配 → 大候选池 → 文档级首轮 rerank →
    # keyword_top_k →（可选全文扩展）→ 与双路文档取交集 → 终轮 rerank。
    enable_keyword_exact: bool = True
    # 关键词路候选块池大小（少数值优先 + 多数值命中数排序后截断）；0 = 跳过关键词路
    keyword_candidate_k: int = 50
    # 关键词路首轮文档 rerank 后保留的文档数；0 = 跳过关键词路
    keyword_top_k: int = 10
    # 抽取专用覆盖（合并进 model_args）
    keyword_extract_model_args: dict = Field(default_factory=lambda: {
        'enable_thinking': False,
        'max_tokens': 256,
        'temperature': 0.0,
        'response_format': {'type': 'json_object'},
    })

    # ── 文档级重排（双路合并/扩展/去重之后） ────────────────────────────
    # 将每份完整文档（块按原文序纯文本拼接，无「资料N」标注）送入 Reranker，
    # 只保留 top_k 份进入最终 LLM 上下文。
    # rerank_top_k=0 视为关闭终轮截断（等价于 enable_rerank=False，保留全部主资料）。
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
    # 单跳 skill 检索前是否用 agent LLM 改写（不走 retrieve 改写配置）
    enable_query_rewrite: bool = False
    # 检索宽度；None 时沿用 retrieve.chunk_candidate_k / node_candidate_k
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
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
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
            'settings', 'doc', 'summary', 'chunk', 'extract', 'build',
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
