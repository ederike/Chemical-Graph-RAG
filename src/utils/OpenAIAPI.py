from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import urlparse

from openai import OpenAI

from .utils import Cache

_EXTRA_BODY_KEYS = (
    "enable_thinking",
    "thinking_budget",
    "enable_search",
    "result_format",
)

_LOCAL_DROP_CREATE_KEYS = (
    "enable_thinking",
    "thinking_budget",
    "enable_search",
    "result_format",
)

# 与 @Cache 装饰的 LLM.generate 共用同一 cache 库，便于校验失败时删毒缓存
_LLM_CACHE = Cache(cache_dir="cache/OpenAI", cache_name="llm_cache")

def split_model_args(model_args):
    """
    Split create() kwargs vs provider extra_body.

    Accepts enable_thinking (and friends) either at top-level of model_args
    or already nested under extra_body. Returns (create_kwargs, extra_body).
    """
    create_kwargs = dict(model_args or {})
    extra_body = dict(create_kwargs.pop("extra_body", None) or {})
    for key in _EXTRA_BODY_KEYS:
        if key in create_kwargs:
            extra_body[key] = create_kwargs.pop(key)
    return create_kwargs, extra_body

def _usage_from_completion(completion) -> dict:
    """Extract token usage; include reasoning_tokens when present."""
    out = {
        "usage_prompt_tokens": None,
        "usage_completion_tokens": None,
        "usage_total_tokens": None,
        "usage_cached_tokens": None,
        "usage_reasoning_tokens": None,
    }
    if completion is None or getattr(completion, "usage", None) is None:
        return out
    usage = completion.usage
    out["usage_prompt_tokens"] = getattr(usage, "prompt_tokens", None)
    out["usage_completion_tokens"] = getattr(usage, "completion_tokens", None)
    out["usage_total_tokens"] = getattr(usage, "total_tokens", None)
    try:
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            out["usage_cached_tokens"] = getattr(details, "cached_tokens", None)
    except Exception:
        pass
    try:
        cdetails = getattr(usage, "completion_tokens_details", None)
        if cdetails is not None:
            out["usage_reasoning_tokens"] = getattr(cdetails, "reasoning_tokens", None)
    except Exception:
        pass
    return out

def _is_private_or_local_url(base_url: str) -> bool:
    """True when base_url points at typical internal / loopback hosts."""
    if not base_url:
        return False
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "0.0.0.0"):
        return True
    if host.startswith("10."):
        return True
    if host.startswith("192.168."):
        return True
    # 172.16.0.0 – 172.31.255.255
    if host.startswith("172."):
        parts = host.split(".")
        try:
            second = int(parts[1])
            if 16 <= second <= 31:
                return True
        except (IndexError, ValueError):
            pass
    return False

def _is_placeholder_key(api_key: str) -> bool:
    k = (api_key or "").strip().lower()
    return k in ("", "none", "empty", "ollama", "token-abc123", "no-key", "nokey")

def _is_qwen3_family(model_id: str) -> bool:
    m = (model_id or "").lower()
    return "qwen3" in m

def _strip_think_tags(content: Optional[str]) -> str:
    """
    对齐 KyOpenAIServer：去掉本地 Qwen 返回的 <think>...</think> 前缀。
    """
    if content is None:
        return ""
    text = str(content)
    if text.startswith("<think>\n\n</think>\n\n"):
        return text[len("<think>\n\n</think>\n\n") :]
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end != -1:
            rest = text[end + len("</think") :]
            if rest.startswith(">"):
                rest = rest[1:]
            return rest.lstrip("\n")
    return text


def _flatten_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or ""))
            else:
                t = getattr(p, "text", None)
                parts.append(str(t) if t else "")
        return "".join(parts)
    return str(content)


def _content_from_message(msg, *, local_mode: bool) -> str:
    """取 assistant 正文；content 为空时回退 reasoning_content（思考模型常见）。"""
    if msg is None:
        return ""
    text = _flatten_message_content(getattr(msg, "content", None))
    if local_mode:
        text = _strip_think_tags(text)
    if str(text or "").strip():
        return text
    reasoning = getattr(msg, "reasoning_content", None)
    if not reasoning:
        return text or ""
    r = str(reasoning)
    return _strip_think_tags(r) if local_mode else r

def _prepare_local_call(
    model_args: dict,
    *,
    resolved_model: str,
    strip_thinking_extra: bool = True,
    inject_no_think: bool = True,
    messages: Optional[list] = None,
):
    """
    本地 GPU 调用准备：
      - 剥离云端 extra_body 扩展（顶层 enable_thinking 等），避免本地 400
      - Qwen3 族：用 chat_template_kwargs.enable_thinking 控制思考
        （缺省 / false → 关；true → 开）
      - inject_no_think：历史参数名，现表示是否注入上述 chat_template_kwargs
    """
    create_kwargs, extra_body = split_model_args(model_args)
    create_kwargs["model"] = resolved_model

    enable_thinking = None
    if "enable_thinking" in extra_body:
        enable_thinking = bool(extra_body.get("enable_thinking"))
    elif "enable_thinking" in (model_args or {}):
        enable_thinking = bool(model_args.get("enable_thinking"))
    # 已写在 chat_template_kwargs 里的显式值优先保留语义
    ctk_in = extra_body.get("chat_template_kwargs")
    if isinstance(ctk_in, dict) and "enable_thinking" in ctk_in:
        enable_thinking = bool(ctk_in.get("enable_thinking"))

    if strip_thinking_extra:
        for k in list(extra_body.keys()):
            if k in _LOCAL_DROP_CREATE_KEYS:
                extra_body.pop(k, None)
        for k in _LOCAL_DROP_CREATE_KEYS:
            create_kwargs.pop(k, None)

    msgs = messages
    # 缺省关思考（enable_thinking is not True）
    want_think = enable_thinking is True
    if inject_no_think and _is_qwen3_family(resolved_model):
        ctk = dict(extra_body.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = want_think
        extra_body["chat_template_kwargs"] = ctk

    return create_kwargs, extra_body, msgs


def _tool_calls_from_message(msg) -> list:
    """Normalize OpenAI / vLLM tool_calls to JSON-serializable dicts."""
    raw = getattr(msg, "tool_calls", None) or []
    out = []
    for i, tc in enumerate(raw):
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                try:
                    args = json.dumps(args or {}, ensure_ascii=False)
                except Exception:
                    args = str(args or "{}")
            out.append({
                "id": str(tc.get("id") or f"call_{i}"),
                "type": tc.get("type") or "function",
                "function": {
                    "name": str(fn.get("name") or ""),
                    "arguments": args or "{}",
                },
            })
            continue
        fn = getattr(tc, "function", None)
        args = getattr(fn, "arguments", None) if fn is not None else None
        if not isinstance(args, str):
            try:
                args = json.dumps(args or {}, ensure_ascii=False)
            except Exception:
                args = str(args or "{}")
        out.append({
            "id": str(getattr(tc, "id", None) or f"call_{i}"),
            "type": getattr(tc, "type", None) or "function",
            "function": {
                "name": str(getattr(fn, "name", None) or "") if fn is not None else "",
                "arguments": args or "{}",
            },
        })
    return out


class LLM:
    def __init__(
        self,
        api_key,
        base_url,
        timeout: float = 300.0,
        max_retries: int = 0,
        local_mode: Optional[bool] = None,
    ):
        """
        timeout: 单次 HTTP 请求超时（秒）。默认 300，避免思考/排队导致回答超时。
        max_retries: SDK 层自动重试次数；默认 0（由业务层 extract 自己重试）。
        local_mode: None 时自动判断（内网 URL 或 placeholder key → True）。
        """
        # 本地 GPU 常用 "none"（KyOpenAIServer）；EMPTY / token-abc123 亦可
        self.api_key = api_key if api_key not in (None, "") else "none"
        self.base_url = base_url
        self.timeout = float(timeout) if timeout is not None else 300.0
        self.max_retries = int(max_retries) if max_retries is not None else 0

        if local_mode is None:
            local_mode = _is_private_or_local_url(base_url or "") or _is_placeholder_key(
                str(self.api_key)
            )
        self.local_mode = bool(local_mode)

        # 连接超时单独放宽：默认 float 超时在部分环境下 connect 仍偏短，
        # 多模态并发时易出现 connect timeout → 识别整批失败。
        try:
            import httpx
            http_timeout = httpx.Timeout(
                self.timeout,
                connect=min(60.0, max(10.0, self.timeout / 4.0)),
            )
        except Exception:
            http_timeout = self.timeout

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=http_timeout,
            max_retries=self.max_retries,
        )

    def drop_generate_cache(self, prompt, model_args) -> int:
        """删除某次 generate 的缓存（抽取 JSON 校验失败时调用，避免永久命中坏结果）。"""
        return _LLM_CACHE.drop_for_generate(prompt, model_args)

    def _create(self, messages, model_args):
        """
        统一 create：本地走 KyOpenAIServer 风格，云端保留 extra_body。
        """
        model_args = dict(model_args or {})
        resolved_model = model_args.get("model") or ""

        if self.local_mode:
            create_kwargs, extra_body, messages = _prepare_local_call(
                model_args,
                resolved_model=resolved_model,
                strip_thinking_extra=True,
                inject_no_think=True,
                messages=messages,
            )
        else:
            create_kwargs, extra_body = split_model_args(model_args)
            # create_kwargs already has model if provided

        call_kwargs = {
            **create_kwargs,
            "messages": messages,
            "stream": False,  # 对齐 KyOpenAIServer.chat
        }
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        return self.client.chat.completions.create(**call_kwargs)

    @Cache(cache_dir="cache/OpenAI", cache_name="llm_cache")
    def generate(self, prompt, model_args, **kwargs):
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        response = {}
        completion = None

        try:
            completion = self._create(messages, model_args)
            response["status"] = 1
            msg = completion.choices[0].message
            response["answer"] = _content_from_message(msg, local_mode=self.local_mode)
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                response["reasoning_content"] = reasoning
        except Exception as e:
            response["status"] = 0
            response["answer"] = str(e)

        response.update(_usage_from_completion(completion))
        return response

    def chat(
        self,
        messages: list,
        model_args: dict,
        *,
        tools: Optional[list] = None,
        tool_choice: Optional[Any] = None,
    ) -> dict:
        """
        多轮 chat，可选 OpenAI function tools。
        不走 generate 缓存（轨迹含工具结果，键不稳定）。
        """
        ma = dict(model_args or {})
        if tools:
            ma["tools"] = tools
            if tool_choice is None:
                tool_choice = "auto"
        if tool_choice is not None:
            ma["tool_choice"] = tool_choice

        response: Dict[str, Any] = {
            "status": 0,
            "answer": "",
            "tool_calls": [],
            "assistant_message": None,
        }
        completion = None
        try:
            completion = self._create(list(messages or []), ma)
            response["status"] = 1
            msg = completion.choices[0].message
            content = _content_from_message(msg, local_mode=self.local_mode)
            response["answer"] = content
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                response["reasoning_content"] = reasoning
            tool_calls = _tool_calls_from_message(msg)
            response["tool_calls"] = tool_calls
            asst: Dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                asst["tool_calls"] = tool_calls
            response["assistant_message"] = asst
        except Exception as e:
            response["status"] = 0
            response["answer"] = str(e)
            response["error"] = str(e)

        response.update(_usage_from_completion(completion))
        return response

    def generate_vision(self, prompt, images, model_args, **kwargs):
        """
        Multimodal generation: text prompt + list of image payloads (single turn).

        images: list of dicts, each either:
          - {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
          - or raw base64 string (will be wrapped as png data url)
        prompt: {"system": str, "user": str}
        """
        user_content = [{"type": "text", "text": prompt.get("user", "")}]
        for img in images or []:
            if isinstance(img, dict) and img.get("type") == "image_url":
                user_content.append(img)
            elif isinstance(img, dict) and "url" in img:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": img["url"]},
                    }
                )
            elif isinstance(img, str):
                url = img if img.startswith("data:") else f"data:image/png;base64,{img}"
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )
            else:
                raise ValueError(f"Unsupported image payload type: {type(img)}")

        messages = []
        system = prompt.get("system") or ""
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})

        response = {}
        completion = None
        try:
            completion = self._create(messages, model_args)
            response["status"] = 1
            msg = completion.choices[0].message
            response["answer"] = _content_from_message(msg, local_mode=self.local_mode)
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                response["reasoning_content"] = reasoning
        except Exception as e:
            response["status"] = 0
            response["answer"] = str(e)

        response.update(_usage_from_completion(completion))
        return response

def normalize_embedding_model_args(model_args: Optional[dict] = None) -> dict:
    """
    规范化 embedding model_args，供 API 与缓存键共用。

    维度别名统一为 OpenAI / xinference 标准字段 ``dimensions``：
      dimension / dim / dims  →  dimensions
    若同时写了 dimensions 与别名，以 ``dimensions`` 为准。
    """
    ma = dict(model_args or {})
    dim_val = ma.get("dimensions")
    if dim_val is None:
        for k in ("dimension", "dim", "dims"):
            if ma.get(k) is not None:
                dim_val = ma.get(k)
                break
    for k in ("dimension", "dim", "dims"):
        ma.pop(k, None)
    if dim_val is not None:
        try:
            d = int(dim_val)
            if d > 0:
                ma["dimensions"] = d
            else:
                ma.pop("dimensions", None)
        except (TypeError, ValueError):
            ma.pop("dimensions", None)
    return ma

class Embedding:
    """
    文本向量。

    - 本地 modelscope（默认，model_args 无 model）：
      对齐 NlpModelServer.sentence_embedding
      POST {base}/nlp/sentence_embedding（失败再试 {base}/sentence_embedding）
      body: source_sentence + access_token + timestamp
    - OpenAI 兼容：当 model_args 含 model 时走 /v1/embeddings。

    model_args 维度字段：
      dimensions（标准）/ dimension / dim / dims 均可；调用前统一为 dimensions。

    超时说明：
      OpenAI SDK 默认 connect timeout 仅约 5s，内网抖动/服务端排队时
      极易出现 "Request timed out."。此处默认拉长超时并做有限重试。
    """

    def __init__(
        self,
        api_key,
        base_url,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_wait: float = 0.5,
    ):
        self.api_key = api_key or "EMPTY"
        # 兼容配置里带 /v1 的 OpenAI 风格地址；本地 route API 用根路径
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = float(timeout) if timeout is not None else 120.0
        self.max_retries = max(1, int(max_retries) if max_retries is not None else 3)
        self.retry_wait = max(0.0, float(retry_wait) if retry_wait is not None else 0.5)
        self._openai_client = None

    def _normalize_local_base(self) -> str:
        """去掉末尾 /v1，得到 modelscope 服务根地址。"""
        base = self.base_url
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        return base.rstrip("/")

    def _openai(self) -> OpenAI:
        if self._openai_client is None:
            # 显式 timeout：避免默认 connect=5s 导致偶发 Request timed out
            # max_retries=0：超时重试由本类控制，避免 SDK 与业务双重重试
            try:
                import httpx

                t = float(self.timeout)
                timeout = httpx.Timeout(
                    connect=min(30.0, t),
                    read=t,
                    write=t,
                    pool=min(30.0, t),
                )
            except Exception:
                timeout = float(self.timeout)
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
                max_retries=0,
            )
        return self._openai_client

    def _make_sign(self) -> dict:
        """对齐 ModelServer.make_sign（无 secret 时仅 timestamp + access_token）。"""
        return {
            "timestamp": str(int(time.time())),
            "access_token": self.api_key,
        }

    def _parse_embedding_payload(self, data: Any) -> Optional[list]:
        """
        兼容两种响应：
          1) NlpModelServer: {code:200, result:{text_embedding:...}, msg:...}
          2) 扁平: {text_embedding:...}
        """
        if not isinstance(data, dict):
            return None
        if "result" in data and isinstance(data["result"], dict):
            emb = data["result"].get("text_embedding")
            if emb is not None:
                return emb
        emb = data.get("text_embedding")
        if emb is not None:
            return emb
        return None

    def sentence_embedding(self, text: str):
        """
        本地文本向量（单条）。
        POST sentence_embedding，source_sentence=[text] → text_embedding[0]
        """
        vecs = self.batch_sentence_embedding([text] if text is not None else [])
        return vecs[0] if vecs else []

    def batch_sentence_embedding(self, text: list):
        """
        本地文本向量（批量）。对齐 NlpModelServer.batch_sentence_embedding。

        优先 POST {base}/nlp/sentence_embedding（NlpModelServer 默认路径），
        若 404 再回退 {base}/sentence_embedding。
        token 放入 JSON body（非仅 header）。
        """
        if not text:
            return []
        import httpx

        base = self._normalize_local_base()
        # NlpModelServer: prefix = host + "nlp/"
        candidates = [
            f"{base}/nlp/sentence_embedding",
            f"{base}/sentence_embedding",
        ]
        payload = dict(self._make_sign())
        payload["source_sentence"] = text
        headers = {
            "access_token": self.api_key,  # 部分网关仍读 header
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None
        with httpx.Client(timeout=120.0) as client:
            for url in candidates:
                try:
                    result = client.post(url, json=payload, headers=headers)
                    # 404 → 换路径；其它 HTTP 错误直接抛
                    if result.status_code == 404:
                        last_err = httpx.HTTPStatusError(
                            f"404 for {url}",
                            request=result.request,
                            response=result,
                        )
                        continue
                    result.raise_for_status()
                    data = result.json() if result.content else {}
                    # NlpModelServer 还会检查 code==200
                    if isinstance(data, dict) and "code" in data and data.get("code") != 200:
                        msg = data.get("msg") or data.get("message") or data
                        raise ValueError(f"{url} post error: {msg}")
                    embedding = self._parse_embedding_payload(data)
                    if embedding is not None:
                        return embedding
                    raise ValueError(f"{url} missing text_embedding: {str(data)[:200]}")
                except httpx.HTTPStatusError as e:
                    last_err = e
                    if e.response is not None and e.response.status_code == 404:
                        continue
                    raise
                except Exception as e:
                    last_err = e
                    raise
        if last_err:
            raise last_err
        return []

    def _generate_local(self, prompt):
        """本地 modelscope：无需 model 名。"""
        response = {}
        try:
            if isinstance(prompt, list):
                emb = self.batch_sentence_embedding(prompt)
                if not emb:
                    response["status"] = 0
                    response["answer"] = "empty embedding"
                else:
                    response["status"] = 1
                    # 与单条接口一致：单元素 list 也返回整批；调用方取 [0] 或整表
                    response["answer"] = emb[0] if len(emb) == 1 else emb
            else:
                emb = self.sentence_embedding(prompt)
                if not emb:
                    response["status"] = 0
                    response["answer"] = "empty embedding"
                else:
                    response["status"] = 1
                    response["answer"] = emb
        except Exception as e:
            response["status"] = 0
            response["answer"] = str(e)

        response["usage_prompt_tokens"] = None
        response["usage_completion_tokens"] = None
        response["usage_total_tokens"] = None
        return response

    @staticmethod
    def _is_retryable_emb_error(exc: BaseException) -> bool:
        """超时 / 连接 / 5xx 等可重试。"""
        name = type(exc).__name__
        msg = str(exc).lower()
        if "timeout" in name.lower() or "timed out" in msg or "timeout" in msg:
            return True
        if "connection" in name.lower() or "connect" in msg:
            return True
        if "503" in msg or "502" in msg or "504" in msg or "429" in msg:
            return True
        return False

    def _generate_openai(self, prompt, model_args):
        """
        OpenAI 兼容 embeddings.create（需 model_args.model）；超时自动重试。

        prompt 可为 str（单条）或 list[str]（批量）。
        - 单条：answer 为 list[float]
        - 批量：answer 为 list[list[float]]，顺序与 input 一致（按 data.index 排序）
        """
        response = {}
        embedding = None
        is_batch = isinstance(prompt, list)
        # 已规范化：dimensions 可直接进 create kwargs
        create_kwargs, extra_body = split_model_args(model_args)
        call_kwargs = {
            **create_kwargs,
            "input": prompt,
        }
        if extra_body:
            call_kwargs["extra_body"] = extra_body

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                embedding = self._openai().embeddings.create(**call_kwargs)
                response["status"] = 1
                if is_batch:
                    # OpenAI/xinference 可能乱序返回；按 index 还原
                    items = sorted(embedding.data, key=lambda d: int(getattr(d, "index", 0) or 0))
                    response["answer"] = [d.embedding for d in items]
                    if len(response["answer"]) != len(prompt):
                        raise ValueError(
                            f"batch embedding size mismatch: "
                            f"got {len(response['answer'])} want {len(prompt)}"
                        )
                else:
                    response["answer"] = embedding.data[0].embedding
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < self.max_retries and self._is_retryable_emb_error(e):
                    # 退避：0.5s, 1s, 1.5s...
                    time.sleep(self.retry_wait * attempt)
                    continue
                response["status"] = 0
                response["answer"] = str(e)
                break

        if last_err is not None and response.get("status") != 1:
            response["status"] = 0
            response["answer"] = str(last_err)

        try:
            response["usage_prompt_tokens"] = embedding.usage.prompt_tokens
            response["usage_total_tokens"] = embedding.usage.total_tokens
            response["usage_completion_tokens"] = 0
        except Exception:
            response["usage_prompt_tokens"] = None
            response["usage_completion_tokens"] = None
            response["usage_total_tokens"] = None

        return response

    def generate(self, prompt, model_args=None, **kwargs):
        """
        统一入口。model_args 无 model 时走本地 sentence_embedding（不必设模型名）；
        有 model 时走 OpenAI 兼容接口。

        先规范化 dimension/dim/dims → dimensions，再进缓存键与 API，
        避免「写了 dimension 但服务端不认 / 缓存键不一致」。

        单条文本请用本方法；多条请用 generate_batch（按条缓存 + 一次 API）。
        """
        model_args = normalize_embedding_model_args(model_args)
        if isinstance(prompt, list):
            # 兼容误传 list：走 batch，再按单条接口形状返回
            rows = self.generate_batch(prompt, model_args=model_args, **kwargs)
            if len(rows) == 1:
                return rows[0]
            # 多条：拼成与旧本地 batch 类似的聚合响应
            if any(r.get("status") != 1 for r in rows):
                bad = next(r for r in rows if r.get("status") != 1)
                return bad
            return {
                "status": 1,
                "answer": [r["answer"] for r in rows],
                "usage_prompt_tokens": sum(
                    (r.get("usage_prompt_tokens") or 0) for r in rows
                    if not r.get("_cache_hit")
                ) or None,
                "usage_completion_tokens": 0,
                "usage_total_tokens": sum(
                    (r.get("usage_total_tokens") or 0) for r in rows
                    if not r.get("_cache_hit")
                ) or None,
                "_cache_hit": all(r.get("_cache_hit") for r in rows),
            }
        return self._generate_cached(prompt, model_args, **kwargs)

    def generate_batch(self, prompts, model_args=None, **kwargs):
        """
        批量向量化：一次 API 请求嵌入多条文本；缓存仍按「单条」键读写，
        与 generate() 完全兼容（续跑/半缓存批次只请求 miss 的子集）。

        Returns:
            list[dict] — 与 generate() 单条响应同结构，长度 == len(prompts)
        """
        import hashlib
        import json

        model_args = normalize_embedding_model_args(model_args)
        use_cache = kwargs.get("use_cache", True)
        if prompts is None:
            return []
        if not isinstance(prompts, list):
            return [self.generate(prompts, model_args=model_args, **kwargs)]
        if len(prompts) == 0:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], model_args=model_args, **kwargs)]

        # 与 @Cache 装饰的 _generate_cached 键一致：{prompt, model_args}
        if not hasattr(self, "_emb_cache") or self._emb_cache is None:
            self._emb_cache = Cache(cache_dir="cache/OpenAI", cache_name="emb_cache")
        cache = self._emb_cache

        n = len(prompts)
        results: List[Optional[dict]] = [None] * n
        miss_idx: List[int] = []
        miss_prompts: List[Any] = []
        miss_hashes: List[str] = []
        miss_inp_json: List[str] = []

        for i, p in enumerate(prompts):
            if use_cache:
                inp = {"prompt": p, "model_args": model_args or {}}
                inp_json = json.dumps(inp, ensure_ascii=False, indent=2)
                inp_hash = hashlib.md5(inp_json.encode("utf-8")).hexdigest()
                out_json = cache.cache_db.search_cache(inp_hash)
                if out_json is not None:
                    try:
                        out = json.loads(out_json)["output"]
                        if isinstance(out, dict):
                            out = dict(out)
                            out["_cache_hit"] = True
                            results[i] = out
                            continue
                    except Exception:
                        pass
                miss_hashes.append(inp_hash)
                miss_inp_json.append(inp_json)
            else:
                miss_hashes.append("")
                miss_inp_json.append("")
            miss_idx.append(i)
            miss_prompts.append(p)

        if miss_prompts:
            if model_args.get("model"):
                api = self._generate_openai(miss_prompts, model_args)
            else:
                api = self._generate_local(miss_prompts)

            if api.get("status") != 1:
                err = {
                    "status": 0,
                    "answer": api.get("answer"),
                    "usage_prompt_tokens": None,
                    "usage_completion_tokens": None,
                    "usage_total_tokens": None,
                    "_cache_hit": False,
                }
                for i in miss_idx:
                    results[i] = dict(err)
            else:
                answers = api.get("answer")
                if not isinstance(answers, list) or (
                    answers and not isinstance(answers[0], (list, tuple))
                ):
                    # 本地单元素或异常形状：尽量兜底
                    if isinstance(answers, list) and answers and isinstance(answers[0], (int, float)):
                        answers = [answers]
                    else:
                        answers = [answers] * len(miss_prompts)

                n_miss = len(miss_prompts)
                # 均分 usage 到每条（仅用于 metrics；缓存里各存一份）
                pt = api.get("usage_prompt_tokens")
                tt = api.get("usage_total_tokens")
                try:
                    pt_each = int(pt) // n_miss if pt is not None else None
                    tt_each = int(tt) // n_miss if tt is not None else None
                    # 余数补给第一条，避免总和漂移
                    pt_rem = (int(pt) - pt_each * n_miss) if pt is not None and pt_each is not None else 0
                    tt_rem = (int(tt) - tt_each * n_miss) if tt is not None and tt_each is not None else 0
                except (TypeError, ValueError):
                    pt_each = tt_each = None
                    pt_rem = tt_rem = 0

                for j, i in enumerate(miss_idx):
                    emb = answers[j] if j < len(answers) else None
                    if emb is None:
                        results[i] = {
                            "status": 0,
                            "answer": "missing embedding in batch response",
                            "usage_prompt_tokens": None,
                            "usage_completion_tokens": None,
                            "usage_total_tokens": None,
                            "_cache_hit": False,
                        }
                        continue
                    up = (pt_each + (pt_rem if j == 0 else 0)) if pt_each is not None else None
                    ut = (tt_each + (tt_rem if j == 0 else 0)) if tt_each is not None else None
                    out = {
                        "status": 1,
                        "answer": emb,
                        "usage_prompt_tokens": up,
                        "usage_completion_tokens": 0,
                        "usage_total_tokens": ut,
                        "_cache_hit": False,
                    }
                    results[i] = out
                    if use_cache and Cache._should_store(out):
                        to_store = {
                            k: v for k, v in out.items() if not str(k).startswith("_")
                        }
                        payload = json.dumps({"output": to_store}, ensure_ascii=False, indent=2)
                        try:
                            cache.cache_db.update_cache(
                                miss_inp_json[j], payload, miss_hashes[j]
                            )
                        except Exception:
                            pass

        # 理论上无 None；兜底
        for i in range(n):
            if results[i] is None:
                results[i] = {
                    "status": 0,
                    "answer": "internal: empty batch slot",
                    "usage_prompt_tokens": None,
                    "usage_completion_tokens": None,
                    "usage_total_tokens": None,
                    "_cache_hit": False,
                }
        return results  # type: ignore[return-value]

    @Cache(cache_dir="cache/OpenAI", cache_name="emb_cache")
    def _generate_cached(self, prompt, model_args=None, **kwargs):
        model_args = model_args or {}
        if model_args.get("model"):
            return self._generate_openai(prompt, model_args)
        return self._generate_local(prompt)

class Reranker:
    """
    文本重排（xinference / OpenAI 兼容 /v1/rerank）。

    典型用法::

        rr = Reranker(api_key='EMPTY', base_url='http://host:9998/v1')
        ranked = rr.rerank(
            query='ABS 722 用途',
            documents=['文档全文1', '文档全文2'],
            model='Qwen3-Reranker-0.6B',
            top_n=4,
        )
        # → [{'index': 1, 'relevance_score': 0.91}, ...]  按分数降序
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "",
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_wait: float = 0.5,
    ):
        self.api_key = api_key or "EMPTY"
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = float(timeout) if timeout is not None else 120.0
        self.max_retries = max(1, int(max_retries) if max_retries is not None else 3)
        self.retry_wait = max(0.0, float(retry_wait) if retry_wait is not None else 0.5)

    def _endpoint_candidates(self) -> List[str]:
        """
        解析重排 URL 候选。
          - base 已含 /v1 → {base}/rerank
          - 否则 → {base}/v1/rerank，再回退 {base}/rerank
        """
        base = self.base_url.rstrip("/")
        if not base:
            return []
        if base.endswith("/v1"):
            return [f"{base}/rerank"]
        return [f"{base}/v1/rerank", f"{base}/rerank"]

    @staticmethod
    def _parse_results(data: Any) -> List[Dict[str, Any]]:
        """
        兼容：
          {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
          {"data":    [{"index": 0, "score": 0.9}, ...]}
          直接 list
        """
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("results") or data.get("data") or data.get("result") or []
            if isinstance(raw, dict):
                raw = raw.get("results") or raw.get("data") or []
        else:
            return []

        out: List[Dict[str, Any]] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if idx is None:
                continue
            score = item.get("relevance_score")
            if score is None:
                score = item.get("score")
            if score is None:
                score = item.get("relevance")
            try:
                idx = int(idx)
                score = float(score) if score is not None else 0.0
            except Exception:
                continue
            out.append({"index": idx, "relevance_score": score})
        out.sort(key=lambda x: -x["relevance_score"])
        return out

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        model: str = "Qwen3-Reranker-0.6B",
        top_n: Optional[int] = None,
        model_args: Optional[dict] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        对 documents 按与 query 的相关性重排。

        返回按 relevance_score 降序的列表::
            [{"index": 原下标, "relevance_score": float}, ...]
        top_n 非空时服务端/本地截断到前 N；失败抛异常（由调用方决定回退）。
        """
        docs = [str(d) if d is not None else "" for d in (documents or [])]
        if not docs:
            return []
        q = (query or "").strip()
        if not q:
            # 无 query 时保持原序
            n = len(docs) if top_n is None else min(int(top_n), len(docs))
            return [{"index": i, "relevance_score": 0.0} for i in range(n)]

        ma = dict(model_args or {})
        model_name = (ma.get("model") or model or "Qwen3-Reranker-0.6B").strip()
        payload: Dict[str, Any] = {
            "model": model_name,
            "query": q,
            "documents": docs,
        }
        if top_n is not None:
            payload["top_n"] = max(1, int(top_n))
        # 透传其余 model_args（如 instruct），避开 model 键
        for k, v in ma.items():
            if k == "model" or k in payload or v is None:
                continue
            payload[k] = v

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # 部分网关读 access_token
        if self.api_key:
            headers["access_token"] = self.api_key

        import httpx

        endpoints = self._endpoint_candidates()
        if not endpoints:
            raise ValueError("Reranker base_url is empty")

        last_err: Optional[Exception] = None
        t = float(self.timeout)
        try:
            timeout = httpx.Timeout(
                connect=min(30.0, t), read=t, write=t, pool=min(30.0, t)
            )
        except Exception:
            timeout = t

        with httpx.Client(timeout=timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                for url in endpoints:
                    try:
                        resp = client.post(url, json=payload, headers=headers)
                        if resp.status_code == 404:
                            last_err = httpx.HTTPStatusError(
                                f"404 for {url}",
                                request=resp.request,
                                response=resp,
                            )
                            continue
                        resp.raise_for_status()
                        data = resp.json() if resp.content else {}
                        ranked = self._parse_results(data)
                        if not ranked:
                            raise ValueError(
                                f"{url} empty rerank results: {str(data)[:200]}"
                            )
                        if top_n is not None:
                            ranked = ranked[: max(1, int(top_n))]
                        return ranked
                    except httpx.HTTPStatusError as e:
                        last_err = e
                        if e.response is not None and e.response.status_code == 404:
                            continue
                        break  # 非 404：换 attempt 重试
                    except Exception as e:
                        last_err = e
                        break
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait * attempt)

        raise RuntimeError(f"Reranker failed after {self.max_retries} attempts: {last_err}")

    def rank_documents(
        self,
        query: str,
        documents: Sequence[str],
        *,
        model: str = "Qwen3-Reranker-0.6B",
        top_n: Optional[int] = None,
        model_args: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        便捷封装：返回带原文的排序结果::
            [{"index", "relevance_score", "document"}, ...]
        """
        ranked = self.rerank(
            query, documents, model=model, top_n=top_n, model_args=model_args
        )
        docs = list(documents or [])
        out = []
        for item in ranked:
            idx = item["index"]
            out.append({
                "index": idx,
                "relevance_score": item["relevance_score"],
                "document": docs[idx] if 0 <= idx < len(docs) else "",
            })
        return out
