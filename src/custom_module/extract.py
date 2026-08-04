from ..module.extract import BaseExtract
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import copy
import time

from ..utils.prompt import PROMPT
from ..utils.utils import Retry


class Extract(BaseExtract):
    """
    原始行为：
      - 成功：status=extract，extra 写入抽取结果
      - 失败：@Retry 耗尽后静默返回 None，不更新该块（内容/status 留空或保持分块时状态）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage_prompt_tokens = 0
        self.usage_completion_tokens = 0
        self.metrics = None  # set by DHMF

    def check_extract(self, text):
        """
        只校验 entities。兼容：
          - {"entities": {...}}
          - [{"entities": {...}}, ...]
          - 直接 {"实体名": "描述", ...}（值全是 str 时视为 entities）
        忽略/丢弃 reference、knowledge（下游不用）。
        返回统一格式: [{"entities": {name: desc, ...}}, ...]
        """
        try:
            res = json.loads(text)
        except Exception:
            try:
                cleaned = (text or '').strip()
                if cleaned.startswith('```'):
                    lines = cleaned.splitlines()
                    if lines and lines[0].startswith('```'):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == '```':
                        lines = lines[:-1]
                    cleaned = '\n'.join(lines)
                res = json.loads(cleaned)
            except Exception:
                return None

        def _norm_entities(ent):
            if not isinstance(ent, dict):
                return None
            if ent and not all(isinstance(k, str) and isinstance(v, str) for k, v in ent.items()):
                return None
            return ent

        items = []
        if isinstance(res, dict):
            if 'entities' in res:
                ent = _norm_entities(res.get('entities'))
                if ent is None:
                    return None
                items = [{'entities': ent}]
            elif res and all(isinstance(v, str) for v in res.values()):
                # 顶层直接是 实体名→描述
                items = [{'entities': res}]
            else:
                return None
        elif isinstance(res, list):
            if not res:
                return None
            for item in res:
                if not isinstance(item, dict):
                    return None
                # 列表项必须带 entities；避免把 {"knowledge":"..."} 误当成实体
                if 'entities' not in item:
                    return None
                ent = _norm_entities(item.get('entities'))
                if ent is None:
                    return None
                items.append({'entities': ent})
        else:
            return None

        return items

    @Retry(max_attempt=3, wait=0.1, timeout=60, config_attr='extract.retry')
    def processing_single_task(self, chunk, **kwargs):
        attempt = kwargs.get('attempt', 1)
        max_attempt = kwargs.get('max_attempt', 3)
        content = chunk['content']
        t0 = time.perf_counter()
        chunk_id = chunk.get('id')

        if attempt > 1:
            self.logger.debug(f"Chunk {chunk_id} began its {attempt} extraction attempt.")

        try:
            prompt_name = self.config.extract.extract_prompt
            if prompt_name not in PROMPT:
                prompt_name = 'extract'
            NPROMPT = PROMPT[prompt_name].format(content=content)

            model_args = copy.deepcopy(self.config.extract.model_args or {})
            model_args.setdefault('enable_thinking', False)
            # DashScope/Qwen JSON Mode：强制合法 JSON（prompt 须含 "JSON"）
            model_args.setdefault('response_format', {'type': 'json_object'})
            if attempt > 1:
                model_args['temperature'] = 1.0

            prompt = {'system': "", 'user': NPROMPT}
            response = self.llmmodel.generate(
                prompt=prompt,
                model_args=model_args,
                attempt=attempt,
                use_cache=self.config.extract.use_cache,
            )
            extract_res = self.check_extract(response.get('answer'))

            cache_hit = bool(response.get('_cache_hit'))
            prompt_tok = response.get('usage_prompt_tokens') or 0
            completion_tok = response.get('usage_completion_tokens') or 0
            total_tok = response.get('usage_total_tokens')
            if total_tok is None:
                total_tok = (prompt_tok or 0) + (completion_tok or 0)

            if not cache_hit:
                self.usage_prompt_tokens += prompt_tok or 0
                self.usage_completion_tokens += completion_tok or 0

            dt = time.perf_counter() - t0
            if self.metrics is not None:
                self.metrics.record(
                    'extract',
                    dt,
                    cache_hit=cache_hit,
                    prompt_tokens=prompt_tok if not cache_hit else 0,
                    completion_tokens=completion_tok if not cache_hit else 0,
                    total_tokens=total_tok if not cache_hit else 0,
                    name=f"chunk_{chunk_id}",
                    extra=f"attempt={attempt}",
                    log=False,
                    accumulate_time=False,
                )

            if extract_res is None:
                # 删掉毒缓存，避免重试永久命中坏 JSON；触发 Retry
                try:
                    self.llmmodel.drop_generate_cache(prompt, model_args)
                except Exception:
                    pass
                ans_preview = (response.get('answer') or '')[:200]
                raise ValueError(
                    f"Extraction JSON invalid (status={response.get('status')}, "
                    f"answer_preview={ans_preview!r})"
                )

            # 保留分块阶段写入的头块/顺序元数据，避免 extract 覆盖后丢失
            prev_meta = {}
            try:
                prev = json.loads(chunk.get('extra') or '{}')
                if isinstance(prev, dict):
                    for k in ('is_head', 'chunk_index', 'role'):
                        if k in prev:
                            prev_meta[k] = prev[k]
            except Exception:
                prev_meta = {}
            name = (chunk.get('name') or '').strip().lower()
            if name == 'head':
                prev_meta.setdefault('is_head', True)
                prev_meta.setdefault('role', 'head')
                prev_meta.setdefault('chunk_index', 0)
            elif name.startswith('body_'):
                prev_meta.setdefault('is_head', False)
                prev_meta.setdefault('role', 'body')
                try:
                    prev_meta.setdefault('chunk_index', int(name.split('_', 1)[1]))
                except Exception:
                    pass
            extra = {
                **prev_meta,
                'extract': {
                    'attempt': attempt,
                    'extract': extract_res,
                    'cost': {
                        'usage_prompt_tokens': response.get('usage_prompt_tokens', None),
                        'usage_completion_tokens': response.get('usage_completion_tokens', None),
                        'usage_total_tokens': response.get('usage_total_tokens', None),
                        'usage_cached_tokens': response.get('usage_cached_tokens', None),
                        'cache_hit': cache_hit,
                    }
                }
            }

            updata_chunk = {
                'id': chunk['id'],
                'status': 'extract',
                'extra': json.dumps(extra, ensure_ascii=False),
            }
            self.chunk_db.buffer.append(updata_chunk)
        except Exception as e:
            # Retry 耗尽后静默返回 None；这里打出原因，避免只见 attempt 日志
            level = self.logger.warning if attempt >= max_attempt else self.logger.debug
            level(
                f"Chunk {chunk_id} extraction attempt {attempt}/{max_attempt} failed: "
                f"{type(e).__name__}: {e}"
            )
            raise

    def processing(self):
        """Progress bar only; per-task metrics stay silent; wall-clock at end."""
        n = len(self.tasks)
        if n == 0:
            return

        t_wall = time.perf_counter()

        def _postfix():
            if self.metrics is None:
                return {}
            s = self.metrics.stage_snapshot('extract')
            return {
                'real': s['real'],
                'tok': s['tokens'],
                'tot_s': f"{s['total_s']:.1f}s",
            }

        if self.config.extract.num_thread <= 1:
            bar = tqdm(self.tasks, desc='extract', unit='chunk')
            for task in bar:
                self.processing_single_task(task)
                bar.set_postfix(**_postfix())
        else:
            with ThreadPoolExecutor(max_workers=self.config.extract.num_thread) as executor:
                futures = [
                    executor.submit(self.processing_single_task, task)
                    for task in self.tasks
                ]
                bar = tqdm(as_completed(futures), total=n, desc='extract', unit='chunk')
                for future in bar:
                    future.result()
                    bar.set_postfix(**_postfix())

        if self.metrics is not None:
            self.metrics.finalize_stage_wall_time(
                'extract', time.perf_counter() - t_wall
            )
