"""
同一条 messages 上的 think → tool → observe 循环。

模型自己决定何时 search / read_doc / graph_neighbors，何时给出最终答案。
优先 OpenAI function calling；接口不支持或 tool_protocol=json 时走 JSON 动作。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, TYPE_CHECKING

from ..utils.OpenAIAPI import LLM
from ..utils.config import resolve_credentials, resolve_llm_timeout
from .prompts import AGENTIC_PROMPT
from .tools import ToolContext, allowed_tool_names, tool_schemas

if TYPE_CHECKING:
    from ..DHMF import DHMF
    from ..utils.config import AgenticConfig

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_TOOLS_ERR_MARKERS = (
    "tool",
    "function",
    "tools",
    "tool_choice",
    "does not support",
    "unknown field",
    "extra_forbidden",
    "not supported",
)


def build_agentic_llm(config) -> LLM:
    agentic_cfg = getattr(config, "agentic", None)
    api_key, base_url = resolve_credentials(config, agentic_cfg)
    return LLM(api_key, base_url, timeout=resolve_llm_timeout(agentic_cfg))


def _add_usage(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None and b is None:
        return None
    return int(a or 0) + int(b or 0)


def _usage_from(resp: Optional[dict]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not isinstance(resp, dict):
        return None, None, None
    return (
        resp.get("usage_prompt_tokens"),
        resp.get("usage_completion_tokens"),
        resp.get("usage_total_tokens"),
    )


def _looks_like_tools_unsupported(err: str) -> bool:
    s = (err or "").lower()
    if not s:
        return False
    return any(m in s for m in _TOOLS_ERR_MARKERS)


def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_action(text: str) -> Optional[dict]:
    """
    解析 JSON 动作。
      {"thought", "tool", "arguments"} 或 {"thought", "answer"}
    失败返回 None。
    """
    raw = _strip_fence(text)
    if not raw:
        return None
    data = None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            data = obj
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            data = obj if isinstance(obj, dict) else None
        except Exception:
            return None
    if not data:
        return None

    thought = str(data.get("thought") or data.get("reasoning") or "").strip()
    if data.get("answer") is not None and not data.get("tool"):
        ans = data.get("answer")
        if not isinstance(ans, str):
            ans = json.dumps(ans, ensure_ascii=False)
        return {"kind": "answer", "thought": thought, "answer": str(ans).strip()}

    tool = data.get("tool") or data.get("name") or data.get("function")
    if isinstance(tool, dict):
        name = tool.get("name") or ""
        arguments = tool.get("arguments") if "arguments" in tool else tool.get("args")
    else:
        name = str(tool or "").strip()
        arguments = data.get("arguments") if "arguments" in data else data.get("args")
    if name:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {"query": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        return {
            "kind": "tool",
            "thought": thought,
            "name": name,
            "arguments": arguments,
        }

    calls = data.get("tool_calls")
    if isinstance(calls, list) and calls:
        first = calls[0] if isinstance(calls[0], dict) else {}
        fn = first.get("function") if isinstance(first.get("function"), dict) else first
        name = str((fn or {}).get("name") or "").strip()
        arguments = (fn or {}).get("arguments") or (fn or {}).get("args") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {"query": arguments}
        if name:
            return {
                "kind": "tool",
                "thought": thought,
                "name": name,
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
    return None


class TraceSink:
    """log_trace 全文写入独立文件，不刷控制台。"""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._fp: Optional[TextIO] = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = open(self.path, "w", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._fp is not None

    def line(self, msg: str) -> None:
        if self._fp is None:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._fp.write(f"{ts}  {msg}\n")
        self._fp.flush()

    def block(self, title: str, body: str) -> None:
        if self._fp is None:
            return
        self.line(f"----- {title} -----")
        text = "" if body is None else str(body)
        self._fp.write(text)
        if text and not text.endswith("\n"):
            self._fp.write("\n")
        self.line("----- end -----")
        self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:
                pass
            self._fp = None


def open_trace_file(working_path: str) -> Path:
    log_dir = Path(working_path) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = log_dir / f"agentic_{ts}.log"
    n = 1
    while path.exists():
        n += 1
        path = log_dir / f"agentic_{ts}_{n}.log"
    return path


@dataclass
class AgenticContext:
    cfg: "AgenticConfig"
    llm: LLM
    tools: ToolContext
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    trace: TraceSink = field(default_factory=TraceSink)

    def _trace(self, msg: str) -> None:
        if self.trace.enabled:
            self.trace.line(msg)
            return
        self.logger.debug(f"[agentic.trace] {msg}")

    def _trace_block(self, title: str, body: str) -> None:
        if self.trace.enabled:
            self.trace.block(title, body)
            return
        preview = (body or "").replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "…"
        self.logger.info(f"[agentic] {title}: {preview}")


def _system_prompt(protocol: str) -> str:
    if protocol == "json":
        return AGENTIC_PROMPT["SYSTEM_JSON"]
    return AGENTIC_PROMPT["SYSTEM_TOOLS"]


def _message_chars(msg: dict) -> int:
    if not isinstance(msg, dict):
        return len(str(msg or ""))
    n = 0
    content = msg.get("content")
    if isinstance(content, str):
        n += len(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                n += len(str(part.get("text") or part.get("content") or ""))
            else:
                n += len(str(part))
    if msg.get("tool_calls"):
        try:
            n += len(json.dumps(msg.get("tool_calls"), ensure_ascii=False))
        except Exception:
            n += len(str(msg.get("tool_calls")))
    return n


def estimate_prompt_tokens(messages: list, tools: Optional[list] = None) -> int:
    """无 usage 时按字符粗估。中英混合约 2 字/token，偏保守以便提前停。"""
    n = 0
    for m in messages or []:
        n += _message_chars(m if isinstance(m, dict) else {"content": str(m)})
    if tools:
        try:
            n += len(json.dumps(tools, ensure_ascii=False))
        except Exception:
            pass
    return max(1, n // 2)


def last_prompt_tokens(resp: Optional[dict], messages: list, tools: Optional[list] = None) -> int:
    if isinstance(resp, dict):
        raw = resp.get("usage_prompt_tokens")
        try:
            n = int(raw)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    return estimate_prompt_tokens(messages, tools)


def _openai_tool_calls_to_actions(resp: dict) -> List[dict]:
    actions = []
    thought = str(resp.get("answer") or "").strip()
    reasoning = str(resp.get("reasoning_content") or "").strip()
    if reasoning and not thought:
        thought = reasoning
    for tc in resp.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                args_obj = json.loads(args) if args else {}
            except Exception:
                args_obj = {"_raw": args}
        elif isinstance(args, dict):
            args_obj = args
        else:
            args_obj = {}
        actions.append({
            "kind": "tool",
            "id": str((tc or {}).get("id") or ""),
            "name": str(fn.get("name") or ""),
            "arguments": args_obj,
            "thought": thought,
            "reasoning": reasoning,
        })
    return actions


def run_agentic_loop(ctx: AgenticContext, query: str) -> dict:
    cfg = ctx.cfg
    q = (query or "").strip()
    max_turns = int(cfg.max_turns)
    protocol = cfg.tool_protocol if cfg.tool_protocol in ("openai", "json") else "auto"
    use_openai = protocol != "json"
    schemas = tool_schemas(cfg) if use_openai else []
    allowed = set(allowed_tool_names(cfg))

    messages: List[dict] = [
        {"role": "system", "content": _system_prompt("openai" if use_openai else "json")},
        {"role": "user", "content": q},
    ]

    turns: List[dict] = []
    pt = ct = tt = None
    last_reasoning = ""
    last_prompt = 0
    budget = int(getattr(cfg, "max_prompt_tokens", 0) or 0)
    reserve = int(getattr(cfg, "prompt_token_reserve", 0) or 0)
    token_threshold = (budget - reserve) if budget > 0 else 0
    if token_threshold < 0:
        token_threshold = 0

    def _accumulate(resp: dict) -> None:
        nonlocal pt, ct, tt, last_prompt
        a, b, c = _usage_from(resp)
        pt = _add_usage(pt, a)
        ct = _add_usage(ct, b)
        tt = _add_usage(tt, c)
        last_prompt = last_prompt_tokens(
            resp, messages, schemas if use_openai else None,
        )

    def _over_token_budget(n: Optional[int] = None) -> bool:
        if token_threshold <= 0:
            return False
        return int(n if n is not None else last_prompt) >= token_threshold

    def _finish(ok: bool, answer: str, reasoning: str = "") -> dict:
        if ok:
            out = _ok(
                answer, turns, ctx, pt, ct, tt, protocol,
                reasoning or last_reasoning,
            )
        else:
            out = _fail(answer, turns, ctx, pt, ct, tt, protocol)
        out["last_prompt_tokens"] = last_prompt
        out["max_prompt_tokens"] = budget or None
        out["prompt_token_threshold"] = token_threshold or None
        return out

    def _force_final(reason: str, turn_tag) -> dict:
        nonlocal last_reasoning
        ctx._trace(f"forcing final answer: {reason} last_prompt={last_prompt}")
        if (not cfg.force_answer_on_max_turns) and reason.startswith("max_turns"):
            return _finish(False, f"超过 max_turns={max_turns} 仍未给出答案")
        messages.append({
            "role": "user",
            "content": AGENTIC_PROMPT["FORCE_ANSWER"] + f"\n（原因：{reason}）",
        })
        resp = _chat(with_tools=False)
        _accumulate(resp)
        answer = str(resp.get("answer") or "").strip()
        parsed = parse_json_action(answer)
        if parsed and parsed.get("kind") == "answer":
            answer = parsed.get("answer") or answer
        reasoning = str(resp.get("reasoning_content") or "").strip()
        if reasoning:
            last_reasoning = reasoning
            ctx._trace_block("force reasoning", reasoning)
        turns.append({
            "turn": turn_tag,
            "kind": "answer",
            "thought": reasoning,
            "answer": answer,
            "forced": True,
            "force_reason": reason,
        })
        ctx._trace_block("final_answer", answer)
        return _finish(True, answer, reasoning=last_reasoning)

    def _chat(*, with_tools: bool) -> dict:
        if with_tools and use_openai and schemas:
            return ctx.llm.chat(
                messages,
                dict(cfg.model_args or {}),
                tools=schemas,
                tool_choice="auto",
            )
        ma = dict(cfg.model_args or {})
        ma.pop("tools", None)
        ma.pop("tool_choice", None)
        return ctx.llm.chat(messages, ma)

    def _switch_to_json(reason: str) -> None:
        nonlocal use_openai, protocol, schemas
        if not use_openai:
            return
        use_openai = False
        protocol = "json"
        schemas = []
        messages[0] = {"role": "system", "content": _system_prompt("json")}
        ctx.logger.warning(f"[agentic] fallback to JSON protocol: {reason}")

    for turn_i in range(1, max_turns + 1):
        if turn_i > 1 and _over_token_budget():
            return _force_final(
                f"prompt tokens {last_prompt} >= {token_threshold} "
                f"(budget {budget}, reserve {reserve})",
                turn_i,
            )
        ctx._trace(
            f"===== turn {turn_i}/{max_turns} protocol={'openai' if use_openai else 'json'} "
            f"last_prompt={last_prompt} threshold={token_threshold or '-'} ====="
        )
        resp = _chat(with_tools=use_openai)
        _accumulate(resp)
        ctx._trace(f"usage prompt={last_prompt} completion={resp.get('usage_completion_tokens')}")

        if int(resp.get("status") or 0) != 1:
            err = str(resp.get("error") or resp.get("answer") or "LLM 调用失败")
            ctx._trace(f"llm error: {err}")
            if use_openai and _looks_like_tools_unsupported(err):
                _switch_to_json(err)
                resp = _chat(with_tools=False)
                _accumulate(resp)
                if int(resp.get("status") or 0) != 1:
                    err = str(resp.get("error") or resp.get("answer") or err)
                    return _finish(False, err)
            else:
                return _finish(False, err)

        reasoning = str(resp.get("reasoning_content") or "").strip()
        content = str(resp.get("answer") or "").strip()
        if reasoning:
            last_reasoning = reasoning
            ctx._trace_block(f"turn {turn_i} reasoning", reasoning)
        if content:
            ctx._trace_block(f"turn {turn_i} content", content)

        actions: List[dict] = []
        openai_calls = resp.get("tool_calls") or []
        if use_openai and openai_calls:
            actions = _openai_tool_calls_to_actions(resp)
        else:
            parsed = parse_json_action(content)
            if parsed:
                actions = [parsed]
            elif use_openai and not content:
                return _finish(False, "模型既未调用工具也未作答")
            else:
                answer = content
                turns.append({
                    "turn": turn_i,
                    "kind": "answer",
                    "thought": reasoning,
                    "answer": answer,
                })
                ctx._trace_block("final_answer", answer)
                return _finish(True, answer)

        if len(actions) == 1 and actions[0].get("kind") == "answer":
            answer = actions[0].get("answer") or ""
            thought = actions[0].get("thought") or reasoning
            turns.append({
                "turn": turn_i,
                "kind": "answer",
                "thought": thought,
                "answer": answer,
            })
            if thought:
                ctx._trace_block(f"turn {turn_i} thought", thought)
            ctx._trace_block("final_answer", answer)
            return _finish(True, answer)

        tool_actions = [a for a in actions if a.get("kind") == "tool"]
        if not tool_actions:
            answer = content or ""
            turns.append({
                "turn": turn_i,
                "kind": "answer",
                "thought": reasoning,
                "answer": answer,
            })
            ctx._trace_block("final_answer", answer)
            return _finish(True, answer)

        if _over_token_budget():
            ctx._trace(
                f"skip further tools, prompt {last_prompt} >= {token_threshold}"
            )
            return _force_final(
                f"prompt tokens {last_prompt} >= {token_threshold} "
                f"(budget {budget}, reserve {reserve})",
                turn_i,
            )

        if use_openai and openai_calls and resp.get("assistant_message"):
            messages.append(resp["assistant_message"])
        else:
            messages.append({"role": "assistant", "content": content})

        turn_record = {
            "turn": turn_i,
            "kind": "tools",
            "thought": tool_actions[0].get("thought") or reasoning,
            "calls": [],
        }
        if turn_record["thought"]:
            ctx._trace_block(f"turn {turn_i} thought", turn_record["thought"])

        for i, act in enumerate(tool_actions):
            name = str(act.get("name") or "").strip()
            arguments = act.get("arguments") if isinstance(act.get("arguments"), dict) else {}
            call_id = str(act.get("id") or f"call_{turn_i}_{i}")
            ctx._trace(
                f"tool_call {name} id={call_id} args={json.dumps(arguments, ensure_ascii=False)}"
            )
            if name not in allowed:
                result = json.dumps(
                    {"error": f"未启用的工具 {name!r}", "allowed": sorted(allowed)},
                    ensure_ascii=False,
                )
            else:
                result = ctx.tools.execute(name, arguments)
            ctx._trace_block(f"tool_result {name}", result)
            turn_record["calls"].append({
                "id": call_id,
                "name": name,
                "arguments": arguments,
                "result": result,
            })
            if use_openai and openai_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"工具 {name} 返回：\n{result}",
                })
        turns.append(turn_record)
        est = estimate_prompt_tokens(messages, schemas if use_openai else None)
        if est > last_prompt:
            last_prompt = est
        ctx._trace(f"after tools estimated_prompt={last_prompt}")
        if _over_token_budget():
            return _force_final(
                f"prompt tokens {last_prompt} >= {token_threshold} "
                f"(budget {budget}, reserve {reserve})",
                turn_i + 1,
            )

    return _force_final(f"max_turns={max_turns}", max_turns + 1)


def _pack(
    *,
    status: int,
    answer: str,
    turns: list,
    ctx: AgenticContext,
    pt, ct, tt,
    protocol: str,
    reasoning: str = "",
) -> dict:
    out = {
        "status": status,
        "answer": answer or "",
        "turns": turns,
        "protocol": protocol,
        "retrieval_sources": list(ctx.tools.sources),
        "retrieval_doc_ids": list(ctx.tools.doc_ids),
        "retrieve_latency_s": float(ctx.tools.retrieve_latency_s or 0.0),
        "retrieve_timing": dict(ctx.tools.retrieve_timing or {}),
        "usage_prompt_tokens": pt,
        "usage_completion_tokens": ct,
        "usage_total_tokens": tt,
    }
    if reasoning:
        out["reasoning_content"] = reasoning
    return out


def _ok(answer, turns, ctx, pt, ct, tt, protocol, reasoning="") -> dict:
    return _pack(
        status=1, answer=answer, turns=turns, ctx=ctx,
        pt=pt, ct=ct, tt=tt, protocol=protocol, reasoning=reasoning,
    )


def _fail(err, turns, ctx, pt, ct, tt, protocol) -> dict:
    return _pack(
        status=0, answer=err, turns=turns, ctx=ctx,
        pt=pt, ct=ct, tt=tt, protocol=protocol,
    )
