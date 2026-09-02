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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..utils.OpenAIAPI import LLM
from ..utils.config import resolve_credentials
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
    return LLM(api_key, base_url)


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


@dataclass
class AgenticContext:
    cfg: "AgenticConfig"
    llm: LLM
    tools: ToolContext
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def _trace(self, msg: str) -> None:
        if self.cfg.log_trace:
            self.logger.info(f"[agentic.trace] {msg}")
        else:
            self.logger.debug(f"[agentic.trace] {msg}")

    def _trace_block(self, title: str, body: str) -> None:
        if not self.cfg.log_trace:
            preview = (body or "").replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "…"
            self.logger.info(f"[agentic] {title}: {preview}")
            return
        text = body if body is not None else ""
        self.logger.info(f"[agentic.trace] ----- {title} -----")
        for line in str(text).splitlines() or [""]:
            self.logger.info(f"[agentic.trace] {line}")
        self.logger.info("[agentic.trace] ----- end -----")


def _system_prompt(protocol: str) -> str:
    if protocol == "json":
        return AGENTIC_PROMPT["SYSTEM_JSON"]
    return AGENTIC_PROMPT["SYSTEM_TOOLS"]


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

    def _accumulate(resp: dict) -> None:
        nonlocal pt, ct, tt
        a, b, c = _usage_from(resp)
        pt = _add_usage(pt, a)
        ct = _add_usage(ct, b)
        tt = _add_usage(tt, c)

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
        ctx._trace(f"===== turn {turn_i}/{max_turns} protocol={'openai' if use_openai else 'json'} =====")
        resp = _chat(with_tools=use_openai)
        _accumulate(resp)

        if int(resp.get("status") or 0) != 1:
            err = str(resp.get("error") or resp.get("answer") or "LLM 调用失败")
            ctx._trace(f"llm error: {err}")
            if use_openai and _looks_like_tools_unsupported(err):
                _switch_to_json(err)
                resp = _chat(with_tools=False)
                _accumulate(resp)
                if int(resp.get("status") or 0) != 1:
                    err = str(resp.get("error") or resp.get("answer") or err)
                    return _fail(err, turns, ctx, pt, ct, tt, protocol)
            else:
                return _fail(err, turns, ctx, pt, ct, tt, protocol)

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
                return _fail("模型既未调用工具也未作答", turns, ctx, pt, ct, tt, protocol)
            else:
                answer = content
                turns.append({
                    "turn": turn_i,
                    "kind": "answer",
                    "thought": reasoning,
                    "answer": answer,
                })
                ctx._trace_block("final_answer", answer)
                return _ok(answer, turns, ctx, pt, ct, tt, protocol, last_reasoning)

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
            return _ok(answer, turns, ctx, pt, ct, tt, protocol, last_reasoning)

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
            return _ok(answer, turns, ctx, pt, ct, tt, protocol, last_reasoning)

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

    if cfg.force_answer_on_max_turns:
        ctx._trace("max_turns reached, forcing final answer")
        messages.append({"role": "user", "content": AGENTIC_PROMPT["FORCE_ANSWER"]})
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
            "turn": max_turns + 1,
            "kind": "answer",
            "thought": reasoning,
            "answer": answer,
            "forced": True,
        })
        ctx._trace_block("final_answer", answer)
        return _ok(answer, turns, ctx, pt, ct, tt, protocol, last_reasoning)

    return _fail(
        f"超过 max_turns={max_turns} 仍未给出答案",
        turns, ctx, pt, ct, tt, protocol,
    )


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
