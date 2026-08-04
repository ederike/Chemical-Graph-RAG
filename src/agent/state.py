"""Agent 运行时状态与步骤数据结构。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class PlanStep(TypedDict, total=False):
    """规划 LLM 输出的单个单跳步骤。"""
    id: str
    question: str
    depends_on: List[str]


class StepResult(TypedDict, total=False):
    """单跳 skill 执行结果。"""
    id: str
    planned_question: str
    resolved_question: str
    answer: str
    sources: List[str]
    status: int
    retrieve_latency_s: float
    latency_s: float
    usage_prompt_tokens: Optional[int]
    usage_completion_tokens: Optional[int]
    usage_total_tokens: Optional[int]


class AgentState(TypedDict, total=False):
    """
    LangGraph 状态。

    流程：plan → execute(循环) → synthesize
    """
    query: str
    plan: List[PlanStep]
    results: Dict[str, StepResult]  # step_id → result
    final_answer: str
    final_status: int
    # 汇总阶段的原始 LLM 响应（与 DHMF.query 对齐字段）
    respond: Dict[str, Any]
    error: str
