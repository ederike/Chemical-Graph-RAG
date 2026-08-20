"""
LangGraph 编排：

    START → plan → execute ⟲ → synthesize → END

依赖图在 plan 后自动并行加入纯 LLM 步（depends_on=[]）。
单跳：1 次检索作答 + 纯 LLM 作答 → synthesize 对照两者。
多跳：全部检索子步 + 纯 LLM → synthesize。
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import AgentContext, execute_node, plan_node, route_after_execute, synthesize_node
from .state import AgentState

def build_agent_graph(ctx: AgentContext) -> Any:
    g = StateGraph(AgentState)
    g.add_node('plan',       partial(plan_node,       ctx))
    g.add_node('execute',    partial(execute_node,    ctx))
    g.add_node('synthesize', partial(synthesize_node, ctx))

    g.add_edge(START, 'plan')
    g.add_edge('plan', 'execute')
    g.add_conditional_edges(
        'execute',
        route_after_execute,
        {'execute': 'execute', 'synthesize': 'synthesize'},
    )
    g.add_edge('synthesize', END)
    return g.compile()
