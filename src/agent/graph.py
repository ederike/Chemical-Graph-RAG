"""
LangGraph 编排：

    START → plan → execute ⟲ → synthesize → END

execute 通过条件边自循环，直到所有步骤完成。
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import AgentContext, execute_node, plan_node, route_after_execute, synthesize_node
from .state import AgentState


def build_agent_graph(ctx: AgentContext) -> Any:
    """编译可 invoke 的多跳 Agent 图。"""
    g = StateGraph(AgentState)

    g.add_node('plan', partial(plan_node, ctx))
    g.add_node('execute', partial(execute_node, ctx))
    g.add_node('synthesize', partial(synthesize_node, ctx))

    g.add_edge(START, 'plan')
    g.add_edge('plan', 'execute')
    g.add_conditional_edges(
        'execute',
        route_after_execute,
        {
            'execute': 'execute',
            'synthesize': 'synthesize',
        },
    )
    g.add_edge('synthesize', END)

    return g.compile()
