"""
轻量多跳 Agent（LangGraph），位于 src/agent。

对外入口为 DHMF 薄封装::

    dhmf.agent_query("……", pretty=True)

本包内部实现规划 / query_skill / 汇总；业务代码请走 DHMF.agent_query。
"""

from .runner import run_agent_query

__all__ = ['run_agent_query']
