"""
多跳 Agent 包。

对外入口请走 DHMF 薄封装::

    dhmf.agent_query("……", pretty=True)

内部：plan → query_skill × N → synthesize。
"""

from .prompts import Agent_PROMPT
from .runner import run_agent_query

__all__ = ['run_agent_query', 'Agent_PROMPT']
