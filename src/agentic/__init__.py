"""
Tool-calling 检索问答。

对外入口请走 DHMF 薄封装::

    dhmf.agentic_query("……", pretty=True)

内部：同一条 messages 上循环 think → search/read_doc/graph_neighbors → 作答。
配置只读 config.agentic。
"""

from .runner import run_agentic_query

__all__ = ["run_agentic_query"]
