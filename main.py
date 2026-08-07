from src.DHMF import DHMF
from src.utils.config import Config
import time 
import json
from benchmark.config import DEFAULT_CONFIG_PATH
from benchmark.workflow import TestQueryWorkflow
config = Config.from_yaml('example/a/config_open.yaml')

ChemicalGraph = DHMF(config)

# --- build pipeline ---
# ChemicalGraph.insert_clear()
# ChemicalGraph.summary_clear()
# ChemicalGraph.chunk_clear()
# ChemicalGraph.extract_clear()
# ChemicalGraph.build_clear()

ChemicalGraph.download_from_oss()
ChemicalGraph.insert_default()   # 逐页 VLM 识别 → doc 全文
ChemicalGraph.summary()          # 全文 LLM 总结 → hyperedge.content（可独立运行）
ChemicalGraph.chunk()            # head=超边总结 + body_n=识别正文分块
ChemicalGraph.extract()
ChemicalGraph.build()            # 绑定已有超边，建 node
ChemicalGraph.vectorization()

# ChemicalGraph.recommend()


# --- dual-path RAG ---
# print(ChemicalGraph.query(
#     'NFPA 评级为健康1,相对密度为1.10左右，自燃温度高于400°C以上的产品有什么',
#     mode='dual_path',
#     pretty=True,
# ))

# print(ChemicalGraph.query(
#     'CAS号为25120-20-1的东西是什么',
#     mode='dual_path',
#     pretty=True,
# ))


# --- 多跳 Agent（配置见 config.agent）---
# print(ChemicalGraph.agent_query(
#     'NFPA 评级为健康1,相对密度为1.10左右，自燃温度高于400°C以上的产品有什么？',
#     pretty=True,
# ))

# print(ChemicalGraph.agent_query(
#     '比较 ABS 722 与 Admex™ 6187 的用途有什么不同',
#     pretty=True,
# ))

wf = TestQueryWorkflow.from_config(DEFAULT_CONFIG_PATH)
wf.run()