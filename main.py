from src.DHMF import DHMF
from src.utils.config import Config
import time 
import json
config = Config.from_yaml('example/a/config_open.yaml')

ChemicalGraph = DHMF(config)

# --- build pipeline ---
# ChemicalGraph.insert_clear()
# ChemicalGraph.chunk_clear()
# ChemicalGraph.extract_clear()
# ChemicalGraph.build_clear()

# ChemicalGraph.download_from_oss()
# ChemicalGraph.insert_default()
# ChemicalGraph.chunk()
# ChemicalGraph.extract()
# ChemicalGraph.build()
# ChemicalGraph.vectorization()

# ChemicalGraph.recommend()


# --- dual-path RAG ---
print(ChemicalGraph.query(
    'NFPA 评级为健康1,相对密度为1.10左右，自燃温度高于400°C以上的产品有什么',
    mode='dual_path',
    pretty=True,
))

# print(ChemicalGraph.query(
#     'CAS号为25120-20-1的东西是什么',
#     mode='dual_path',
#     pretty=True,
# ))


# --- 多跳 Agent（配置见 config.agent）---
# print(ChemicalGraph.agent_query(
#     'NFPA 评级为健康1,相对密度为1.10左右，自燃温度高于400°C以上的产品有什么',
#     pretty=True,
# ))

# print(ChemicalGraph.agent_query(
#     '比较 ABS 722 与 Admex™ 6187 的用途有什么不同',
#     pretty=True,
# ))