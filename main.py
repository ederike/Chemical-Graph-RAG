from src.DHMF import DHMF
from src.utils.config import Config

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
# print(ChemicalGraph.query(
#     'Admex™ 6187的密度/比重参数是多少',
#     mode='dual_path',
#     pretty=True,
# ))

# --- 多跳 Agent（配置见 config.agent）---
print(ChemicalGraph.agent_query(
    'Advantex™ DM 与 Advantex™ 和 Advantex™ AR 的自燃温度有什么不同？？',
    pretty=True,
))

# print(ChemicalGraph.agent_query(
#     '比较 ABS 722 与 Admex™ 6187 的用途有什么不同',
#     pretty=True,
# ))
