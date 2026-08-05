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

# --- vectorization ---
ChemicalGraph.vectorization(db_name='node')

# ChemicalGraph.recommend()


# --- dual-path RAG ---
# start = time.time()
print(ChemicalGraph.query(
    'Admex™ 6187的密度/比重参数是多少',
    mode='dual_path',
    pretty=True,
))
# end = time.time()
# with open('query_time_候选.jsonl', 'a') as f:
#     f.write(json.dumps({'query_time': end - start}) + '\n')


# --- 多跳 Agent（配置见 config.agent）---
# print(ChemicalGraph.agent_query(
#     'Advantex™ DM 与 Advantex™ 和 Advantex™ AR 的自燃温度有什么不同？？',
#     pretty=True,
# ))

# print(ChemicalGraph.agent_query(
#     '比较 ABS 722 与 Admex™ 6187 的用途有什么不同',
#     pretty=True,
# ))