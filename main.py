from src.DHMF import DHMF
from src.utils.config import Config

config = Config.from_yaml('example/a/config_open.yaml')

ChemicalGraph = DHMF(config)

# --- build pipeline ---
# ChemicalGraph.insert_clear()
# ChemicalGraph.chunk_clear()
# ChemicalGraph.extract_clear()
# ChemicalGraph.build_clear()

ChemicalGraph.download_from_oss()
ChemicalGraph.insert_default()
ChemicalGraph.chunk()
ChemicalGraph.extract()
ChemicalGraph.build()
ChemicalGraph.vectorization()

# ChemicalGraph.recommend()


# --- dual-path ---

print(ChemicalGraph.query(
    '比较Eastman™ 1,4-CHDA-HP与Eastman™ CHDM-D的分子量，并说明两者在典型应用中对硬度和柔韧性的关键属性描述有何异同。',
    mode='dual_path',
    pretty=True,
))

# print(ChemicalGraph.query(
#     '列出所有明确提及“汽车”应用的产品，并综合比较这些产品在关键属性中关于耐腐蚀性或水解稳定性的表述。',
#     mode='dual_path',
#     pretty=True,
# ))
