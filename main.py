from src.DHMF import DHMF
from src.utils.config import Config

config = Config.from_yaml('example/a/config_open.yaml')

dhmf = DHMF(config)

# --- build pipeline ---
dhmf.insert_clear()
dhmf.chunk_clear()
dhmf.extract_clear()
dhmf.build_clear()

dhmf.download_from_oss()
dhmf.insert_default()
dhmf.chunk()
dhmf.extract()
dhmf.build()
dhmf.vectorization()

# dhmf.recommend()


# --- dual-path ---

# print(dhmf.query(
#     '比较Eastman™ 1,4-CHDA-HP与Eastman™ CHDM-D的分子量，并说明两者在典型应用中对硬度和柔韧性的关键属性描述有何异同。',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     '列出所有明确提及“汽车”应用的产品，并综合比较这些产品在关键属性中关于耐腐蚀性或水解稳定性的表述。',
#     mode='dual_path',
#     pretty=True,
# ))
