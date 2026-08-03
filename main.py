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

print(dhmf.query(
    '合成高耐水解水性聚酯，同时使用 CHDA-HP 二酸与 CHDM-D 二醇，两者分别提供什么核心优势？',
    mode='dual_path',
    pretty=True,
))
