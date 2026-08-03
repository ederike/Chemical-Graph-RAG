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

dhmf.recommend()


# --- dual-path ---

# print(dhmf.query(
#     'CAS号为7440-32-6的文件是什么',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     'EC.No为208-914-3的文件是什么',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     'EC.No为208-914开头的的文件是什么',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     'EC.No为208然后914最后不记得了的的文件是什么',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     'EC号开头为208的溶液文件有什么',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     '有没有胶粘剂，要环氧树脂的??',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     '给我多推荐几款汽车的玻璃水',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     '针对研磨抛光类产品的应用需求，您有什么推荐方案？',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     '对汽车抛光的产品有什么推荐方案？',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     '推荐几个锂相关产品?',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     '客户是做水烟里面的胶囊的，属于EVA材料，这款胶囊遇水之后很容易破掉，所以他咨询有没有抗氧剂加进去能增加韧性，就是抗老化，求推荐',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     '黑色抛光羊毛球中聚酰胺含量是多少',
#     mode='dual_path',
#     pretty=True,
# ))

# print(dhmf.query(
#     '环氧树脂260中云母成分是多少,容易被点燃吗?',
#     mode='dual_path',
#     pretty=True,
# ))

print(dhmf.query(
    'PH值是7.1的甲醛清除剂是什么',
    mode='dual_path',
    pretty=True,
))
