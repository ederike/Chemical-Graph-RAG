"""Pipeline entry: build index then optionally run benchmark."""
from src.DHMF import DHMF
from src.utils.config import Config

if __name__ == '__main__':
    config = Config.from_yaml('example/a/config_open.yaml')
    graph = DHMF(config)

    # graph.download_from_oss()
    # graph.dedupe_downloaded_docs()

    graph.insert_default()
    graph.summary()
    graph.chunk()
    graph.extract()
    graph.build()

    # graph.vectorization_clear()
    graph.vectorization()

    # 检索前按 yaml 的 chunk_max_vectors / node_max_vectors 常驻 FAISS 分片；
    # 不调用则仍按片 load/unload。须与 query 同一进程。
    # graph.pin_retrieve_indexes()
    # print(graph.query('...', mode='dual_path', pretty=True))
    # print(graph.agent_query('满足 NFPA 评级为健康1、相对密度约1.10、自燃温度高于400°C 的产品，其生产公司还有什么抗氧化剂产品吗？', pretty=True))
    # graph.unpin_retrieve_indexes()

