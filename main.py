"""Pipeline entry: build index then optionally run benchmark."""
from src.DHMF import DHMF
from src.utils.config import Config

if __name__ == '__main__':
    config = Config.from_yaml('example/a/config_open.yaml')
    graph = DHMF(config)

    # graph.download_from_oss()
    # graph.dedupe_downloaded_docs()

    # graph.insert_default()
    # graph.summary()
    # graph.chunk()
    # graph.extract()
    # graph.build()
    # graph.vectorization()

    graph.pin_retrieve_indexes()
    # print(graph.query('...', mode='dual_path', pretty=True))
    # print(graph.agent_query('',
    #                          pretty=True))
    print(graph.agentic_query(
        '我在开发一款微发泡塑木复合材料板材，配方中使用了N-β-氨乙基-γ-氨丙基甲基二甲氧基硅烷作为氨基硅烷偶联剂。现在我想在板材表面涂覆一层环保型防腐耐指纹金属表面漆（基于水性环氧树脂体系），请问我板材配方中的这种硅烷偶联剂，是否属于该表面漆专利推荐使用的硅烷偶联剂类型？如果不匹配，该表面漆推荐的具体硅烷偶联剂化学名称是什么？',
        pretty=True,
    ))
    graph.unpin_retrieve_indexes()

