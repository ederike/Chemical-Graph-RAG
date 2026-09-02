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

    # graph.pin_retrieve_indexes()
    # print(graph.query('...', mode='dual_path', pretty=True))
    # print(graph.agent_query('',
    #                          pretty=True))
    # print(graph.agentic_query(
    #     '我是一家生产宠物食品包装模具的工厂工程师，正在评估使用陶氏的 SILASTIC™ RTV-3487 硅橡胶体系来复制复杂的宠物食品造型模具。考虑到宠物食品可能含有类似伊士曼 Eastman™ ProGIT SF3 饲料添加剂中的中链脂肪酸（MCFAs）等成分，且模具需要长期接触这些物质，请分析：1. 该硅橡胶体系在长期接触侵蚀性注料时的耐化学性能表现及潜在风险；2. 如果模具需要在 160°C 的高温环境下进行快速固化或储存，该材料是否适用？3. 基于其应用限制，该模具是否可以直接用于接触最终宠物食品？',
    #     pretty=True,
    # ))
    # graph.unpin_retrieve_indexes()

