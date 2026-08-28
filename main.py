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
    print(graph.agent_query('我们厂做保温装饰一体板，水性弹性丙烯酸涂料辊涂下线后覆 PE 保护膜，叠压码垛入库。天热堆放一两周后问题来了：揭膜时漆膜跟着膜被撕下来，板与板叠压面也互相粘连掉漆，废板率很高。乳液 Tg 偏低、漆膜偏软这个改不了（板子要弯折）。加蜡粉以前试过，表面会沾油污。不想上双组分。从配方上怎么解决？',
                             pretty=True))
    # graph.unpin_retrieve_indexes()

