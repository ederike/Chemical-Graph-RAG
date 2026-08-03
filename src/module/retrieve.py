from typing import Dict
import logging
from ..utils.database import BaseDB, BaseVDB
from ..utils.config import Config


class BaseRetrieve:
    def __init__(self, db: Dict[str, BaseDB], vdb: Dict[str, BaseVDB], logger: logging.Logger, config: Config):
        self.config = config
        self.logger = logger

        self.db = db
        self.vdb = vdb

    def vector_match(self, db_name, vector, topk=10):
        vdb = self.vdb[db_name]
        db = self.db[db_name]
        vdb_res = vdb.search(vector, topk)
        res = [
            {'distance': item['distance'], 'result': db.search('id', item['id'])[0]}
            for item in vdb_res
        ]
        return res
