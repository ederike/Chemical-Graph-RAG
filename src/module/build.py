from tqdm import tqdm
import json
import logging
from typing import Dict
from ..utils.database import BaseDB
from ..utils.config import Config

class BaseBuild:
    def __init__(self,db:Dict[str,BaseDB],logger:logging.Logger,config:Config):
        self.config = config
        self.logger = logger
        
        self.chunk_db = db['chunk']        
        self.hyperedge_db = db['hyperedge']
        self.node_db = db['node']
        self.edge_db = db['edge']

    def prepare_save(self):
        hyperedge_ids_map={}
        if 'hyperedge' in self.config.build.target:
            max_hyperedge_id=self.hyperedge_db.db.execute("SELECT MAX(id) FROM hyperedge")[0]['MAX(id)']
            if max_hyperedge_id is None:
                max_hyperedge_id=0
            for index in range(self.hyperedge_id_temp):
                hyperedge_ids_map[index] = max_hyperedge_id + 1 + index

        if 'node' in self.config.build.target:
            nodes=self.node_db.buffer
            for node in nodes:
                node['hyperedge_id'] = hyperedge_ids_map.get(node['hyperedge_id'],None)

        if 'edge' in self.config.build.target:
            edges=self.edge_db.buffer
            for edge in edges:
                edge['hyperedge_id'] = hyperedge_ids_map.get(edge['hyperedge_id'],None)


    def processing_single_task(self,task):
        extra=task['extra']
        
        extra=json.loads(extra)
        doc_id=task['doc_id']
        chunk_id=task['id']
        extracts=extra['extract']['extract']

        for extract in extracts:
            entities = extract.get('entities') or {}
            if 'hyperedge' in self.config.build.target:
                # 不再依赖 knowledge；内容优先用 chunk 原文
                hyperedge={
                    'doc_id':doc_id,
                    'chunk_id':chunk_id,
                    'content': task.get('content') or extract.get('knowledge') or '',
                    'extra':json.dumps({'entities': entities},ensure_ascii=False),
                }
                self.hyperedge_db.buffer.append(hyperedge)
            
            if 'node' in self.config.build.target:
                if isinstance(entities, dict):
                    node_iter = entities.items()
                    for name, content in node_iter:
                        node={
                            'doc_id':doc_id,
                            'chunk_id':chunk_id,
                            'hyperedge_id':self.hyperedge_id_temp,
                            'name':name,
                            'content': content or '',
                        }
                        self.node_db.buffer.append(node)
                else:
                    # 兼容旧 list 实体名
                    for name in entities:
                        node={
                            'doc_id':doc_id,
                            'chunk_id':chunk_id,
                            'hyperedge_id':self.hyperedge_id_temp,
                            'name':name,
                            'content':'',
                        }
                        self.node_db.buffer.append(node)
    
            if 'edge' in self.config.build.target:
                names = list(entities.keys()) if isinstance(entities, dict) else list(entities or [])
                for test in range(max(0, len(names)-1)):
                    edge={
                        'doc_id':doc_id,
                        'chunk_id':chunk_id,
                        'hyperedge_id':self.hyperedge_id_temp,
                        'name':"aaa",
                    }
                    self.edge_db.buffer.append(edge)

            update_chunk={
                'id': chunk_id,
                'status':'build',
            }
            self.chunk_db.buffer.append(update_chunk)
            self.hyperedge_id_temp+=1

    def prepare(self):
        if self.config.settings.debug:
            self.tasks=self.chunk_db.search_all()
        else:
            self.tasks=self.chunk_db.search('status',"extract")
        self.hyperedge_db.buffer_clear()
        self.node_db.buffer_clear()
        self.edge_db.buffer_clear()
        self.logger.debug(f"The number of chunks to be built :{len(self.tasks)}")
    
    def processing(self):
        self.hyperedge_id_temp = 0
        for task in tqdm(self.tasks):
            self.processing_single_task(task)

    def save(self):
        self.prepare_save()

        self.chunk_db.update(self.chunk_db.buffer)
        self.chunk_db.buffer_clear()

        self.hyperedge_db.add(self.hyperedge_db.buffer)
        self.hyperedge_db.buffer_clear()

        self.node_db.add(self.node_db.buffer)
        self.node_db.buffer_clear()

        self.edge_db.add(self.edge_db.buffer)
        self.edge_db.buffer_clear()

    def clear(self):
        self.chunk_db.update_key('status', 'extract')
        self.hyperedge_db.clear()
        self.node_db.clear()
        self.edge_db.clear()