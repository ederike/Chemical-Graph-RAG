import tiktoken
from tqdm import tqdm
from typing import Dict
import logging
from ..utils.utils import hash_str
from ..utils.database import BaseDB
from ..utils.config import Config

class BaseDoc:
    def __init__(self,db:Dict[str,BaseDB],logger:logging.Logger,config:Config):
        self.config = config
        self.doc_db = db['doc']
        self.logger = logger

        if self.config.doc.count_token:
            self.tokener = tiktoken.get_encoding("cl100k_base")

    def processing_single_task(self,task):
        text = task['content']
        file_name = task['name']
        hash=hash_str(text)
        add_doc={
            'name':file_name,
            'content':text,
            'hash':hash,
        }
        if self.config.doc.count_token:
            add_doc['tokens']=len(self.tokener.encode(text))
        self.doc_db.buffer.append(add_doc)

    def prepare(self,doc_list):
        self.tasks = doc_list 
        self.doc_db.buffer_clear()
        self.logger.debug(f"Number of inserted documents :{len(self.tasks)}")

    def processing(self):
        for task in tqdm(self.tasks):
            self.processing_single_task(task)
    
    def save(self):
        self.doc_db.add(self.doc_db.buffer)
        self.doc_db.buffer_clear()

    def clear(self):
        self.doc_db.clear()        

