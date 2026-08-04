from tqdm import tqdm
import json
import logging
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils.OpenAIAPI import Embedding
from ..utils.database import BaseDB,BaseVDB
from ..utils.utils import Retry
from ..utils.config import Config, resolve_credentials

class BaseVectorization:
    def __init__(self,logger:logging.Logger,config:Config):
        self.config = config
        self.logger = logger
        api_key, base_url = resolve_credentials(config, config.vectorization)
        emb_timeout, emb_retries, emb_wait = 120.0, 3, 0.5
        try:
            v_retry = getattr(config.vectorization, 'retry', None)
            if v_retry is not None:
                emb_timeout = float(getattr(v_retry, 'timeout', emb_timeout) or emb_timeout)
                emb_retries = int(getattr(v_retry, 'max_attempt', emb_retries) or emb_retries)
                emb_wait = float(getattr(v_retry, 'wait', emb_wait) or emb_wait)
        except Exception:
            pass
        self.embedding = Embedding(
            api_key=api_key,
            base_url=base_url,
            timeout=emb_timeout,
            max_retries=emb_retries,
            retry_wait=max(0.5, emb_wait),
        )

    @Retry(max_attempt=5, wait=0.1, timeout=600, config_attr='vectorization.retry')
    def processing_single_task(self,task,**kwargs):
        emb_content=task['embedding_content']
        if emb_content is None:
            emb_content=task['content']
        if emb_content is None:
            return
        
        model_args=self.config.vectorization.model_args

        response = self.embedding.generate(emb_content,model_args=model_args,use_cache=self.config.vectorization.use_cache)
        if response['status']!=1:
            raise Exception(f"Embedding failed, status: {response['status']}")
        
        emb = response['answer']
        update_task={
            'id':task['id'],
            'embedding_status':'done',
        }
        add_task_vdb={
            'id':task['id'],
            'embedding':json.dumps(emb,ensure_ascii=False),
        }
        self.task_db.buffer.append(update_task)
        self.task_vdb.buffer.append(add_task_vdb)

    def prepare(self,db: BaseDB, vdb: BaseVDB):
        self.task_db=db
        self.task_vdb=vdb
        if self.config.settings.debug:
            self.tasks=self.task_db.search_all()
        else:
            self.tasks=self.task_db.search('embedding_status','undone')
        self.task_db.buffer_clear()
        self.logger.debug(f"The number of tasks to be vectorized :{len(self.tasks)}")

    def processing(self):
        if self.config.vectorization.num_thread<=1:
            for task in tqdm(self.tasks):
                self.processing_single_task(task)
        else:
            with ThreadPoolExecutor(max_workers=self.config.vectorization.num_thread) as executor:
                futures = [executor.submit(self.processing_single_task, task) for task in self.tasks]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()

    def save(self):
        self.task_vdb.add(self.task_vdb.buffer)
        self.task_vdb.buffer_clear()
        self.task_vdb.save()

        self.task_db.update(self.task_db.buffer)
        self.task_db.buffer_clear()
        
    def clear(self,db: BaseDB,vdb: BaseVDB):
        db.update_key('embedding_status',None)
        db.clear()

        vdb.clear()
