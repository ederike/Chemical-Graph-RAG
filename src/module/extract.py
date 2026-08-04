from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from typing import Dict
import copy
from ..utils.database import BaseDB
from ..utils.config import Config

from ..utils.OpenAIAPI import LLM
from  ..utils.prompt import PROMPT
from ..utils.utils import Retry
from ..utils.config import resolve_credentials

class BaseExtract:
    def __init__(self,db:Dict[str,BaseDB],logger:logging.Logger,config:Config):
        self.config = config
        self.chunk_db = db['chunk']
        self.logger = logger

        api_key, base_url = resolve_credentials(config, config.extract)
        self.llmmodel = LLM(api_key, base_url)

    def check_extract(self,text):
        """只要求 entities；统一为 [{"entities": {...}}, ...]。支持 JSON Mode 纯 JSON 与代码块包裹。"""
        try:
            res = json.loads(text)
        except Exception:
            try:
                cleaned = (text or '').strip()
                if cleaned.startswith('```'):
                    lines = cleaned.splitlines()
                    if lines and lines[0].startswith('```'):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == '```':
                        lines = lines[:-1]
                    cleaned = '\n'.join(lines)
                res = json.loads(cleaned)
            except Exception:
                return None

        def _norm_entities(ent):
            if not isinstance(ent, dict):
                return None
            if ent and not all(isinstance(k, str) and isinstance(v, str) for k, v in ent.items()):
                return None
            return ent

        items = []
        if isinstance(res, dict):
            if 'entities' in res:
                ent = _norm_entities(res.get('entities'))
                if ent is None:
                    return None
                items = [{'entities': ent}]
            elif res and all(isinstance(v, str) for v in res.values()):
                items = [{'entities': res}]
            else:
                return None
        elif isinstance(res, list):
            if not res:
                return None
            for item in res:
                if not isinstance(item, dict):
                    return None
                if 'entities' not in item:
                    return None
                ent = _norm_entities(item.get('entities'))
                if ent is None:
                    return None
                items.append({'entities': ent})
        else:
            return None
        return items

    @Retry(max_attempt=5, wait=0.1, timeout=600, config_attr='extract.retry')
    def processing_single_task(self,chunk,**kwargs):
        attempt=kwargs.get('attempt',1)
        content=chunk['content']

        if attempt > 1:
            self.logger.debug(f"Chunk {chunk['id']} began its {attempt} extraction attempt.")

        NPROMPT= PROMPT[self.config.extract.extract_prompt].format(content=content)

        model_args = copy.deepcopy(self.config.extract.model_args or {})
        model_args.setdefault('enable_thinking', False)
        model_args.setdefault('response_format', {'type': 'json_object'})
        if attempt > 1:
            model_args['temperature'] = 1.0

        response = self.llmmodel.generate(prompt={'system':"",'user':NPROMPT},model_args=model_args,attempt=attempt,use_cache=self.config.extract.use_cache)
        extract_res = self.check_extract(response['answer'])

        if extract_res is None:
            raise ValueError(f"Extraction failed after {attempt} attempts.")
        
        extra={
            'extract':{
                'attempt':attempt,
                'extract':extract_res,
                'cost':{
                    'usage_prompt_tokens':response.get('usage_prompt_tokens',None),
                    'usage_completion_tokens':response.get('usage_completion_tokens',None),
                    'usage_total_tokens':response.get('usage_total_tokens',None),
                    'usage_cached_tokens':response.get('usage_cached_tokens',None),
                }
            }
        }
        
        updata_chunk={
            'id': chunk['id'],
            'status':'extract',
            'extra':json.dumps(extra,ensure_ascii=False),
        }
        self.chunk_db.buffer.append(updata_chunk)


    def prepare(self):
        if self.config.settings.debug:
            self.tasks=self.chunk_db.search_all()
        else:
            self.tasks=self.chunk_db.search('status',"new")
        self.chunk_db.buffer_clear()
        self.logger.debug(f"The number of chunks to be extracted :{len(self.tasks)}")
        

    def processing(self):
        if self.config.extract.num_thread<=1:
            for task in tqdm(self.tasks):
                self.processing_single_task(task)
        else:
            with ThreadPoolExecutor(max_workers=self.config.extract.num_thread) as executor:
                futures = [executor.submit(self.processing_single_task, task) for task in self.tasks]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()

    def save(self):
        self.chunk_db.update(self.chunk_db.buffer)
        self.chunk_db.buffer_clear()

    def clear(self):
        self.chunk_db.update_key('status','new')
        self.chunk_db.update_key('extra',None)
        