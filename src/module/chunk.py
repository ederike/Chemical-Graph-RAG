import tiktoken
from tqdm import tqdm
import re
import logging
from typing import Dict
from ..utils.database import BaseDB
from ..utils.config import Config

class BaseChunk:
    def __init__(self,db:Dict[str,BaseDB],logger:logging.Logger,config:Config):
        self.config = config
        self.doc_db = db['doc']
        self.chunk_db = db['chunk']
        self.logger = logger

        self.tokener = tiktoken.get_encoding("cl100k_base")

    def dynamic_division(self,text,tokener,min_tokens = 100,max_tokens = 400):

        lines = text.splitlines(True)
        chunks=[]
        chunk=""
        chunk_tokens=0
        for line in lines:

            line_tokens = len(tokener.encode(line))
            if chunk_tokens + line_tokens < min_tokens:
                chunk += line
                chunk_tokens += line_tokens
                continue

            if chunk_tokens + line_tokens <= max_tokens:
                chunk += line
                chunk_tokens += line_tokens
                chunks.append({'content':chunk.strip(),'tokens':chunk_tokens})
                chunk=""
                chunk_tokens=0
                continue
            
            punctuation = '。！？；：.?!:;'
            pattern = rf'([{re.escape(punctuation)}])'
            parts = re.split(pattern, line)
            sentences = ["".join(parts[i:i+2]) for i in range(0, len(parts)-1, 2)]
            if len(parts) % 2 == 1:
                sentences.append(parts[-1])

            for sentence in sentences:

                
                sentence_tokens_ = tokener.encode(sentence)
                sentence_tokens = len(sentence_tokens_)

                if chunk_tokens + sentence_tokens < min_tokens:
                    chunk += sentence
                    chunk_tokens += sentence_tokens
                    continue

                if chunk_tokens + sentence_tokens <= max_tokens:
                    chunk += sentence
                    chunk_tokens += sentence_tokens
                    chunks.append({'content':chunk.strip(),'tokens':chunk_tokens})
                    chunk=""
                    chunk_tokens=0
                    continue
                
                temp_token=[]
                for token in sentence_tokens_:
                    if chunk_tokens + 1 < min_tokens:
                        temp_token += [token]
                        chunk_tokens += 1
                        continue
                    
                    if chunk_tokens + 1 <= max_tokens:
                        temp_token += [token]
                        chunk_tokens += 1
                        chunk += tokener.decode(temp_token)
                        chunks.append({'content':chunk.strip(),'tokens':chunk_tokens})
                        temp_token=[]
                        chunk=""
                        chunk_tokens=0
                        continue
                
                if len(temp_token) > 0:
                    chunk += tokener.decode(temp_token)

        if chunk_tokens > 0:
            chunks.append({'content':chunk.strip(),'tokens':chunk_tokens})
        
        return chunks

    def processing_single_task(self,task):
        text=task['content']
        doc_id=task['id']

        chunks = self.dynamic_division(
            text=text,
            tokener=self.tokener,
            min_tokens=self.config.chunk.chunk_size_min,
            max_tokens=self.config.chunk.chunk_size_max,
        )
        for chunk in chunks:
            add_chunk={
                "doc_id": doc_id,
                "content": chunk['content'],
            }
            if self.config.chunk.count_token:
                add_chunk['tokens']=len(self.tokener.encode(chunk['content']))
            self.chunk_db.buffer.append(add_chunk)

        update_doc={
            'id': doc_id,
            "status": "chunk",
        }
        self.doc_db.buffer.append(update_doc)

    def prepare(self):
        if self.config.settings.debug:
            self.tasks=self.doc_db.search_all()
        else:
            self.tasks=self.doc_db.search('status',"new")
        self.doc_db.buffer_clear()
        self.chunk_db.buffer_clear()
        self.logger.debug(f"The number of documents to be chunk :{len(self.tasks)}")
        
    def processing(self):
        for task in tqdm(self.tasks):
            self.processing_single_task(task)

    def save(self):
        self.doc_db.update(self.doc_db.buffer)
        self.doc_db.buffer_clear()
        
        self.chunk_db.add(self.chunk_db.buffer)
        self.chunk_db.buffer_clear()

    def clear(self):
        self.chunk_db.clear()
        self.doc_db.update_key('status','new')
