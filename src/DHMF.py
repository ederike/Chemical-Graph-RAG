import logging
import time
from datetime import datetime
from pathlib import Path

from .module.doc import Doc
from .module.summary import Summary
from .module.chunk import Chunk
from .module.extract import Extract
from .module.build import Build
from .module.vectorization import Vectorization
from .module.retrieve import Retrieve
from .module.recommend import Recommend

from .utils.OpenAIAPI import LLM
from .utils.prompt import PROMPT

from .utils.storage import DocDB,ChunkDB,HyperedgeDB,NodeDB,EdgeDB,DocVDB,ChunkVDB,HyperedgeVDB,NodeVDB,EdgeVDB
from .utils.config import Config
from .utils.metrics import PipelineMetrics

class DHMF:
    def __init__(self,config:Config):
        self.config = config
        self.init_logger()

        db_path = Path(self.config.settings.working_path) / 'DB' / 'main.db'
        vdb_path = Path(self.config.settings.working_path) / 'DB' / 'vdb'

        self.db={
            'doc':DocDB(db_path),
            'chunk':ChunkDB(db_path),
            'hyperedge':HyperedgeDB(db_path),
            'node':NodeDB(db_path),
            'edge':EdgeDB(db_path),
        }

        self.vdb={
            'doc':DocVDB(vdb_path,self.config.vectorization.dim),
            'chunk':ChunkVDB(vdb_path,self.config.vectorization.dim),
            'hyperedge':HyperedgeVDB(vdb_path,self.config.vectorization.dim),
            'node':NodeVDB(vdb_path,self.config.vectorization.dim),
            'edge':EdgeVDB(vdb_path,self.config.vectorization.dim),
        }

        metrics_path = Path(self.config.settings.working_path) / 'DB' / 'build_metrics.json'
        self.metrics = PipelineMetrics(logger=self.logger, persist_path=metrics_path)
        self.metrics.load_lifetime()

        self.doc_module = Doc(db=self.db,logger=self.logger,config=self.config)
        self.summary_module = Summary(db=self.db,logger=self.logger,config=self.config)
        self.chunk_module = Chunk(db=self.db,logger=self.logger,config=self.config)
        self.extract_module = Extract(db=self.db,logger=self.logger,config=self.config)
        self.build_module = Build(db=self.db,logger=self.logger,config=self.config)
        self.vectorization_module = Vectorization(logger=self.logger,config=self.config)
        self.retrieve_module = Retrieve(db=self.db,vdb=self.vdb,logger=self.logger,config=self.config)
        self.recommend_module = Recommend(
            db=self.db, vdb=self.vdb, logger=self.logger, config=self.config
        )

        self.doc_module.metrics = self.metrics
        self.summary_module.metrics = self.metrics
        self.chunk_module.metrics = self.metrics
        self.extract_module.metrics = self.metrics
        self.build_module.metrics = self.metrics
        self.vectorization_module.metrics = self.metrics

    def init_logger(self):
        """
        Console-only by default. File log is created only when a build stage starts
        (see ensure_build_logger). Query never creates a new log file.
        """
        self.logger = logging.getLogger(f"DHMF.{id(self)}")
        self.logger.handlers.clear()
        self.logger.propagate = False

        self._log_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._log_formatter)
        self.logger.addHandler(console_handler)

        self._file_handler = None
        self._build_log_path = None
        self.logger.setLevel(
            logging.DEBUG if self.config.settings.debug else logging.INFO
        )

        from .utils.config import resolve_credentials
        api_key, base_url = resolve_credentials(self.config, self.config.retrieve)
        self.llmmodel = LLM(api_key, base_url)

    def ensure_build_logger(self):
        """
        Open a build-only log file once per DHMF instance.
        Safe to call multiple times; no-op if already open.
        """
        if self._file_handler is not None:
            return self._build_log_path

        log_dir = Path(self.config.settings.working_path) / 'log'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"build_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(self._log_formatter)
        self.logger.addHandler(file_handler)
        self._file_handler = file_handler
        self._build_log_path = log_path

        self.logger.info(f"[log] build log file: {log_path}")
        return log_path

    def begin_build(self, *, reset_session: bool = False):
        """
        Enter build mode: attach file logger + start pipeline metrics (if needed).

        reset_session=True: clear this-run counters and start a fresh session
        (still inherits lifetime totals from disk).
        """
        self.ensure_build_logger()
        if reset_session or self.metrics._wall_start is None:
            if reset_session:
                self.metrics.reset()
                self.metrics.load_lifetime()
            if self.metrics._wall_start is None:
                self.metrics.start_pipeline()

    def reset_metrics(self, start: bool = True):
        """Reset session cost counters; optionally mark pipeline wall-clock start."""
        self.ensure_build_logger()
        self.metrics.reset()
        self.metrics.load_lifetime()
        if start:
            self.metrics.start_pipeline()

    def log_metrics_summary(self):
        """Log this build + lifetime total; persist cumulative costs."""
        self.ensure_build_logger()
        self.metrics.end_pipeline()

    def download_from_oss(self, limit=None, file_type=None, skip_existing: bool = True):
        """
        Query spider_product (dm_data_mysql) and download OSS objects into working_path/doc.

        Does NOT recognize or insert — call insert_default() afterwards.

        Args:
            limit: max rows; 0 or None uses config.oss_download.limit
                   (0 in config = no limit). Function arg wins when not None.
            file_type: 1=TDS, 2=MSDS, 'all'=both; None → config.oss_download.file_type
            skip_existing: skip when local {product_name}_{id}.pdf already exists

        Returns:
            summary dict from download_rows_from_oss
        """
        from .utils.oss_download import fetch_spider_products, download_rows_from_oss

        oss_cfg = getattr(self.config, 'oss_download', None)
        mysql_cfg = getattr(self.config, 'dm_data_mysql', None)
        if mysql_cfg is None:
            raise RuntimeError("config.dm_data_mysql is required for download_from_oss")

        if limit is None:
            limit = int(getattr(oss_cfg, 'limit', 0) or 0) if oss_cfg else 0
        else:
            limit = int(limit)

        if file_type is None:
            file_type = getattr(oss_cfg, 'file_type', 'all') if oss_cfg else 'all'

        table = getattr(oss_cfg, 'table', 'spider_product') if oss_cfg else 'spider_product'
        download_subdir = (
            getattr(oss_cfg, 'download_dir', None)
            or getattr(self.config.doc, 'doc_dir', 'doc')
            or 'doc'
        )
        bucket_key = (
            getattr(oss_cfg, 'bucket_key', 'ky-products-files')
            if oss_cfg else 'ky-products-files'
        )
        download_dir = Path(self.config.settings.working_path) / download_subdir

        self.logger.info(
            f"Start download_from_oss: dir={download_dir}, "
            f"file_type={file_type}, limit={limit if limit > 0 else 'none'}"
        )

        rows = fetch_spider_products(
            mysql_cfg,
            table=table,
            file_type=file_type,
            limit=limit,
            logger=self.logger,
        )
        summary = download_rows_from_oss(
            rows,
            ali_oss=getattr(self.config, 'ali_oss', None) or {},
            bucket_key=bucket_key,
            download_dir=download_dir,
            skip_existing=skip_existing,
            logger=self.logger,
        )
        self.logger.info(f"Finish download_from_oss: {summary}")
        return summary

    def insert_default(self):
        """Insert docs from working_path/doc (pdf recognition or plain txt)."""
        self.begin_build()
        source_type = getattr(self.config.doc, 'source_type', 'pdf')
        if source_type == 'txt':
            doc_list = []
            folder_path = Path(self.config.settings.working_path) / getattr(
                self.config.doc, 'doc_dir', 'doc'
            )
            for file_path in folder_path.glob('*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_list.append({'name': file_path.name, 'content': f.read()})
            self.insert(doc_list)
        else:
            self.insert_pdfs()

    def insert_pdfs(self, pdf_paths=None, skip_existing: bool = True):
        """
        Pre-insert: PDF under working_path/doc → images → vision recognition → doc table.
        Each PDF becomes one independent document (texts are not mixed).

        按 doc.flush_every 分批：识别一批 → 入库一批，避免 2000 份全文同时驻留内存。

        skip_existing: skip files already present in doc table (default True),
                       so re-run will not re-recognize or re-insert them.
        """
        self.begin_build()
        self.logger.info("Start PDF recognition before insert (source: working_path/doc).")

        if pdf_paths is None:
            all_pdfs = self.doc_module.list_pdf_files()
        else:
            all_pdfs = [Path(p) for p in pdf_paths]

        if skip_existing:
            existing = self.doc_module.get_existing_doc_names()
            to_process = [p for p in all_pdfs if p.name not in existing]
            skipped = len(all_pdfs) - len(to_process)
            if skipped:
                self.logger.info(
                    f"Skip {skipped} PDF(s) already in doc table before batched recognize."
                )
        else:
            to_process = list(all_pdfs)

        if not to_process:
            self.logger.info("No new documents to insert (all skipped or empty).")
            return

        try:
            flush_every = int(getattr(self.config.doc, 'flush_every', 1000) or 1000)
        except (TypeError, ValueError):
            flush_every = 1000
        flush_every = max(1, flush_every)

        total_inserted = 0
        n_batches = (len(to_process) + flush_every - 1) // flush_every
        self.logger.info(
            f"PDF recognize+insert batched: files={len(to_process)} "
            f"flush_every={flush_every} batches={n_batches}"
        )
        for bi in range(0, len(to_process), flush_every):
            batch = to_process[bi: bi + flush_every]
            batch_i = bi // flush_every + 1
            self.logger.info(
                f"PDF batch {batch_i}/{n_batches}: recognize {len(batch)} file(s)"
            )
            doc_list = self.doc_module.prepare_from_pdfs(
                pdf_paths=batch,
                skip_existing=False,
                progress_total=len(to_process),
            )
            if not doc_list:
                self.logger.warning(
                    f"PDF batch {batch_i}/{n_batches}: no recognized docs"
                )
                continue
            self.insert(doc_list)
            total_inserted += len(doc_list)
        self.logger.info(
            f"Finish PDF recognize+insert: batches={n_batches} "
            f"inserted≈{total_inserted}"
        )

    def insert(self, doc_list):
        """
        Insert documents into doc table.
        Already-existing names/hashes are skipped inside Doc.prepare / processing.
        """
        self.begin_build()
        self.logger.info(f"Start inserting documents. Candidates: {len(doc_list)}")
        self.doc_module.prepare(doc_list)
        if not self.doc_module.tasks:
            self.logger.info("No new documents to insert after dedup filter.")
            return
        self.doc_module.processing()
        self.doc_module.save()
        self.logger.info(
            f"Finish inserting documents. Inserted: {len(self.doc_module.tasks)}"
        )

    def insert_clear(self):
        """清空 doc 表；同步清空 summary 写入的 hyperedge，避免悬空超边。"""
        self.doc_module.clear()
        try:
            self.summary_module.clear()
        except Exception as e:
            self.logger.warning(f"insert_clear: summary/hyperedge clear skipped: {e}")
        self.vectorization_clear('doc')

    def summary(self):
        """
        文档总结（可独立运行）：doc.content → LLM → hyperedge.content。
        须在 insert 之后、chunk 之前执行。
        """
        self.begin_build()
        self.logger.info("Start document summary → hyperedge.content.")
        self.summary_module.usage_prompt_tokens = 0
        self.summary_module.usage_completion_tokens = 0
        self.summary_module.prepare()
        if not self.summary_module.tasks:
            self.logger.info("No documents to summarize.")
            return
        self.summary_module.processing()
        self.summary_module.save()
        self.metrics.log_stage('summary')
        self.logger.info(
            f"Finish summary. "
            f"stage_tokens prompt={self.summary_module.usage_prompt_tokens} "
            f"completion={self.summary_module.usage_completion_tokens}"
        )

    def summary_clear(self):
        """清除超边总结；doc status summary→new。不影响 chunk/node（请按需 chunk_clear/build_clear）。"""
        self.begin_build()
        self.summary_module.clear()
        self.vectorization_clear('hyperedge')

    def delete(
        self,
        names,
        delete_edge: bool = True,
        delete_vectors: bool = True,
        clear_cache: bool = False,
    ):
        """
        Delete document(s) and all related table rows by file name(s).

        Removes from: doc, chunk, hyperedge, node (and edge by default).
        Optionally removes corresponding vectors from VDB.
        Optionally clears PDF recognition cache for the file(s).

        Args:
            names: str or list/tuple of file names, e.g. 'TDS_48400.pdf'
                   or ['TDS_48400.pdf', 'a.txt']. Match is exact on doc.name.
            delete_edge: also delete edge rows linked by doc_id (default True).
            delete_vectors: also remove embeddings from VDB (default True).
            clear_cache: if True, also delete PDF recognition cache entries
                         for this file name so next run will re-recognize (default False).

        Returns:
            dict summary: {name: {doc_ids, doc, chunk, hyperedge, node, edge, cache, ...}}
        """
        if isinstance(names, str):
            name_list = [names]
        else:
            name_list = list(names)

        summary = {}
        for name in name_list:
            docs = self.db['doc'].search('name', name)
            counts = {
                'found': bool(docs),
                'doc_ids': [d['id'] for d in docs] if docs else [],
                'doc': 0,
                'chunk': 0,
                'hyperedge': 0,
                'node': 0,
                'edge': 0,
                'cache': 0,
            }

            if not docs:
                self.logger.warning(f"delete: no document found with name={name!r}")
                if clear_cache:
                    counts['cache'] = self.doc_module.clear_recognition_cache(name)
                summary[name] = counts
                continue

            doc_ids = counts['doc_ids']

            chunk_ids, hyperedge_ids, node_ids, edge_ids = [], [], [], []
            for doc_id in doc_ids:
                chunk_ids.extend([r['id'] for r in self.db['chunk'].search('doc_id', doc_id)])
                hyperedge_ids.extend([r['id'] for r in self.db['hyperedge'].search('doc_id', doc_id)])
                node_ids.extend([r['id'] for r in self.db['node'].search('doc_id', doc_id)])
                if delete_edge:
                    edge_ids.extend([r['id'] for r in self.db['edge'].search('doc_id', doc_id)])

            for doc_id in doc_ids:
                counts['node'] += self.db['node'].delete('doc_id', doc_id)
                counts['hyperedge'] += self.db['hyperedge'].delete('doc_id', doc_id)
                counts['chunk'] += self.db['chunk'].delete('doc_id', doc_id)
                if delete_edge:
                    counts['edge'] += self.db['edge'].delete('doc_id', doc_id)
                counts['doc'] += self.db['doc'].delete('id', doc_id)

            if delete_vectors:
                try:
                    self.vdb['doc'].remove(doc_ids)
                    self.vdb['chunk'].remove(chunk_ids)
                    self.vdb['hyperedge'].remove(hyperedge_ids)
                    self.vdb['node'].remove(node_ids)
                    if delete_edge and edge_ids:
                        self.vdb['edge'].remove(edge_ids)
                except Exception as e:
                    self.logger.warning(f"delete: VDB cleanup partial failure for {name!r}: {e}")

            if clear_cache:
                counts['cache'] = self.doc_module.clear_recognition_cache(name)

            self.logger.info(
                f"delete: removed {name!r} "
                f"doc={counts['doc']} chunk={counts['chunk']} "
                f"hyperedge={counts['hyperedge']} node={counts['node']} "
                f"edge={counts['edge']} cache={counts['cache']}"
            )
            summary[name] = counts

        if hasattr(self.retrieve_module, '_precomputed'):
            self.retrieve_module._precomputed = False

        return summary

    def chunk(self):
        self.begin_build()
        self.logger.info(f"Start chunking.")
        self.chunk_module.prepare()
        self.chunk_module.processing()
        self.chunk_module.save()
        self.metrics.log_stage('chunk')
        self.logger.info(f"Finish chunking.")

    def chunk_clear(self):
        self.begin_build()
        self.chunk_module.clear()
        self.vectorization_clear('chunk')

    def extract(self):
        self.begin_build()
        self.logger.info(f"Start extracting.")
        self.extract_module.usage_prompt_tokens = 0
        self.extract_module.usage_completion_tokens = 0
        self.extract_module.prepare()
        self.extract_module.processing()
        self.extract_module.save()
        self.metrics.log_stage('extract')
        self.logger.info(
            f"Finish extracting. "
            f"stage_tokens prompt={self.extract_module.usage_prompt_tokens} "
            f"completion={self.extract_module.usage_completion_tokens}"
        )

    def extract_clear(self):
        self.begin_build()
        self.extract_module.clear()

    def build(self):
        self.begin_build()
        self.logger.info(f"Start building.")
        self.build_module.prepare()
        self.build_module.processing()
        self.build_module.save()
        self.metrics.log_stage('build')
        self.logger.info(f"Finish building.")
    
    def build_clear(self):
        """
        清除构图产物（node/edge）与相关向量；不删除 summary 写入的 hyperedge 正文。
        若需重做总结请调用 summary_clear()。
        """
        self.begin_build()
        self.build_module.clear()
        self.vectorization_clear('node')
        self.vectorization_clear('edge')

    def vectorization(self,db_name=None, finalize_metrics: bool = True):
        """
        Vectorize configured tables.
        finalize_metrics: after all targets done, log this-build + lifetime cost
                          summary and persist cumulative metrics.
        """
        self.begin_build()
        if db_name is None:
            db_name_list=self.config.vectorization.default_target
        else:
            db_name_list=[db_name]

        self.vectorization_module.usage_prompt_tokens = 0
        self.vectorization_module.usage_total_tokens = 0
            
        for db_name in db_name_list:
            self.logger.info(f"Start vectorization {db_name}.")
            self.vectorization_module.prepare(self.db[db_name],self.vdb[db_name])
            self.vectorization_module.processing()  # tqdm progress + stage metrics summary
            self.vectorization_module.save()
            self.logger.info(f"Finish vectorization {db_name}.")

        if finalize_metrics:
            self.log_metrics_summary()

    def vectorization_clear(self,db_name):
        self.begin_build()
        self.vectorization_module.clear(self.db[db_name],self.vdb[db_name])

    def recommend(self):
        """Offline similar-hyperedge recommendation (clears then recomputes)."""
        self.logger.info("Start recommend (similar hyperedges).")
        summary = self.recommend_module.run()
        if hasattr(self.retrieve_module, '_precomputed'):
            self.retrieve_module._precomputed = False
        self.logger.info(f"Finish recommend. summary={summary}")
        return summary

    @staticmethod
    def parse_query_answer(text: str) -> dict:
        """
        Split model answer into Thought / Answer sections when present.
        Supports labels: Thought:/思考: and Answer:/回答:/结论:
        """
        import re

        raw = (text or '').strip()
        if not raw:
            return {'thought': '', 'answer': '', 'raw': raw}

        ans_pat = re.compile(
            r'(?:^|\n)\s*(?:Answer|回答|结论)\s*[:：]\s*',
            re.IGNORECASE,
        )
        thought_pat = re.compile(
            r'^\s*(?:Thought|思考)\s*[:：]\s*',
            re.IGNORECASE,
        )

        ans_matches = list(ans_pat.finditer(raw))
        if ans_matches:
            m = ans_matches[-1]
            head = raw[: m.start()].strip()
            answer = raw[m.end():].strip()
            thought = thought_pat.sub('', head, count=1).strip() if head else ''
            return {'thought': thought, 'answer': answer, 'raw': raw}

        if thought_pat.match(raw):
            thought = thought_pat.sub('', raw, count=1).strip()
            return {'thought': thought, 'answer': '', 'raw': raw}
        return {'thought': '', 'answer': raw, 'raw': raw}

    @classmethod
    def format_query_response(cls, respond, *, query: str = None) -> str:
        """
        Pretty-print LLM query response for terminal readability.
        Separates Thought / Answer; falls back to full text if unparsed.
        """
        if not isinstance(respond, dict):
            return str(respond)

        status = respond.get('status')
        raw_answer = respond.get('answer')
        if raw_answer is None:
            raw_answer = str(respond)

        parsed = cls.parse_query_answer(str(raw_answer))
        lines = []
        lines.append('=' * 60)
        lines.append('Query Result')
        lines.append('=' * 60)
        if query:
            lines.append(f'question: {query}')
            lines.append('-' * 60)
        if status is not None:
            lines.append(f'status:   {"ok" if status == 1 else f"fail({status})"}')

        if parsed['thought']:
            lines.append('')
            lines.append('## Thought')
            lines.append('-' * 60)
            lines.append(parsed['thought'])

        if parsed['answer']:
            lines.append('')
            lines.append('## Answer')
            lines.append('-' * 60)
            lines.append(parsed['answer'])
        elif not parsed['thought']:
            lines.append('')
            lines.append('## Answer')
            lines.append('-' * 60)
            lines.append(parsed['raw'])

        pt = respond.get('usage_prompt_tokens')
        ct = respond.get('usage_completion_tokens')
        tt = respond.get('usage_total_tokens')
        if tt is None and pt is not None and ct is not None:
            try:
                tt = int(pt) + int(ct)
            except Exception:
                tt = None
        latency_s = respond.get('latency_s')
        retrieve_latency_s = respond.get('retrieve_latency_s')
        bits = []
        if any(v is not None for v in (pt, ct, tt)):
            bits.append(
                f'in={pt if pt is not None else 0} '
                f'out={ct if ct is not None else 0} '
                f'total={tt if tt is not None else 0}'
            )
        if retrieve_latency_s is not None:
            try:
                bits.append(f'retrieve={float(retrieve_latency_s):.3f}s')
            except Exception:
                bits.append(f'retrieve={retrieve_latency_s}')
        if latency_s is not None:
            try:
                bits.append(f'latency={float(latency_s):.3f}s')
            except Exception:
                bits.append(f'latency={latency_s}')
        if bits:
            lines.append('')
            lines.append('tokens  ' + ' | '.join(bits))
        lines.append('=' * 60)
        return '\n'.join(lines)

    def query(self, query, mode='dual_path', pretty=False):
        """Dual-path RAG query. pretty=True returns formatted Thought/Answer text."""
        if mode != 'dual_path':
            raise ValueError(
                f"Unknown query mode: {mode!r}. Supported: 'dual_path'. "
                f"For multi-hop agent use agent_query()."
            )

        t_all = time.perf_counter()
        t0 = time.perf_counter()
        retrieval_items = self.retrieve_module.retrieve_items(query)
        retrieval_result = self.retrieve_module._format_retrieved_chunks(retrieval_items)
        retrieve_latency_s = time.perf_counter() - t0
        system_prompt = PROMPT.get('query_answer_system', '')
        respond_prompt = (
            PROMPT.get('query_answer', '{retrieval_result}\n.Question: {query}')
            .replace('{retrieval_result}', str(retrieval_result or ''))
            .replace('{query}', str(query or ''))
        )
        respond = self.llmmodel.generate(
            prompt={'system': system_prompt, 'user': respond_prompt},
            model_args=self.config.retrieve.model_args,
            use_cache=getattr(self.config.retrieve, 'use_cache', True),
        )
        if isinstance(respond, dict):
            respond['retrieve_latency_s'] = retrieve_latency_s
            respond['latency_s'] = time.perf_counter() - t_all
            sources, doc_ids = [], []
            seen_src, seen_did = set(), set()
            for it in retrieval_items or []:
                src = it.get('source')
                if src and src not in seen_src:
                    seen_src.add(src)
                    sources.append(str(src))
                did = it.get('doc_id')
                if did is not None and did not in seen_did:
                    seen_did.add(did)
                    doc_ids.append(did)
            respond['retrieval_sources'] = sources
            respond['retrieval_doc_ids'] = doc_ids

        if pretty:
            return self.format_query_response(respond, query=query)
        return respond

    def agent_query(self, query, pretty=False):
        """Multi-hop agent query (plan → single-hop skills → synthesize)."""
        from .agent.runner import run_agent_query
        return run_agent_query(self, query, pretty=pretty)
