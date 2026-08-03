from ..module.vectorization import BaseVectorization
from ..utils.utils import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import time


class Vectorization(BaseVectorization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = None  # set by DHMF
        self.usage_prompt_tokens = 0
        self.usage_total_tokens = 0

    @Retry(max_attempt=5, wait=0.1, timeout=600)
    def processing_single_task(self, task, **kwargs):
        emb_content = task['embedding_content']
        if emb_content is None:
            emb_content = task['content']
        if emb_content is None:
            return

        t0 = time.perf_counter()
        model_args = self.config.vectorization.model_args
        table = getattr(self.task_db, 'table', 'unknown')

        response = self.embedding.generate(
            emb_content,
            model_args=model_args,
            use_cache=self.config.vectorization.use_cache,
        )
        if response['status'] != 1:
            raise Exception(f"Embedding failed, status: {response['status']}")

        cache_hit = bool(response.get('_cache_hit'))
        prompt_tok = response.get('usage_prompt_tokens') or 0
        completion_tok = response.get('usage_completion_tokens') or 0
        total_tok = response.get('usage_total_tokens')
        if total_tok is None:
            total_tok = (prompt_tok or 0) + (completion_tok or 0)

        if not cache_hit:
            self.usage_prompt_tokens += prompt_tok or 0
            self.usage_total_tokens += total_tok or 0

        dt = time.perf_counter() - t0
        if self.metrics is not None:
            # silent per-task; wall-clock finalized once in processing()
            self.metrics.record(
                f'vectorization:{table}',
                dt,
                cache_hit=cache_hit,
                prompt_tokens=prompt_tok if not cache_hit else 0,
                completion_tokens=completion_tok if not cache_hit else 0,
                total_tokens=total_tok if not cache_hit else 0,
                name=f"{table}_{task.get('id')}",
                log=False,
                accumulate_time=False,
            )

        emb = response['answer']
        update_task = {
            'id': task['id'],
            'embedding_status': 'done',
        }
        add_task_vdb = {
            'id': task['id'],
            'embedding': json.dumps(emb, ensure_ascii=False),
        }
        self.task_db.buffer.append(update_task)
        self.task_vdb.buffer.append(add_task_vdb)

    def processing(self):
        """Run with a single progress bar; metrics use batch wall-clock."""
        table = getattr(self.task_db, 'table', 'unknown')
        stage = f'vectorization:{table}'
        n = len(self.tasks)
        if n == 0:
            return

        t_wall = time.perf_counter()

        def _postfix():
            if self.metrics is None:
                return {}
            s = self.metrics.stage_snapshot(stage)
            # do not show cache hits (too noisy for node/hyperedge/chunk)
            return {
                'real': s['real'],
                'tok': s['tokens'],
                'tot_s': f"{s['total_s']:.1f}s",
            }

        if self.config.vectorization.num_thread <= 1:
            bar = tqdm(self.tasks, desc=f'vectorize:{table}', unit='item')
            for task in bar:
                self.processing_single_task(task)
                bar.set_postfix(**_postfix())
        else:
            with ThreadPoolExecutor(max_workers=self.config.vectorization.num_thread) as executor:
                futures = [executor.submit(self.processing_single_task, task) for task in self.tasks]
                bar = tqdm(as_completed(futures), total=n, desc=f'vectorize:{table}', unit='item')
                for future in bar:
                    future.result()
                    bar.set_postfix(**_postfix())

        if self.metrics is not None:
            # Multi-thread: report batch wall-clock, not sum of concurrent tasks
            self.metrics.finalize_stage_wall_time(
                stage, time.perf_counter() - t_wall
            )
            self.metrics.log_stage(stage)
