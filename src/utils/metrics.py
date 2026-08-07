"""Pipeline cost metrics: real processing time + token usage (cache hits excluded from totals).

Session = current build run.
Lifetime = persisted across builds (working_path/DB/build_metrics.json).
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

def _safe_int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0

def _safe_float(v) -> float:
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0

@dataclass
class StageAgg:
    """Per-stage aggregation."""
    real_seconds: float = 0.0
    wall_seconds: float = 0.0  # includes cache-hit task durations for visibility
    real_calls: int = 0
    cache_hits: int = 0
    skipped: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class CostSnapshot:
    """Serializable cost totals (time + tokens)."""
    total_seconds: float = 0.0
    wall_seconds: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    real_task_count: int = 0
    cache_hit_count: int = 0
    skip_count: int = 0
    build_count: int = 0

    def add(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(
            total_seconds=self.total_seconds + other.total_seconds,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            total_prompt_tokens=self.total_prompt_tokens + other.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens + other.total_completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            real_task_count=self.real_task_count + other.real_task_count,
            cache_hit_count=self.cache_hit_count + other.cache_hit_count,
            skip_count=self.skip_count + other.skip_count,
            build_count=self.build_count + other.build_count,
        )

    def to_dict(self) -> dict:
        return {
            'total_seconds': self.total_seconds,
            'wall_seconds': self.wall_seconds,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens,
            'real_task_count': self.real_task_count,
            'cache_hit_count': self.cache_hit_count,
            'skip_count': self.skip_count,
            'build_count': self.build_count,
        }

    @classmethod
    def from_dict(cls, data: dict = None) -> "CostSnapshot":
        data = data or {}
        return cls(
            total_seconds=_safe_float(data.get('total_seconds')),
            wall_seconds=_safe_float(data.get('wall_seconds')),
            total_prompt_tokens=_safe_int(data.get('total_prompt_tokens')),
            total_completion_tokens=_safe_int(data.get('total_completion_tokens')),
            total_tokens=_safe_int(data.get('total_tokens')),
            real_task_count=_safe_int(data.get('real_task_count')),
            cache_hit_count=_safe_int(data.get('cache_hit_count')),
            skip_count=_safe_int(data.get('skip_count')),
            build_count=_safe_int(data.get('build_count')),
        )

class PipelineMetrics:
    """
    Track build cost from recognition → vectorization.

    - Session totals: only real API work this run (cache hit does NOT add)
    - Lifetime totals: persisted JSON, inherited across builds
    - Multi-thread stages use finalize_stage_wall_time for true elapsed time
    """

    def __init__(self, logger=None, persist_path: Optional[Path] = None):
        self.logger = logger
        self.persist_path: Optional[Path] = Path(persist_path) if persist_path else None
        self._lock = threading.Lock()
        # session (current build)
        self.total_seconds: float = 0.0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.real_task_count: int = 0
        self.cache_hit_count: int = 0
        self.skip_count: int = 0
        self.stages: Dict[str, StageAgg] = {}
        self._wall_start: Optional[float] = None
        self._wall_end: Optional[float] = None
        self._session_committed: bool = False
        # lifetime (before this session)
        self.lifetime_before = CostSnapshot()
        self._lifetime_loaded: bool = False

    def set_persist_path(self, path: Path):
        self.persist_path = Path(path)

    def load_lifetime(self, path: Optional[Path] = None) -> CostSnapshot:
        """Load cumulative build costs from disk (before this session)."""
        p = Path(path) if path else self.persist_path
        snap = CostSnapshot()
        if p is not None and p.exists():
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # support both flat and nested shapes
                if isinstance(data, dict) and 'lifetime' in data:
                    snap = CostSnapshot.from_dict(data.get('lifetime') or {})
                else:
                    snap = CostSnapshot.from_dict(data if isinstance(data, dict) else {})
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[metrics] load lifetime failed: {e}")
                snap = CostSnapshot()
        self.lifetime_before = snap
        self._lifetime_loaded = True
        return snap

    def save_lifetime(self, path: Optional[Path] = None, *, after_commit: bool = True):
        """
        Persist lifetime totals.
        after_commit=True: write lifetime_before + current session (if committed use lifetime_before only
        which already includes session after commit).
        """
        p = Path(path) if path else self.persist_path
        if p is None:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            lifetime = self.lifetime_total() if after_commit else self.lifetime_before
            payload = {
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'lifetime': lifetime.to_dict(),
                'last_session': self.session_snapshot().to_dict(),
            }
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[metrics] save lifetime failed: {e}")

    def session_snapshot(self) -> CostSnapshot:
        wall = self._session_wall()
        return CostSnapshot(
            total_seconds=self.total_seconds,
            wall_seconds=wall if wall is not None else 0.0,
            total_prompt_tokens=self.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens,
            total_tokens=self.total_tokens,
            real_task_count=self.real_task_count,
            cache_hit_count=self.cache_hit_count,
            skip_count=self.skip_count,
            build_count=1 if (self.real_task_count > 0 or self.total_seconds > 0 or self.total_tokens > 0) else 0,
        )

    def lifetime_total(self) -> CostSnapshot:
        """lifetime_before + current session (session not double-counted after commit)."""
        if self._session_committed:
            return self.lifetime_before
        return self.lifetime_before.add(self.session_snapshot())

    def _session_wall(self) -> Optional[float]:
        if self._wall_start is None:
            return None
        end = self._wall_end if self._wall_end is not None else time.perf_counter()
        return end - self._wall_start

    def commit_session_to_lifetime(self):
        """Fold this session into lifetime_before once, then persist."""
        with self._lock:
            if self._session_committed:
                return
            session = CostSnapshot(
                total_seconds=self.total_seconds,
                wall_seconds=self._session_wall() or 0.0,
                total_prompt_tokens=self.total_prompt_tokens,
                total_completion_tokens=self.total_completion_tokens,
                total_tokens=self.total_tokens,
                real_task_count=self.real_task_count,
                cache_hit_count=self.cache_hit_count,
                skip_count=self.skip_count,
                build_count=1,
            )
            # only count as a build if there was any work or wall time
            if session.total_tokens == 0 and session.total_seconds <= 0 and session.wall_seconds <= 0:
                session.build_count = 0
            self.lifetime_before = self.lifetime_before.add(session)
            self._session_committed = True
        self.save_lifetime(after_commit=True)

    def reset(self):
        with self._lock:
            self.total_seconds = 0.0
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_tokens = 0
            self.real_task_count = 0
            self.cache_hit_count = 0
            self.skip_count = 0
            self.stages = {}
            self._wall_start = None
            self._wall_end = None
            self._session_committed = False

    def start_pipeline(self):
        # load prior cumulative costs for inheritance display
        if not self._lifetime_loaded and self.persist_path is not None:
            self.load_lifetime()
        self._wall_start = time.perf_counter()
        self._wall_end = None
        self._session_committed = False
        if self.logger:
            self.logger.info("[metrics] pipeline start (recognition → vectorization)")
            prev = self.lifetime_before
            if prev.build_count or prev.total_tokens or prev.total_seconds:
                self.logger.info(
                    f"[metrics] inherited lifetime (before this build): "
                    f"builds={prev.build_count} "
                    f"time={prev.total_seconds:.3f}s wall={prev.wall_seconds:.3f}s "
                    f"tokens prompt={prev.total_prompt_tokens} "
                    f"completion={prev.total_completion_tokens} "
                    f"total={prev.total_tokens}"
                )
            else:
                self.logger.info("[metrics] inherited lifetime: empty (first build)")

    def end_pipeline(self):
        self._wall_end = time.perf_counter()
        self.log_summary()
        self.commit_session_to_lifetime()
        self.log_lifetime_summary()

    def _stage(self, name: str) -> StageAgg:
        if name not in self.stages:
            self.stages[name] = StageAgg()
        return self.stages[name]

    def record(
        self,
        stage: str,
        duration: float,
        *,
        cache_hit: bool = False,
        skipped: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = None,
        name: str = None,
        extra: str = None,
        log: bool = False,
        accumulate_time: bool = True,
    ):
        """
        Record one task under a stage.

        cache_hit / skipped: do NOT add duration or tokens to pipeline totals.
        Real work: add tokens (and duration if accumulate_time) to totals.

        accumulate_time=False: still count calls/tokens, but leave time for a later
        wall-clock finalize (used by multi-threaded PDF recognition so log shows
        real elapsed time, not sum of concurrent task times).
        """
        prompt_tokens = _safe_int(prompt_tokens)
        completion_tokens = _safe_int(completion_tokens)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        else:
            total_tokens = _safe_int(total_tokens)

        duration = float(duration or 0.0)
        _ = (name, extra, log)

        with self._lock:
            st = self._stage(stage)
            if accumulate_time:
                st.wall_seconds += duration

            if skipped:
                st.skipped += 1
                self.skip_count += 1
            elif cache_hit:
                st.cache_hits += 1
                self.cache_hit_count += 1
            else:
                st.real_calls += 1
                st.prompt_tokens += prompt_tokens
                st.completion_tokens += completion_tokens
                st.total_tokens += total_tokens
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                self.real_task_count += 1
                if accumulate_time:
                    st.real_seconds += duration
                    self.total_seconds += duration

    def finalize_stage_wall_time(self, stage: str, wall_seconds: float):
        """
        Set a stage's real processing time to measured wall-clock elapsed.

        For multi-threaded stages the sum of per-task durations exceeds real
        elapsed time; call this after the batch finishes so logs report only the
        true total processing time.
        """
        wall_seconds = float(wall_seconds or 0.0)
        with self._lock:
            st = self._stage(stage)
            old = st.real_seconds
            st.wall_seconds = wall_seconds
            if st.real_calls <= 0:
                self.total_seconds = max(0.0, self.total_seconds - old)
                st.real_seconds = 0.0
            else:
                self.total_seconds = max(0.0, self.total_seconds - old + wall_seconds)
                st.real_seconds = wall_seconds

    def stage_snapshot(self, stage: str) -> dict:
        """Thread-safe snapshot of one stage (for progress bar postfix)."""
        with self._lock:
            st = self.stages.get(stage)
            if st is None:
                return {
                    'real': 0, 'cache': 0, 'skip': 0,
                    'real_s': 0.0, 'tokens': 0,
                    'total_s': self.total_seconds, 'total_tok': self.total_tokens,
                }
            return {
                'real': st.real_calls,
                'cache': st.cache_hits,
                'skip': st.skipped,
                'real_s': st.real_seconds,
                'tokens': st.total_tokens,
                'total_s': self.total_seconds,
                'total_tok': self.total_tokens,
            }

    def log_stage(self, stage: str):
        """INFO one-line summary for a finished stage (no per-item cache-hit noise)."""
        if not self.logger:
            return
        snap = self.stage_snapshot(stage)
        if stage.startswith('vectorization:') or stage in ('chunk', 'build', 'extract'):
            self.logger.info(
                f"[metrics] stage={stage} done "
                f"real={snap['real']} "
                f"real_time={snap['real_s']:.3f}s stage_tokens={snap['tokens']} "
                f"| session_time={snap['total_s']:.3f}s "
                f"session_tokens={snap['total_tok']}"
            )
        else:
            self.logger.info(
                f"[metrics] stage={stage} done "
                f"real={snap['real']} cache_hit={snap['cache']} skip={snap['skip']} "
                f"real_time={snap['real_s']:.3f}s stage_tokens={snap['tokens']} "
                f"| session_time={snap['total_s']:.3f}s "
                f"session_tokens={snap['total_tok']}"
            )

    def record_from_response(
        self,
        stage: str,
        duration: float,
        response: dict = None,
        *,
        name: str = None,
        force_cache_hit: bool = None,
        extra: str = None,
        log: bool = True,
    ):
        """Helper: pull tokens + _cache_hit from an OpenAI-style response dict."""
        response = response or {}
        if force_cache_hit is not None:
            cache_hit = force_cache_hit
        else:
            cache_hit = bool(response.get('_cache_hit'))
        self.record(
            stage,
            duration,
            cache_hit=cache_hit,
            prompt_tokens=response.get('usage_prompt_tokens') or 0,
            completion_tokens=response.get('usage_completion_tokens') or 0,
            total_tokens=response.get('usage_total_tokens'),
            name=name,
            extra=extra,
            log=log,
        )

    @contextmanager
    def measure(self, stage: str, *, name: str = None, log: bool = True):
        """
        Context manager: yield a dict to fill with cache_hit / tokens, then record.
        """
        info = {
            'cache_hit': False,
            'skipped': False,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': None,
            'extra': None,
        }
        t0 = time.perf_counter()
        try:
            yield info
        finally:
            dt = time.perf_counter() - t0
            self.record(
                stage,
                dt,
                cache_hit=bool(info.get('cache_hit')),
                skipped=bool(info.get('skipped')),
                prompt_tokens=info.get('prompt_tokens') or 0,
                completion_tokens=info.get('completion_tokens') or 0,
                total_tokens=info.get('total_tokens'),
                name=name,
                extra=info.get('extra'),
                log=log,
            )

    def log_summary(self):
        """Log this build session cost (time + tokens)."""
        if not self.logger:
            return
        wall = self._session_wall()
        lines = [
            "[metrics] ========== 本次构建消耗 (this build) ==========",
            f"[metrics] session stage_time_sum={self.total_seconds:.3f}s "
            f"(real API stages; multi-thread=batch wall-clock; cache_hit/skip 不计)",
            f"[metrics] session wall_clock_time={wall:.3f}s" if wall is not None else
            "[metrics] session wall_clock_time=n/a",
            f"[metrics] session tokens prompt={self.total_prompt_tokens} "
            f"completion={self.total_completion_tokens} total={self.total_tokens}",
            f"[metrics] session real_tasks={self.real_task_count} "
            f"cache_hits={self.cache_hit_count} skips={self.skip_count}",
        ]
        if wall is not None and self.total_seconds > wall * 1.5 + 1.0:
            lines.append(
                f"[metrics] WARN stage_time_sum ({self.total_seconds:.1f}s) >> "
                f"wall_clock ({wall:.1f}s): some multi-thread stage may not have "
                f"called finalize_stage_wall_time()"
            )
        for stage, st in self.stages.items():
            if stage.startswith('vectorization:') or stage in ('chunk', 'build', 'extract'):
                lines.append(
                    f"[metrics]   stage={stage}: wall={st.real_seconds:.3f}s "
                    f"(ref_wall={st.wall_seconds:.3f}s) "
                    f"real_calls={st.real_calls} "
                    f"tokens={st.prompt_tokens}/{st.completion_tokens}/{st.total_tokens}"
                )
            else:
                lines.append(
                    f"[metrics]   stage={stage}: wall={st.real_seconds:.3f}s "
                    f"(ref_wall={st.wall_seconds:.3f}s) "
                    f"real_calls={st.real_calls} cache_hits={st.cache_hits} "
                    f"skipped={st.skipped} "
                    f"tokens={st.prompt_tokens}/{st.completion_tokens}/{st.total_tokens}"
                )
        lines.append("[metrics] ==================================================")
        for line in lines:
            self.logger.info(line)

    def log_lifetime_summary(self):
        """Log inherited + this session = grand total."""
        if not self.logger:
            return
        prev = self.lifetime_before
        session = self.session_snapshot()
        # after commit, lifetime_before already includes session
        if self._session_committed:
            total = prev
            # reconstruct previous without session for display
            prev_display = CostSnapshot(
                total_seconds=max(0.0, prev.total_seconds - session.total_seconds),
                wall_seconds=max(0.0, prev.wall_seconds - session.wall_seconds),
                total_prompt_tokens=max(0, prev.total_prompt_tokens - session.total_prompt_tokens),
                total_completion_tokens=max(0, prev.total_completion_tokens - session.total_completion_tokens),
                total_tokens=max(0, prev.total_tokens - session.total_tokens),
                real_task_count=max(0, prev.real_task_count - session.real_task_count),
                cache_hit_count=max(0, prev.cache_hit_count - session.cache_hit_count),
                skip_count=max(0, prev.skip_count - session.skip_count),
                build_count=max(0, prev.build_count - session.build_count),
            )
            grand = prev
        else:
            prev_display = prev
            grand = prev.add(session)

        lines = [
            "[metrics] ========== 累计构建消耗 (lifetime total) ==========",
            f"[metrics] previous: builds={prev_display.build_count} "
            f"time={prev_display.total_seconds:.3f}s wall={prev_display.wall_seconds:.3f}s "
            f"tokens={prev_display.total_prompt_tokens}/"
            f"{prev_display.total_completion_tokens}/{prev_display.total_tokens}",
            f"[metrics] this build: time={session.total_seconds:.3f}s "
            f"wall={session.wall_seconds:.3f}s "
            f"tokens={session.total_prompt_tokens}/"
            f"{session.total_completion_tokens}/{session.total_tokens}",
            f"[metrics] TOTAL: builds={grand.build_count} "
            f"time={grand.total_seconds:.3f}s wall={grand.wall_seconds:.3f}s "
            f"tokens prompt={grand.total_prompt_tokens} "
            f"completion={grand.total_completion_tokens} "
            f"total={grand.total_tokens}",
            "[metrics] ====================================================",
        ]
        for line in lines:
            self.logger.info(line)

    def as_dict(self) -> dict:
        wall = self._session_wall()
        return {
            'total_seconds': self.total_seconds,
            'wall_seconds': wall,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens,
            'real_task_count': self.real_task_count,
            'cache_hit_count': self.cache_hit_count,
            'skip_count': self.skip_count,
            'session_committed': self._session_committed,
            'lifetime_before': self.lifetime_before.to_dict(),
            'lifetime_total': self.lifetime_total().to_dict(),
            'stages': {
                k: {
                    'real_seconds': v.real_seconds,
                    'wall_seconds': v.wall_seconds,
                    'real_calls': v.real_calls,
                    'cache_hits': v.cache_hits,
                    'skipped': v.skipped,
                    'prompt_tokens': v.prompt_tokens,
                    'completion_tokens': v.completion_tokens,
                    'total_tokens': v.total_tokens,
                }
                for k, v in self.stages.items()
            },
        }
