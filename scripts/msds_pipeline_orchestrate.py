#!/usr/bin/env python3
"""
Unattended pipeline (may run for days):

  wait for the other-terminal benchmark_agentic_5 process
  → python main.py (resume node embeddings)
  → benchmark_agentic_1 all-mode on MSDS, OOM-safe
  → sanity-check results; retry on crash / empty / all-error

Does not edit src/. Status: ~/.grok/long-running-background-tasks/msds_pipeline/status.json
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/Chemical-Graph-RAG")
PY = Path("/root/miniconda3/envs/test/bin/python")
DIR = Path.home() / ".grok" / "long-running-background-tasks" / "msds_pipeline"
STATUS = DIR / "status.json"
ORCH_LOG = DIR / "orchestrate.log"
PIDFILE = DIR / "orchestrate.pid"
DB = ROOT / "example" / "a" / "DB" / "main.db"
EVALS = ROOT / "results" / "benchmark_agentic_1_仅msds" / "evals.json"
QUESTIONS = ROOT / "results" / "benchmark_agentic_1_仅msds" / "questions.json"
REPORT = ROOT / "results" / "benchmark_agentic_1_仅msds" / "report.json"
REPORT_XLSX = ROOT / "results" / "benchmark_agentic_1_仅msds" / "report.xlsx"
BENCH5_EVALS = ROOT / "results" / "benchmark_agentic_5_与weknora对比" / "evals.json"

VEC_RESTARTS = 20
EVAL_RESTARTS = 12
STALL_VEC_S = 90 * 60
STALL_EVAL_S = 8 * 3600
# Other-terminal eval may be cancelled and restarted; wait this long
# after the process disappears before treating it as finished.
BENCH5_RESTART_GRACE_S = 30 * 60


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    DIR.mkdir(parents=True, exist_ok=True)
    with ORCH_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(**kw) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if STATUS.is_file():
        try:
            data = json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.update(kw)
    data["updated_at"] = now()
    data["pid"] = os.getpid()
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATUS)


def read_status() -> dict:
    if not STATUS.is_file():
        return {}
    try:
        return json.loads(STATUS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pids_matching(needle: str) -> list[int]:
    out = []
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,cmd"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return out
    for line in proc.stdout.splitlines()[1:]:
        line = line.strip()
        if needle not in line:
            continue
        if "msds_pipeline_orchestrate" in line:
            continue
        if "health_snapshot" in line or "GROK_AGENT" in line:
            continue
        cmd = line.split(None, 1)[-1] if line.split(None, 1) else ""
        if "python" not in cmd:
            continue
        try:
            out.append(int(line.split(None, 1)[0]))
        except (TypeError, ValueError):
            continue
    return out


def _bench5_meta() -> tuple:
    """Return (done_flag, n_done, n_total)."""
    if not BENCH5_EVALS.is_file():
        return False, None, None
    try:
        meta = json.loads(BENCH5_EVALS.read_text(encoding="utf-8")).get("meta") or {}
    except Exception:
        return False, None, None
    prog = meta.get("progress") or {}
    n_done = prog.get("done")
    n_total = prog.get("total")
    if n_done is None:
        n_done = meta.get("n_questions")
    if n_total is None:
        n_total = meta.get("n_total_planned")
    done_flag = bool(meta.get("done") is True)
    try:
        if n_done is not None and n_total is not None and int(n_total) > 0:
            if int(n_done) >= int(n_total):
                done_flag = True
    except (TypeError, ValueError):
        pass
    return done_flag, n_done, n_total


def wait_bench5() -> None:
    """Wait until the other-terminal benchmark_agentic_5 run is finished.

    Must not start vectorization just because the process vanished: the user
    may cancel a stuck question and restart the same module. Only proceed when
    evals.json reports completion, or the process has been gone longer than
    BENCH5_RESTART_GRACE_S *and* evals already look complete.
    Incomplete + process gone: keep waiting (user required the eval to finish).
    """
    write_status(phase="wait_bench5", detail="waiting for benchmark_agentic_5")
    last_n = -1
    idle = 0
    gone_since = None
    while True:
        pids = pids_matching("benchmark.benchmark_agentic_5")
        done_flag, n_done, n_total = _bench5_meta()
        if pids:
            gone_since = None
            if n_done != last_n:
                log(f"bench5 still running pids={pids} progress={n_done}/{n_total}")
                last_n = n_done
                idle = 0
            else:
                idle += 1
                if idle % 20 == 0:
                    log(
                        f"bench5 still running pids={pids} "
                        f"progress={n_done}/{n_total} (no change)"
                    )
            write_status(
                phase="wait_bench5",
                detail=f"pids={pids} progress={n_done}/{n_total}",
                bench5_pids=pids,
                bench5_progress=n_done,
                bench5_total=n_total,
            )
            time.sleep(30)
            continue

        if done_flag:
            log(f"bench5 finished evals done=true progress={n_done}/{n_total}")
            return

        now_ts = time.time()
        if gone_since is None:
            gone_since = now_ts
            log(
                f"bench5 process gone; evals incomplete "
                f"progress={n_done}/{n_total}; waiting up to "
                f"{int(BENCH5_RESTART_GRACE_S)}s for a restart"
            )
        gone_for = now_ts - gone_since
        write_status(
            phase="wait_bench5",
            detail=(
                f"pids=[] progress={n_done}/{n_total} "
                f"gone_s={int(gone_for)} waiting_restart"
            ),
            bench5_pids=[],
            bench5_progress=n_done,
            bench5_total=n_total,
            bench5_gone_s=int(gone_for),
        )
        if gone_for < BENCH5_RESTART_GRACE_S:
            time.sleep(30)
            continue
        # Still incomplete after grace: keep waiting. User required the
        # other-terminal eval to finish; they may restart hours later.
        if int(gone_for) % 600 < 40:
            log(
                f"bench5 still absent after {int(gone_for)}s; "
                f"evals incomplete {n_done}/{n_total}; keep waiting"
            )
        time.sleep(30)


def sqlite_connect(timeout: float = 600.0):
    con = sqlite3.connect(str(DB), timeout=timeout)
    con.execute("PRAGMA busy_timeout=600000")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=FILE")
    return con


def ensure_indexes() -> None:
    write_status(phase="index", detail="CREATE INDEX IF NOT EXISTS")
    stmts = [
        "CREATE INDEX IF NOT EXISTS idx_node_emb_status_id ON node(embedding_status, id)",
        "CREATE INDEX IF NOT EXISTS idx_chunk_emb_status_id ON chunk(embedding_status, id)",
        "CREATE INDEX IF NOT EXISTS idx_chunk_doc_id ON chunk(doc_id, id)",
        "CREATE INDEX IF NOT EXISTS idx_hyperedge_doc_id ON hyperedge(doc_id)",
        "CREATE INDEX IF NOT EXISTS idx_hyperedge_chunk_id ON hyperedge(chunk_id)",
    ]
    con = sqlite_connect()
    try:
        for sql in stmts:
            log(f"index start: {sql}")
            t0 = time.time()
            con.execute(sql)
            con.commit()
            log(f"index done in {time.time()-t0:.1f}s: {sql}")
            write_status(phase="index", detail=f"done {sql}")
    finally:
        con.close()


def embedding_undone(table: str) -> int:
    con = sqlite_connect()
    try:
        row = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE embedding_status = 'undone'"
        ).fetchone()
        return int(row[0] if row else 0)
    finally:
        con.close()


def vectorization_complete() -> bool:
    try:
        n_chunk = embedding_undone("chunk")
        n_node = embedding_undone("node")
    except Exception as e:
        log(f"undone count failed: {e}")
        return False
    log(f"undone chunk={n_chunk} node={n_node}")
    write_status(phase="vectorize_check", undone_chunk=n_chunk, undone_node=n_node)
    return n_chunk == 0 and n_node == 0


def spawn(cmd: list[str], logfile: Path, extra_env: dict | None = None) -> subprocess.Popen:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    if extra_env:
        env.update(extra_env)
    fp = logfile.open("a", encoding="utf-8")
    fp.write(f"\n\n===== spawn {now()} cmd={' '.join(cmd)} =====\n")
    fp.flush()
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=fp,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    ), fp  # type: ignore[return-value]


def wait_child(
    proc: subprocess.Popen,
    fp,
    *,
    phase: str,
    logpath: Path,
    stall_s: float,
) -> int:
    last_size = -1
    last_change = time.time()
    while True:
        rc = proc.poll()
        try:
            sz = logpath.stat().st_size
        except OSError:
            sz = 0
        if sz != last_size:
            last_size = sz
            last_change = time.time()
            tail = _tail(logpath, 8)
            write_status(
                phase=phase,
                child_pid=proc.pid,
                log=str(logpath),
                log_bytes=sz,
                tail=tail,
            )
        idle = time.time() - last_change
        if idle > stall_s and rc is None:
            log(f"STALL {phase} pid={proc.pid} no log growth for {idle:.0f}s")
            write_status(phase=phase, stalled=True, stall_s=idle, child_pid=proc.pid)
            # do not kill: embedding/search can pause on a slow API; just record
            last_change = time.time()
        if rc is not None:
            try:
                fp.close()
            except Exception:
                pass
            if rc in (137, 9, 247, -9):
                log(f"{phase} pid={proc.pid} killed rc={rc} (likely OOM/SIGKILL)")
            elif rc in (139, -11):
                log(f"{phase} pid={proc.pid} segfault rc={rc}")
            else:
                log(f"{phase} pid={proc.pid} exited rc={rc}")
            tail = _tail(logpath, 30)
            if any(
                s in tail.lower()
                for s in ("killed", "memoryerror", "out of memory", "cannot allocate")
            ):
                log(f"{phase} log looks like OOM/kill; will retry")
            return int(rc)
        time.sleep(20)


def _tail(path: Path, n: int = 12) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = [ln for ln in data.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def _adopt_existing(needle: str, logfile: Path, phase: str, stall_s: float) -> bool:
    """If a matching child is already running, wait for it instead of spawning."""
    pids = [p for p in pids_matching(needle) if p != os.getpid()]
    if not pids:
        return False
    pid = pids[0]
    log(f"adopt existing {phase} pid={pid} cmd~{needle}")
    write_status(phase=phase, child_pid=pid, adopted=True)

    class _Proc:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None if alive(self.pid) else 0

    class _Fp:
        def close(self):
            return None

    wait_child(_Proc(), _Fp(), phase=phase, logpath=logfile, stall_s=stall_s)
    return True


def run_vectorization() -> None:
    if vectorization_complete():
        log("vectorization already complete")
        return
    vec_log = DIR / "vectorize.log"
    for attempt in range(1, VEC_RESTARTS + 1):
        if vectorization_complete():
            return
        log(f"vectorize attempt {attempt}/{VEC_RESTARTS}")
        write_status(phase="vectorize", attempt=attempt)
        if _adopt_existing(str(ROOT / "main.py"), vec_log, "vectorize", STALL_VEC_S):
            if vectorization_complete():
                return
            continue
        proc, fp = spawn([str(PY), str(ROOT / "main.py")], vec_log)
        rc = wait_child(
            proc, fp, phase="vectorize", logpath=vec_log, stall_s=STALL_VEC_S
        )
        if rc == 0 and vectorization_complete():
            log("vectorization finished")
            return
        if rc == 0:
            log("main.py rc=0 but undone rows remain; retry")
        else:
            log(f"main.py crashed rc={rc}; retry in 30s")
        time.sleep(30)
    raise RuntimeError(f"vectorization failed after {VEC_RESTARTS} attempts")


def questions_looks_sane() -> tuple[bool, str]:
    if not QUESTIONS.is_file():
        return False, "missing questions.json"
    try:
        data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"questions.json parse error: {e}"
    qs = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(qs, list) or not qs:
        return False, "no questions"
    ok = 0
    empty = 0
    err = 0
    for q in qs:
        if not isinstance(q, dict):
            empty += 1
            continue
        if q.get("gen_status") == 1 and str(q.get("question") or "").strip():
            ok += 1
        else:
            if q.get("gen_error"):
                err += 1
            else:
                empty += 1
    meta = data.get("meta") or {}
    planned = int(meta.get("total") or 200)
    msg = f"questions ok={ok}/{planned} empty={empty} gen_error={err}"
    if ok == 0:
        return False, "all questions empty: " + msg
    if ok < max(10, planned * 0.5):
        return False, "too few questions: " + msg
    return True, msg


def eval_looks_sane(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing evals.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"evals.json parse error: {e}"
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        return False, "no results"
    meta = data.get("meta") or {}
    n = len(results)
    empty = 0
    err = 0
    for r in results:
        ag = (r or {}).get("agentic") or (r or {}).get("系统A") or {}
        if not isinstance(ag, dict):
            ag = {}
        ans = str(ag.get("answer") or "").strip()
        qerr = ag.get("query_error")
        if qerr:
            err += 1
        if not ans:
            empty += 1
    planned = int(meta.get("n_total_planned") or meta.get("n_questions") or 0) or n
    done_flag = bool(meta.get("done"))
    msg = (
        f"n={n}/{planned} done={done_flag} empty_answer={empty} "
        f"query_error={err}"
    )
    if not done_flag and n < planned:
        return False, "incomplete: " + msg
    if empty == n:
        return False, "all answers empty: " + msg
    if err == n:
        return False, "all query_error: " + msg
    if n >= 5 and empty / n > 0.6:
        return False, "too many empty answers: " + msg
    if n >= 5 and err / n > 0.6:
        return False, "too many query_error: " + msg
    if not REPORT.is_file() and not REPORT_XLSX.is_file():
        # report may still be written after evaluate
        if done_flag:
            return True, "evals ok, report pending: " + msg
    return True, msg


def run_eval() -> None:
    eval_log = DIR / "eval.log"
    for attempt in range(1, EVAL_RESTARTS + 1):
        ok, msg = eval_looks_sane(EVALS)
        if ok and REPORT_XLSX.is_file():
            log(f"eval already sane: {msg}")
            return
        log(f"eval attempt {attempt}/{EVAL_RESTARTS} ({msg if EVALS.is_file() else 'no evals yet'})")
        write_status(phase="eval", attempt=attempt, detail=msg if EVALS.is_file() else "")
        if _adopt_existing(
            "run_benchmark_agentic_1_msds_safe.py", eval_log, "eval", STALL_EVAL_S
        ):
            ok, msg = eval_looks_sane(EVALS)
            if ok:
                return
            continue
        proc, fp = spawn(
            [str(PY), str(ROOT / "scripts" / "run_benchmark_agentic_1_msds_safe.py")],
            eval_log,
        )
        rc = wait_child(
            proc, fp, phase="eval", logpath=eval_log, stall_s=STALL_EVAL_S
        )
        ok, msg = eval_looks_sane(EVALS)
        log(f"eval rc={rc} sanity={ok} {msg}")
        if rc == 0 and ok:
            if not REPORT_XLSX.is_file() and EVALS.is_file():
                log("evals ok but xlsx missing; report step should have run")
            if ok:
                return
        qok, qmsg = questions_looks_sane()
        log(f"questions sanity={qok} {qmsg}")
        if not qok:
            log("question generation looks bad; retrying full all-mode")
        time.sleep(20)
    ok, msg = eval_looks_sane(EVALS)
    if ok:
        return
    raise RuntimeError(f"eval failed after {EVAL_RESTARTS} attempts: {msg}")


def meminfo() -> str:
    try:
        txt = Path("/proc/meminfo").read_text(encoding="utf-8")
    except Exception:
        return ""
    keys = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    got = {}
    for line in txt.splitlines():
        k = line.split(":", 1)[0]
        if k in keys:
            got[k] = line.split(":", 1)[1].strip()
    return " ".join(f"{k}={got[k]}" for k in keys if k in got)


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    log(f"orchestrator start pid={os.getpid()} cwd={ROOT} mem={meminfo()}")
    write_status(phase="start", mem=meminfo())
    os.chdir(ROOT)

    try:
        # Strictly serial. The other-terminal eval holds GPU/RAM; do nothing
        # (including SQLite indexes on example/a) until it has finished.
        wait_bench5()
        try:
            ensure_indexes()
        except Exception as e:
            log(f"index non-fatal (vectorization can still resume): {e}")
        run_vectorization()
        run_eval()
        ok, msg = eval_looks_sane(EVALS)
        write_status(
            phase="done" if ok else "failed",
            detail=msg,
            evals=str(EVALS),
            report=str(REPORT_XLSX),
            mem=meminfo(),
        )
        log(f"pipeline {'DONE' if ok else 'FAILED'}: {msg}")
        return 0 if ok else 1
    except Exception as e:
        log(f"pipeline FAILED: {e}")
        write_status(phase="failed", detail=str(e), mem=meminfo())
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        write_status(phase="failed", detail="KeyboardInterrupt")
        raise
