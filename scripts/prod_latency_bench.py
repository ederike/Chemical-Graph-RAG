"""生产环境速度与延迟抽样。

从三个既有评测问题集各随机抽取固定题数，经公网站点的 Agentic
流式接口逐题作答，记录客户端端到端耗时、首个过程事件延迟、服务端
处理耗时与检索耗时，并导出 Excel。

账号口令只从环境变量读取，不写入结果文件：

    export CGR_BENCH_USER=...
    export CGR_BENCH_PASSWORD=...
    python scripts/prod_latency_bench.py

中断后用同一 --out 目录再次执行即可续跑（抽样清单会固定下来）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent

# 三个评测基准各对应一份问题集，合计默认 90 题。
QUESTION_SETS = (
    {
        "set_id": "agentic_1",
        "set_name": "库内自生成试题",
        "path": "results/benchmark_agentic_1_含专利/questions.json",
    },
    {
        "set_id": "coatings_set1",
        "set_name": "建筑涂料测试问题集1",
        "path": "results/benchmark_agentic_2_questions1_agentic_new/set1_dataset.json",
    },
    {
        "set_id": "patent",
        "set_name": "专利配方试题",
        "path": "results/benchmark_agentic_4_专利测试集/questions.json",
    },
)

TIMING_KEYS = (
    "precompute_s",
    "rewrite_s",
    "embed_s",
    "chunk_s",
    "node_s",
    "keyword_s",
    "expand_s",
    "rerank_s",
    "total_s",
)


def _load_questions(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise SystemExit(f"无法从 {path} 读出 questions 列表")
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "gen_status" in row and int(row.get("gen_status") or 0) != 1:
            continue
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        out.append(
            {
                "question_id": str(row.get("id") or ""),
                "question": question,
                "hop": row.get("hop"),
                "question_type": row.get("question_type") or "",
            }
        )
    return out


def sample_questions(n_per_set: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    picked: List[dict] = []
    for spec in QUESTION_SETS:
        path = ROOT / spec["path"]
        pool = _load_questions(path)
        if len(pool) < n_per_set:
            raise SystemExit(
                f"{spec['set_name']} 可用题 {len(pool)}，少于抽取数 {n_per_set}"
            )
        chosen = rng.sample(pool, n_per_set)
        for item in chosen:
            picked.append(
                {
                    "set_id": spec["set_id"],
                    "set_name": spec["set_name"],
                    **item,
                }
            )
    return picked


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _round(value: Optional[float], ndigits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ndigits)


class Client:
    def __init__(self, base_url: str, username: str, password: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        self._token = ""
        self._lock = threading.Lock()

    def _request(self, path: str, payload: dict, *, token: str = "") -> Any:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["Accept"] = "text/event-stream"
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:400]}") from exc
        return resp

    def login(self) -> str:
        resp = self._request(
            "/api/auth/login",
            {"username": self.username, "password": self.password},
        )
        with resp:
            data = json.loads(resp.read().decode("utf-8"))
        token = str(data.get("token") or "")
        if not token:
            raise RuntimeError("登录成功但未返回 token")
        with self._lock:
            self._token = token
        return token

    def token(self) -> str:
        with self._lock:
            if self._token:
                return self._token
        return self.login()

    def ask(self, question: str) -> dict:
        """走与网页相同的 /api/stream，mode=agentic。"""
        token = self.token()
        t0 = time.perf_counter()
        resp = self._request(
            "/api/stream",
            {"query": question, "mode": "agentic"},
            token=token,
        )
        first_event_s = None
        n_steps = 0
        done = None
        error = ""
        buf = b""
        try:
            with resp:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        event = json.loads(line[5:].decode("utf-8", errors="replace"))
                        now = time.perf_counter() - t0
                        if first_event_s is None:
                            first_event_s = now
                        etype = event.get("type")
                        if etype == "step":
                            n_steps += 1
                        elif etype == "error":
                            error = str(event.get("message") or event)
                            done = event
                            break
                        elif etype == "done":
                            done = event
                            break
                    if done is not None:
                        break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        client_total_s = time.perf_counter() - t0
        data = (done or {}).get("data") if isinstance(done, dict) else None
        if not isinstance(data, dict):
            data = {}
        timing = data.get("retrieve_timing") or {}
        if not isinstance(timing, dict):
            timing = {}
        answer = str(data.get("answer") or "")
        status = data.get("status")
        ok = (not error) and int(status or 0) == 1 and bool(answer.strip())
        if not done and not error:
            error = "流结束但未收到 done"
        row = {
            "ok": ok,
            "http_error": error,
            "client_total_s": client_total_s,
            "first_event_s": first_event_s,
            "server_latency_s": data.get("latency_s"),
            "retrieve_latency_s": data.get("retrieve_latency_s"),
            "n_turns": len(data.get("turns") or []),
            "n_sources": len(data.get("retrieval_sources") or []),
            "n_steps": n_steps,
            "answer_chars": len(answer),
            "answer_preview": answer[:400],
            "usage_prompt_tokens": data.get("usage_prompt_tokens"),
            "usage_completion_tokens": data.get("usage_completion_tokens"),
            "status": status,
        }
        for key in TIMING_KEYS:
            row[key] = timing.get(key)
        return row


def _append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def _load_done(path: Path) -> Dict[str, dict]:
    done: Dict[str, dict] = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        key = f"{row.get('set_id')}::{row.get('question_id')}"
        done[key] = row
    return done


def _stat_block(rows: List[dict]) -> dict:
    def nums(field: str) -> List[float]:
        out = []
        for row in rows:
            value = row.get(field)
            if value is None or value == "":
                continue
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                continue
        return out

    def pack(field: str) -> dict:
        values = nums(field)
        if not values:
            return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
        return {
            "n": len(values),
            "mean": _round(statistics.fmean(values)),
            "median": _round(statistics.median(values)),
            "p90": _round(_percentile(values, 0.90)),
            "max": _round(max(values)),
        }

    n = len(rows)
    n_ok = sum(1 for row in rows if row.get("ok"))
    return {
        "n": n,
        "n_ok": n_ok,
        "n_fail": n - n_ok,
        "success_rate": _round(n_ok / n, 4) if n else None,
        "client_total_s": pack("client_total_s"),
        "first_event_s": pack("first_event_s"),
        "server_latency_s": pack("server_latency_s"),
        "retrieve_latency_s": pack("retrieve_latency_s"),
        "n_turns_mean": _round(statistics.fmean(nums("n_turns"))) if nums("n_turns") else None,
        "n_sources_mean": _round(statistics.fmean(nums("n_sources"))) if nums("n_sources") else None,
        "answer_chars_mean": _round(statistics.fmean(nums("answer_chars")), 0) if nums("answer_chars") else None,
    }


def summarize(rows: List[dict], *, wall_s: float, meta: dict) -> dict:
    groups = {"全部": rows}
    for spec in QUESTION_SETS:
        groups[spec["set_name"]] = [
            row for row in rows if row.get("set_id") == spec["set_id"]
        ]
    return {
        "meta": meta,
        "wall_s": _round(wall_s),
        "groups": {name: _stat_block(items) for name, items in groups.items()},
    }


def write_excel(path: Path, rows: List[dict], summary: dict) -> None:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="1F4E79")
    fail_fill = PatternFill("solid", fgColor="F4CCCC")
    thin = Border(
        left=Side(style="thin", color="D0D7E2"),
        right=Side(style="thin", color="D0D7E2"),
        top=Side(style="thin", color="D0D7E2"),
        bottom=Side(style="thin", color="D0D7E2"),
    )
    wrap = Alignment(wrap_text=True, vertical="center")

    def style_header(ws, ncol: int) -> None:
        for col in range(1, ncol + 1):
            cell = ws.cell(1, col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes = "A2"
        ws.row_dimensions[1].height = 22

    ws = wb.active
    ws.title = "汇总"
    meta = summary.get("meta") or {}
    ws["A1"] = "生产端速度与延迟抽样"
    ws["A1"].font = Font(bold=True, size=14)
    info = [
        ("测试时间", meta.get("started_at", "")),
        ("入口", meta.get("base_url", "")),
        ("接口", "POST /api/stream  mode=agentic"),
        ("抽样", f"每个问题集 {meta.get('n_per_set')} 题，种子 {meta.get('seed')}"),
        ("并发", meta.get("concurrency")),
        ("整批墙钟（秒）", summary.get("wall_s")),
    ]
    for i, (k, v) in enumerate(info, start=3):
        ws.cell(i, 1, k).font = Font(bold=True)
        ws.cell(i, 2, v)
    ws.cell(9, 1, "说明").font = Font(bold=True)
    ws.merge_cells(start_row=9, start_column=2, end_row=10, end_column=8)
    note = ws.cell(
        9, 2,
        "客户端端到端耗时含网络与排队；服务端处理时延为接口返回的 latency_s，"
        "从工作线程开始计算。首事件延迟为收到第一条过程事件的时间。"
        "检索耗时为各次工具检索之和。",
    )
    note.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[9].height = 20
    ws.row_dimensions[10].height = 20

    headers = [
        "问题集",
        "题数",
        "成功",
        "失败",
        "成功率",
        "端到端均值(秒)",
        "端到端中位数(秒)",
        "端到端P90(秒)",
        "端到端最大(秒)",
        "服务端均值(秒)",
        "服务端中位数(秒)",
        "服务端P90(秒)",
        "服务端最大(秒)",
        "检索均值(秒)",
        "检索中位数(秒)",
        "首事件均值(秒)",
        "首事件中位数(秒)",
        "平均轮次",
        "平均来源数",
    ]
    start = 12
    for col, name in enumerate(headers, start=1):
        cell = ws.cell(start, col, name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    order = ["全部"] + [spec["set_name"] for spec in QUESTION_SETS]
    for r, name in enumerate(order, start=start + 1):
        block = (summary.get("groups") or {}).get(name) or {}
        client = block.get("client_total_s") or {}
        server = block.get("server_latency_s") or {}
        retrieve = block.get("retrieve_latency_s") or {}
        first = block.get("first_event_s") or {}
        rate = block.get("success_rate")
        values = [
            name,
            block.get("n"),
            block.get("n_ok"),
            block.get("n_fail"),
            rate,
            client.get("mean"),
            client.get("median"),
            client.get("p90"),
            client.get("max"),
            server.get("mean"),
            server.get("median"),
            server.get("p90"),
            server.get("max"),
            retrieve.get("mean"),
            retrieve.get("median"),
            first.get("mean"),
            first.get("median"),
            block.get("n_turns_mean"),
            block.get("n_sources_mean"),
        ]
        for col, value in enumerate(values, start=1):
            cell = ws.cell(r, col, value)
            cell.border = thin
            cell.alignment = Alignment(vertical="center")
            if col == 5 and isinstance(value, float):
                cell.number_format = "0.00%"
        if block.get("n_fail"):
            ws.cell(r, 4).fill = fail_fill
    ws.row_dimensions[start].height = 30
    ws.freeze_panes = "A13"
    widths = [22, 8, 8, 8, 10, 16, 18, 16, 16, 16, 18, 16, 16, 14, 16, 16, 18, 12, 12]
    for i, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width

    detail = wb.create_sheet("逐题结果")
    detail_headers = [
        "序号",
        "问题集",
        "题号",
        "问题",
        "是否成功",
        "端到端耗时(秒)",
        "首事件延迟(秒)",
        "服务端处理(秒)",
        "检索耗时(秒)",
        "轮次",
        "来源数",
        "过程事件数",
        "回答字数",
        "预计算(秒)",
        "查询改写(秒)",
        "嵌入(秒)",
        "正文块检索(秒)",
        "节点检索(秒)",
        "关键词(秒)",
        "扩展(秒)",
        "重排序(秒)",
        "检索合计(秒)",
        "错误信息",
        "回答摘要",
    ]
    for col, name in enumerate(detail_headers, start=1):
        detail.cell(1, col, name)
    style_header(detail, len(detail_headers))
    field_order = [
        "set_name",
        "question_id",
        "question",
        "ok",
        "client_total_s",
        "first_event_s",
        "server_latency_s",
        "retrieve_latency_s",
        "n_turns",
        "n_sources",
        "n_steps",
        "answer_chars",
        "precompute_s",
        "rewrite_s",
        "embed_s",
        "chunk_s",
        "node_s",
        "keyword_s",
        "expand_s",
        "rerank_s",
        "total_s",
        "http_error",
        "answer_preview",
    ]
    for i, row in enumerate(rows, start=1):
        detail.cell(i + 1, 1, i).border = thin
        for col, field in enumerate(field_order, start=2):
            value = row.get(field)
            if field == "ok":
                value = "成功" if value else "失败"
            elif isinstance(value, float):
                value = round(value, 3)
            cell = detail.cell(i + 1, col, value)
            cell.border = thin
            cell.alignment = wrap if field in {"question", "answer_preview", "http_error"} else Alignment(vertical="center")
        if not row.get("ok"):
            for col in range(1, 6):
                detail.cell(i + 1, col).fill = fail_fill
        detail.row_dimensions[i + 1].height = 32
    detail_widths = [
        8, 22, 28, 48, 10,
        16, 16, 16, 14,
        8, 10, 12, 12,
        12, 14, 12, 16, 14, 12, 12, 12, 14,
        28, 48,
    ]
    for i, width in enumerate(detail_widths, start=1):
        detail.column_dimensions[get_column_letter(i)].width = width

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def run(args: argparse.Namespace) -> None:
    username = args.user or os.environ.get("CGR_BENCH_USER", "")
    password = args.password or os.environ.get("CGR_BENCH_PASSWORD", "")
    if not username or not password:
        raise SystemExit("请设置 CGR_BENCH_USER 与 CGR_BENCH_PASSWORD，或传入 --user / --password")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    sample_path = out / "sample.json"
    jsonl_path = out / "results.jsonl"
    summary_path = out / "summary.json"
    excel_path = out / "生产端速度延迟测试.xlsx"

    if sample_path.is_file() and not args.resample:
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        questions = sample["questions"]
    else:
        questions = sample_questions(args.n_per_set, args.seed)
        sample = {
            "seed": args.seed,
            "n_per_set": args.n_per_set,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sets": [dict(spec) for spec in QUESTION_SETS],
            "questions": questions,
        }
        sample_path.write_text(
            json.dumps(sample, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    done = _load_done(jsonl_path)
    pending = [
        q for q in questions
        if f"{q['set_id']}::{q['question_id']}" not in done
    ]
    print(
        f"questions={len(questions)} done={len(done)} pending={len(pending)} "
        f"concurrency={args.concurrency}",
        flush=True,
    )

    client = Client(args.base_url, username, password, args.timeout)
    client.login()
    lock = threading.Lock()
    started = time.perf_counter()
    failures = 0

    def work(item: dict) -> dict:
        row = dict(item)
        row["started_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            measured = client.ask(item["question"])
        except Exception as exc:
            measured = {
                "ok": False,
                "http_error": f"{type(exc).__name__}: {exc}",
                "client_total_s": None,
                "first_event_s": None,
                "server_latency_s": None,
                "retrieve_latency_s": None,
                "n_turns": 0,
                "n_sources": 0,
                "n_steps": 0,
                "answer_chars": 0,
                "answer_preview": "",
                "status": 0,
            }
        row.update(measured)
        _append_jsonl(jsonl_path, row, lock)
        return row

    if pending:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(work, item) for item in pending]
            for i, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                flag = "ok" if row.get("ok") else "FAIL"
                if not row.get("ok"):
                    failures += 1
                elapsed = row.get("client_total_s")
                elapsed_s = f"{float(elapsed):.1f}s" if isinstance(elapsed, (int, float)) else "-"
                print(
                    f"[{len(done)+i}/{len(questions)}] {flag} {row.get('set_id')} "
                    f"{row.get('question_id')} {elapsed_s}",
                    flush=True,
                )

    rows = []
    done = _load_done(jsonl_path)
    for item in questions:
        key = f"{item['set_id']}::{item['question_id']}"
        if key not in done:
            raise SystemExit(f"缺少结果: {key}")
        rows.append(done[key])

    wall_s = time.perf_counter() - started
    # 续跑时墙钟只覆盖本次进程；完整墙钟用各题客户端耗时无法还原排队。
    # 若本次无待跑题，墙钟记 0，并在 meta 中保留 jsonl 的时间跨度。
    meta = {
        "base_url": args.base_url,
        "seed": sample.get("seed", args.seed),
        "n_per_set": sample.get("n_per_set", args.n_per_set),
        "concurrency": args.concurrency,
        "timeout_s": args.timeout,
        "started_at": sample.get("created_at"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "n_questions": len(rows),
    }
    summary = summarize(rows, wall_s=wall_s, meta=meta)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_excel(excel_path, rows, summary)
    print(f"excel={excel_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    overall = summary["groups"]["全部"]
    print(
        f"success={overall['n_ok']}/{overall['n']} "
        f"client_mean={overall['client_total_s']['mean']} "
        f"server_p50={overall['server_latency_s']['median']} "
        f"server_p90={overall['server_latency_s']['p90']}",
        flush=True,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="生产端 Agentic 速度与延迟抽样")
    parser.add_argument("--base-url", default="http://182.92.85.174:8060")
    parser.add_argument("--user", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--n-per-set", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--out",
        default="docs/prod_latency_20260922",
    )
    parser.add_argument("--resample", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args)


if __name__ == "__main__":
    main()
