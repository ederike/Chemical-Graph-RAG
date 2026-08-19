#!/usr/bin/env python3
"""打印识别 vLLM 当前占用：KV / running / waiting。

  python scripts/probe_vllm.py
  python scripts/probe_vllm.py --config example/a/config_open.yaml
  python scripts/probe_vllm.py --url http://113.108.154.212:8030
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_CONFIG = _ROOT / 'example' / 'a' / 'config_open.yaml'


def _metrics_base(url: str) -> str:
    u = (url or '').rstrip('/')
    if u.endswith('/v1'):
        u = u[:-3]
    return u.rstrip('/')


def _fetch(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={'User-Agent': 'probe_vllm'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode('utf-8', 'replace')


def _parse_prom(raw: str) -> dict:
    out = {}
    for line in raw.splitlines():
        if not line or line.startswith('#') or ' ' not in line:
            continue
        key, val = line.rsplit(' ', 1)
        try:
            out[key] = float(val)
        except ValueError:
            continue
    return out


def _val(metrics: dict, name: str):
    for key, val in metrics.items():
        if '_created' in key or '_bucket' in key:
            continue
        if key == name or key.startswith(name + '{'):
            return val
    return None


def _sum_count(metrics: dict, name: str):
    s = c = None
    for key, val in metrics.items():
        if key.startswith(name + '_sum'):
            s = val
        elif key.startswith(name + '_count'):
            c = val
    return s, c


def _success(metrics: dict) -> dict:
    out = {}
    prefix = 'vllm:request_success_total'
    for key, val in metrics.items():
        if not key.startswith(prefix):
            continue
        reason = 'total'
        if 'finished_reason="' in key:
            reason = key.split('finished_reason="', 1)[1].split('"', 1)[0]
        out[reason] = int(val)
    return out


def resolve_url(args) -> str:
    if args.url:
        return _metrics_base(args.url)
    cfg_path = Path(args.config)
    try:
        from src.utils.config import Config
        cfg = Config.from_yaml(str(cfg_path))
        base = getattr(getattr(cfg.doc, 'recognition', None), 'base_url', '') or ''
    except Exception:
        base = ''
        if cfg_path.is_file():
            for line in cfg_path.read_text(encoding='utf-8').splitlines():
                if 'base_url' in line and 'http' in line:
                    base = line.split(':', 1)[1].strip().strip('\'"')
                    break
    if not base:
        raise SystemExit(f'no base_url in {cfg_path}; pass --url')
    return _metrics_base(base)


def main():
    parser = argparse.ArgumentParser(description='Probe vLLM KV / queue occupancy')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--url', default='', help='server root or /v1 URL')
    parser.add_argument('--timeout', type=float, default=8.0)
    args = parser.parse_args()

    base = resolve_url(args)
    now = dt.datetime.now().strftime('%H:%M:%S')
    try:
        raw = _fetch(f'{base}/metrics', args.timeout)
    except Exception as e:
        print(f'[{now}] {base}  metrics FAIL: {e}')
        sys.exit(1)

    m = _parse_prom(raw)
    kv = _val(m, 'vllm:kv_cache_usage_perc')
    run = _val(m, 'vllm:num_requests_running')
    wait = _val(m, 'vllm:num_requests_waiting')
    ok = _success(m)
    q_sum, q_n = _sum_count(m, 'vllm:request_queue_time_seconds')
    e2e_sum, e2e_n = _sum_count(m, 'vllm:e2e_request_latency_seconds')
    dec_sum, dec_n = _sum_count(m, 'vllm:request_decode_time_seconds')

    load_s = ''
    try:
        load_s = _fetch(f'{base}/load', args.timeout).strip()
    except Exception:
        load_s = ''

    kv_s = f'{kv * 100:.2f}%' if kv is not None else 'n/a'
    run_s = int(run) if run is not None else 'n/a'
    wait_s = int(wait) if wait is not None else 'n/a'
    done = sum(ok.values())
    print(f'[{now}] {base}')
    print(f'  KV={kv_s}  running={run_s}  waiting={wait_s}  done={done} {ok}')
    extras = []
    if q_n:
        extras.append(f'queue_avg={q_sum / q_n:.2f}s')
    if e2e_n:
        extras.append(f'e2e_avg={e2e_sum / e2e_n:.1f}s')
    if dec_n:
        extras.append(f'decode_avg={dec_sum / dec_n:.1f}s')
    if load_s:
        try:
            extras.append(f'load={json.loads(load_s)}')
        except Exception:
            extras.append(f'load={load_s}')
    if extras:
        print('  ' + '  '.join(extras))


if __name__ == '__main__':
    main()
