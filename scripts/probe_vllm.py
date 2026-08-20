#!/usr/bin/env python3
"""打印识别 vLLM 当前占用：KV / running / waiting，并测一段窗口的 token/s。

  python scripts/probe_vllm.py
  python scripts/probe_vllm.py --window 12
  python scripts/probe_vllm.py --config example/a/config_open.yaml
  python scripts/probe_vllm.py --url http://113.108.154.212:8030
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
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


def _fetch_metrics(base: str, timeout: float) -> dict:
    return _parse_prom(_fetch(f'{base}/metrics', timeout))


def _rate(delta, dt):
    if delta is None or dt <= 0:
        return None
    return delta / dt


def _fmt_rate(v):
    return 'n/a' if v is None else f'{v:.1f}'


def measure_throughput(base: str, timeout: float, window: float, m0=None, t0=None):
    """Sample generation / prompt counters over `window` seconds."""
    if m0 is None or t0 is None:
        t0 = time.time()
        m0 = _fetch_metrics(base, timeout)
    remain = window - (time.time() - t0)
    if remain > 0:
        time.sleep(remain)
    t1 = time.time()
    m1 = _fetch_metrics(base, timeout)
    dt = t1 - t0

    g0 = _val(m0, 'vllm:generation_tokens_total')
    g1 = _val(m1, 'vllm:generation_tokens_total')
    p0 = _val(m0, 'vllm:prompt_tokens_total')
    p1 = _val(m1, 'vllm:prompt_tokens_total')
    r0 = _val(m0, 'vllm:num_requests_running')
    r1 = _val(m1, 'vllm:num_requests_running')
    w0 = _val(m0, 'vllm:num_requests_waiting')
    w1 = _val(m1, 'vllm:num_requests_waiting')
    kv0 = _val(m0, 'vllm:kv_cache_usage_perc')
    kv1 = _val(m1, 'vllm:kv_cache_usage_perc')

    gen_d = None if g0 is None or g1 is None else g1 - g0
    prompt_d = None if p0 is None or p1 is None else p1 - p0
    gen_ps = _rate(gen_d, dt)
    prompt_ps = _rate(prompt_d, dt)
    r_avg = None
    if r0 is not None or r1 is not None:
        r_avg = ((r0 or 0) + (r1 or 0)) / 2.0
    per = None if gen_ps is None or not r_avg else gen_ps / r_avg
    return {
        'dt': dt,
        'm1': m1,
        'kv0': kv0,
        'kv1': kv1,
        'r0': r0,
        'r1': r1,
        'w0': w0,
        'w1': w1,
        'r_avg': r_avg,
        'gen_d': gen_d,
        'prompt_d': prompt_d,
        'gen_ps': gen_ps,
        'prompt_ps': prompt_ps,
        'per_running': per,
    }


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
    parser = argparse.ArgumentParser(
        description='Probe vLLM occupancy and token throughput')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--url', default='', help='server root or /v1 URL')
    parser.add_argument('--timeout', type=float, default=8.0)
    parser.add_argument(
        '--window', type=float, default=8.0,
        help='seconds to sample generation tokens (0 = occupancy only)')
    args = parser.parse_args()

    base = resolve_url(args)
    now = dt.datetime.now().strftime('%H:%M:%S')
    t0 = time.time()
    try:
        m = _fetch_metrics(base, args.timeout)
    except Exception as e:
        print(f'[{now}] {base}  metrics FAIL: {e}')
        sys.exit(1)

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

    if args.window > 0:
        try:
            tp = measure_throughput(
                base, args.timeout, args.window, m0=m, t0=t0)
        except Exception as e:
            print(f'  throughput FAIL: {e}')
        else:
            r0 = int(tp['r0']) if tp['r0'] is not None else 'n/a'
            r1 = int(tp['r1']) if tp['r1'] is not None else 'n/a'
            w0 = int(tp['w0']) if tp['w0'] is not None else 'n/a'
            w1 = int(tp['w1']) if tp['w1'] is not None else 'n/a'
            print(
                f'  throughput  window={tp["dt"]:.1f}s  '
                f'running {r0}→{r1}  waiting {w0}→{w1}'
            )
            per_s = (
                f'{tp["per_running"]:.2f}'
                if tp['per_running'] is not None else 'n/a'
            )
            gen_d = 'n/a' if tp['gen_d'] is None else f'{tp["gen_d"]:.0f}'
            print(
                f'  gen={_fmt_rate(tp["gen_ps"])} tok/s  '
                f'per_running={per_s}  '
                f'prompt={_fmt_rate(tp["prompt_ps"])} tok/s  '
                f'delta_gen={gen_d}'
            )


if __name__ == '__main__':
    main()
