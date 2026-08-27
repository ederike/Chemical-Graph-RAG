"""
Rewrite FAISS shard files to a new size without re-embedding.

Examples:
  # merge 100000-vector shards into 500000-vector shards
  python scripts/repack_vdb.py --config example/a/config_open.yaml --shard-max 500000

  # one monolithic file per store (needs RAM for the full index)
  python scripts/repack_vdb.py --config example/a/config_open.yaml --mono

  # only node
  python scripts/repack_vdb.py --config example/a/config_open.yaml --shard-max 500000 --name node

After a larger size (or --mono), set vectorization.shard_max_vectors in the
yaml to the same value (or remove / set 0 for mono). Otherwise the next
vectorization() still seals new writes at the old 100000.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.DHMF import DHMF
from src.utils.config import Config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='pipeline yaml, e.g. example/a/config_open.yaml')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        '--shard-max',
        type=int,
        dest='shard_max',
        help='max vectors per rewritten shard (e.g. 500000)',
    )
    g.add_argument(
        '--mono',
        action='store_true',
        help='merge each store into a single {name}.vdb',
    )
    p.add_argument(
        '--name',
        default=None,
        help="one store: doc/chunk/hyperedge/node/edge (default: all)",
    )
    p.add_argument(
        '--force',
        action='store_true',
        help='rewrite even if the current layout already matches',
    )
    args = p.parse_args()

    dest = 0 if args.mono else int(args.shard_max)
    if dest < 0:
        p.error('--shard-max must be >= 0')

    config = Config.from_yaml(args.config)
    graph = DHMF(config)
    out = graph.repack_vectors(
        args.name,
        shard_max_vectors=dest,
        force=args.force,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
