#!/usr/bin/env python3
"""
从 DHMF 主库（SQLite）按 doc_id 导出文档及其关联超边、块、节点到 JSON。

输出仅保留 name / content：
{
  "doc": {"name": "...", "content": "..."},
  "hyperedges": [{"name": "...", "content": "..."}, ...],
  "chunks": [...],
  "nodes": [...]
}

示例：
  python scripts/export_doc_subgraph.py --doc-id 1
  python scripts/export_doc_subgraph.py --db example/a/DB/main.db --doc-id 5 -o out.json
  python scripts/export_doc_subgraph.py --doc-name "some.pdf"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_db() -> Path:
    return _project_root() / "example" / "a" / "DB" / "main.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def _name_content(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "name": row["name"] if row["name"] is not None else "",
        "content": row["content"] if row["content"] is not None else "",
    }


def _fetch_name_content(
    conn: sqlite3.Connection,
    table: str,
    doc_id: int,
) -> List[Dict[str, Any]]:
    if not _table_exists(conn, table):
        return []
    rows = conn.execute(
        f"SELECT name, content FROM {table} WHERE doc_id = ? ORDER BY id",
        (doc_id,),
    ).fetchall()
    return [_name_content(r) for r in rows]


def resolve_doc_ids(
    conn: sqlite3.Connection,
    doc_ids: Optional[Sequence[int]],
    doc_names: Optional[Sequence[str]],
) -> List[int]:
    ids: List[int] = []
    seen = set()

    if doc_ids:
        for did in doc_ids:
            did = int(did)
            if did not in seen:
                ids.append(did)
                seen.add(did)

    if doc_names:
        for name in doc_names:
            rows = conn.execute(
                "SELECT id FROM doc WHERE name = ?", (name,)
            ).fetchall()
            if not rows:
                raise ValueError(f"No doc found with name={name!r}")
            for r in rows:
                did = int(r["id"])
                if did not in seen:
                    ids.append(did)
                    seen.add(did)

    if not ids:
        raise ValueError("Must provide at least one --doc-id or --doc-name")
    return ids


def export_one_doc(conn: sqlite3.Connection, doc_id: int) -> Dict[str, Any]:
    doc_rows = conn.execute(
        "SELECT name, content FROM doc WHERE id = ?", (doc_id,)
    ).fetchall()
    if not doc_rows:
        raise ValueError(f"No doc found with id={doc_id}")

    return {
        "doc": _name_content(doc_rows[0]),
        "hyperedges": _fetch_name_content(conn, "hyperedge", doc_id),
        "chunks": _fetch_name_content(conn, "chunk", doc_id),
        "nodes": _fetch_name_content(conn, "node", doc_id),
    }


def _default_output(doc_id: int) -> Path:
    out_dir = _project_root() / "scripts" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"doc_{doc_id}.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export doc/hyperedges/chunks/nodes (name+content only) by doc_id",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=_default_db(),
        help=f"Path to main.db (default: {_default_db()})",
    )
    p.add_argument(
        "--doc-id",
        type=int,
        nargs="+",
        default=None,
        help="One or more document ids (each writes its own file if multiple)",
    )
    p.add_argument(
        "--doc-name",
        type=str,
        nargs="+",
        default=None,
        help="One or more document names (resolved to ids)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (only valid for a single doc; default: scripts/outputs/doc_<id>.json)",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (default: 2; use 0 for compact)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    db_path = args.db if args.db.is_absolute() else _project_root() / args.db

    try:
        conn = _connect(db_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        doc_ids = resolve_doc_ids(conn, args.doc_id, args.doc_name)
        if args.output is not None and len(doc_ids) > 1:
            print(
                "Error: --output can only be used with a single document",
                file=sys.stderr,
            )
            return 1

        indent = None if args.indent <= 0 else args.indent
        print(f"db: {db_path}")

        for did in doc_ids:
            try:
                payload = export_one_doc(conn, int(did))
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

            if args.output is not None:
                out_path = args.output
            else:
                out_path = _default_output(int(did))
            if not out_path.is_absolute():
                out_path = _project_root() / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=indent)
                f.write("\n")

            print(
                f"  doc_id={did} name={payload['doc']['name']!r} "
                f"hyperedges={len(payload['hyperedges'])} "
                f"chunks={len(payload['chunks'])} "
                f"nodes={len(payload['nodes'])} "
                f"-> {out_path}"
            )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
