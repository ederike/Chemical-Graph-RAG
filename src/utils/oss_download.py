"""
Query spider_product from MySQL and download files from Aliyun OSS.
Used by DHMF.download_from_oss(); does not run PDF recognition.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from .utils import TQDM_BAR_FORMAT

logger_default = logging.getLogger(__name__)

# Characters illegal in filenames on common OSes
_ILLEGAL_FILENAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

def _oss_object_key(oss_url: Any) -> str:
    """Normalize oss_url to an object key (strip scheme/host if a full URL)."""
    raw = (str(oss_url) if oss_url is not None else '').strip()
    if not raw:
        return ''
    # Full URL → path after host; otherwise treat as key.
    if '://' in raw:
        without_scheme = raw.split('://', 1)[1]
        slash = without_scheme.find('/')
        raw = without_scheme[slash + 1:] if slash >= 0 else ''
    return raw.lstrip('/')


def _file_md5(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _keep_key_for_dup_pdf(path: Path) -> tuple:
    """Prefer OSS-object basename (usually a hex hash) over legacy {name}_{id}.pdf."""
    stem = path.stem
    if re.fullmatch(r'[0-9a-fA-F]{16,}', stem):
        return (0, 0, path.name)
    if '_' in stem:
        tail = stem.rsplit('_', 1)[-1]
        if tail.isdigit():
            return (1, int(tail), path.name)
    return (2, 0, path.name)


def dedupe_local_pdfs(
    download_dir: Path,
    *,
    dry_run: bool = False,
    logger: logging.Logger = None,
) -> dict:
    """
    Collapse already-downloaded PDFs that are byte-identical.

    Keeps one file per content MD5 (smallest numeric name suffix), deletes the rest.
    Safe to run before insert so recognition does not send copies to the VLM.
    """
    log = logger or logger_default
    download_dir = Path(download_dir)
    summary = {
        'scanned': 0,
        'unique': 0,
        'deleted': 0,
        'kept': [],
        'removed': [],
    }
    if not download_dir.is_dir():
        return summary

    files = sorted(
        [p for p in download_dir.iterdir() if p.is_file() and p.suffix.lower() == '.pdf'],
        key=lambda p: p.name,
    )
    summary['scanned'] = len(files)
    groups: Dict[str, List[Path]] = {}
    for p in files:
        try:
            digest = _file_md5(p)
        except OSError as e:
            log.warning(f"[oss_download] skip hash {p.name}: {e}")
            continue
        groups.setdefault(digest, []).append(p)

    summary['unique'] = len(groups)
    for digest, paths in groups.items():
        if len(paths) <= 1:
            continue
        keep = min(paths, key=_keep_key_for_dup_pdf)
        for p in paths:
            if p == keep:
                continue
            summary['removed'].append(p.name)
            if not dry_run:
                try:
                    p.unlink()
                    summary['deleted'] += 1
                except OSError as e:
                    log.warning(f"[oss_download] failed to delete dup {p.name}: {e}")
            else:
                summary['deleted'] += 1
        summary['kept'].append(keep.name)
        log.info(
            f"[oss_download] local dup md5={digest[:10]}… keep={keep.name} "
            f"drop={len(paths) - 1}"
        )

    log.info(
        f"[oss_download] local dedupe: scanned={summary['scanned']} "
        f"unique={summary['unique']} deleted={summary['deleted']}"
        f"{' (dry_run)' if dry_run else ''}"
    )
    return summary


def filename_from_oss_key(object_key: str) -> Optional[str]:
    """
    Local filename = last path component of the OSS object key.

    tds/8970b3dc0324473bdadfdf547c008ce4.pdf
      → 8970b3dc0324473bdadfdf547c008ce4.pdf

    Same key (or same basename) always maps to the same file, so skip_existing
    and cross-table incremental download both work without {product}_{id}.
    """
    key = (object_key or '').strip()
    if not key:
        return None
    key = key.split('?', 1)[0].split('#', 1)[0].rstrip('/')
    name = Path(key).name
    if not name or name in ('.', '..'):
        return None
    name = _ILLEGAL_FILENAME_RE.sub('_', name).strip(' .')
    return name or None


def sanitize_product_filename(product_name: str, row_id: Any) -> Optional[str]:
    """Legacy {product}_{id}.pdf name. Kept for callers; download no longer uses it."""
    if row_id is None or str(row_id).strip() == '':
        return None
    rid = str(row_id).strip()

    if product_name is None:
        return f"null_name_{rid}.pdf"
    name = str(product_name).strip()
    if not name:
        return f"null_name_{rid}.pdf"
    if name.lower().endswith('.pdf'):
        name = name[:-4]
    name = _ILLEGAL_FILENAME_RE.sub('_', name).strip(' ._')
    if not name:
        return f"null_name_{rid}.pdf"
    return f"{name}_{rid}.pdf"

def _normalize_file_type(file_type: Union[str, int, None]) -> Optional[int]:
    """
    Return SQL type filter value, or None for all types.
    Accepts: all / 0 / 'all' / '0' → None; 1 / '1' → 1; 2 / '2' → 2
    """
    if file_type is None:
        return None
    if isinstance(file_type, str):
        ft = file_type.strip().lower()
        if ft in ('', 'all', '0', 'none', '*'):
            return None
        if ft in ('1', 'tds'):
            return 1
        if ft in ('2', 'msds'):
            return 2
        raise ValueError(f"Unsupported file_type: {file_type!r} (use 1, 2, or all)")
    iv = int(file_type)
    if iv == 0:
        return None
    if iv in (1, 2):
        return iv
    raise ValueError(f"Unsupported file_type: {file_type!r} (use 1, 2, or all)")

def _get_bucket_conf(ali_oss: dict, bucket_key: str) -> dict:
    if not ali_oss:
        raise ValueError("config.ali_oss is empty; set OSS credentials in yaml")
    # Support both 'ky-products-files' and nested dict as written in master_env
    conf = ali_oss.get(bucket_key)
    if conf is None and bucket_key == 'ky-products-files':
        conf = ali_oss.get('ky_products_files')
    if not conf:
        raise ValueError(
            f"OSS bucket {bucket_key!r} not found in config.ali_oss. "
            f"Keys: {list(ali_oss.keys())}"
        )
    return conf

def _mysql_table_columns(cur, table: str) -> set:
    """Return lowercase column names for `table` (current database)."""
    # SHOW COLUMNS is portable and does not require information_schema grants.
    cur.execute(f"SHOW COLUMNS FROM `{table}`")
    cols = set()
    for row in cur.fetchall() or []:
        name = row.get('Field') if isinstance(row, dict) else (row[0] if row else None)
        if name:
            cols.add(str(name).lower())
    return cols


def fetch_spider_products(
    mysql_conf: Any,
    *,
    table: str = 'spider_product',
    file_type: Union[str, int, None] = 'all',
    limit: int = 0,
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """
    SELECT downloadable rows from `table`.

    Optional columns are used only when they exist:
      - is_delete: WHERE is_delete = 0
      - type: AND type = file_type (1/2). Missing column → no type filter, download all.
    Required: id, product_name, oss_url.
    ORDER BY id ASC. limit<=0 means no LIMIT.
    """
    log = logger or logger_default
    try:
        import pymysql
    except ImportError as e:
        raise ImportError(
            "pymysql is required for download_from_oss. "
            "Install with: pip install pymysql"
        ) from e

    host = getattr(mysql_conf, 'host', None) or (mysql_conf.get('host') if isinstance(mysql_conf, dict) else '')
    port = getattr(mysql_conf, 'port', None) if not isinstance(mysql_conf, dict) else mysql_conf.get('port', 3306)
    user = getattr(mysql_conf, 'user', None) if not isinstance(mysql_conf, dict) else mysql_conf.get('user')
    password = getattr(mysql_conf, 'password', None) if not isinstance(mysql_conf, dict) else mysql_conf.get('password')
    db = getattr(mysql_conf, 'db', None) if not isinstance(mysql_conf, dict) else mysql_conf.get('db')

    if not host or not user or not db:
        raise ValueError("dm_data_mysql host/user/db must be set in config")

    port = int(port or 3306)
    type_filter = _normalize_file_type(file_type)

    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password or '',
        database=db,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=30,
        read_timeout=120,
    )
    try:
        with conn.cursor() as cur:
            columns = _mysql_table_columns(cur, table)
            required = ('id', 'product_name', 'oss_url')
            missing = [c for c in required if c not in columns]
            if missing:
                raise ValueError(
                    f"table `{table}` missing required column(s) {missing}; "
                    f"have {sorted(columns)}"
                )

            select_cols = list(required)
            for extra in ('type', 'is_delete'):
                if extra in columns:
                    select_cols.append(extra)

            sql = f"SELECT {', '.join(select_cols)} FROM `{table}`"
            where: list = []
            params: list = []
            if 'is_delete' in columns:
                where.append("is_delete = 0")
            if type_filter is not None and 'type' in columns:
                where.append("type = %s")
                params.append(type_filter)
            elif type_filter is not None and 'type' not in columns:
                log.info(
                    f"[oss_download] table `{table}` has no `type` column; "
                    f"ignore file_type={file_type!r}, download all"
                )
                type_filter = None
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY id ASC"
            if limit is not None and int(limit) > 0:
                sql += " LIMIT %s"
                params.append(int(limit))

            log.info(
                f"[oss_download] query {table}: type="
                f"{type_filter if type_filter is not None else 'all'}, "
                f"limit={limit if limit and int(limit) > 0 else 'none'}, "
                f"host={host}/{db}"
            )
            cur.execute(sql, params)
            rows = list(cur.fetchall() or [])
    finally:
        conn.close()

    log.info(f"[oss_download] fetched {len(rows)} row(s) from {table}")
    return rows

def download_rows_from_oss(
    rows: List[Dict[str, Any]],
    *,
    ali_oss: dict,
    bucket_key: str = 'ky-products-files',
    download_dir: Path,
    skip_existing: bool = True,
    num_thread: int = 16,
    logger: logging.Logger = None,
) -> dict:
    """
    Download each unique OSS object once. Local name = basename of oss_url
    (e.g. tds/8970b3dc….pdf → 8970b3dc….pdf).

    Same object key / same basename → one file. Later rows skipped_duplicate.
    skip_existing: dest already on disk (works across tables / reruns).
    num_thread: parallel GET workers (1 = serial).
    Returns summary dict.
    """
    log = logger or logger_default
    try:
        import oss2
    except ImportError as e:
        raise ImportError(
            "oss2 is required for download_from_oss. "
            "Install with: pip install oss2"
        ) from e

    conf = _get_bucket_conf(ali_oss, bucket_key)
    access_key = conf.get('ACCESS_KEY') or conf.get('access_key') or ''
    secret_key = conf.get('SECRET_KEY') or conf.get('secret_key') or ''
    endpoint = conf.get('END_POINT') or conf.get('end_point') or conf.get('endpoint') or ''
    bucket_name = conf.get('BUCKET_NAME') or conf.get('bucket_name') or ''
    if not all([access_key, secret_key, endpoint, bucket_name]):
        raise ValueError(
            f"Incomplete OSS config for {bucket_key}: need ACCESS_KEY, SECRET_KEY, END_POINT, BUCKET_NAME"
        )

    try:
        workers = max(1, int(num_thread or 1))
    except (TypeError, ValueError):
        workers = 16

    auth = oss2.Auth(access_key, secret_key)
    # oss2.Bucket is not documented as thread-safe; one client per worker thread.
    tls = threading.local()

    def _thread_bucket():
        b = getattr(tls, 'bucket', None)
        if b is None:
            b = oss2.Bucket(auth, endpoint, bucket_name)
            tls.bucket = b
        return b

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'total': len(rows),
        'downloaded': 0,
        'skipped_existing': 0,
        'skipped_duplicate': 0,
        'skipped_invalid': 0,
        'failed': 0,
        'files': [],
        'num_thread': workers,
    }

    seen_keys: Dict[str, str] = {}
    seen_names: Dict[str, str] = {}
    jobs: List[Tuple[Any, str, Path]] = []  # (row_id, object_key, dest)

    for row in rows:
        row_id = row.get('id')
        object_key = _oss_object_key(row.get('oss_url'))

        if not object_key:
            log.warning(
                f"[oss_download] skip id={row_id}: empty oss_url"
            )
            summary['skipped_invalid'] += 1
            continue

        if object_key in seen_keys:
            summary['skipped_duplicate'] += 1
            log.debug(
                f"[oss_download] skip duplicate key id={row_id} "
                f"key={object_key!r} already={seen_keys[object_key]}"
            )
            continue

        filename = filename_from_oss_key(object_key)
        if not filename:
            log.warning(
                f"[oss_download] skip id={row_id}: cannot derive filename "
                f"from key={object_key!r}"
            )
            summary['skipped_invalid'] += 1
            continue

        if filename in seen_names:
            summary['skipped_duplicate'] += 1
            log.debug(
                f"[oss_download] skip duplicate name id={row_id} "
                f"key={object_key!r} already={seen_names[filename]}"
            )
            continue

        seen_keys[object_key] = filename
        seen_names[filename] = filename
        dest = download_dir / filename

        if skip_existing and dest.exists():
            log.debug(f"[oss_download] skip existing: {dest.name}")
            summary['skipped_existing'] += 1
            continue

        jobs.append((row_id, object_key, dest))

    def _download_one(job: Tuple[Any, str, Path]):
        row_id, object_key, dest = job
        # Recheck on the worker: another thread / previous run may have written dest.
        if skip_existing and dest.exists():
            return 'skip', dest, None
        tmp = dest.with_name(dest.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
        try:
            _thread_bucket().get_object_to_file(object_key, str(tmp))
            os.replace(str(tmp), str(dest))
            return 'ok', dest, None
        except Exception as e:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            return 'fail', dest, (row_id, object_key, e)

    log.info(
        f"[oss_download] start downloads={len(jobs)} "
        f"threads={workers} unique_keys={len(seen_keys)} "
        f"already_on_disk={summary['skipped_existing']} "
        f"dup_rows={summary['skipped_duplicate']}"
    )

    bar = tqdm(
        total=len(jobs),
        desc='oss_download',
        unit='file',
        bar_format=TQDM_BAR_FORMAT,
    )
    try:
        results = []
        n_ok = 0
        n_fail = 0
        n_skip = 0

        def _note(r):
            nonlocal n_ok, n_fail, n_skip
            results.append(r)
            if r[0] == 'ok':
                n_ok += 1
            elif r[0] == 'skip':
                n_skip += 1
            else:
                n_fail += 1
            bar.update(1)
            bar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)

        if workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                _note(_download_one(j))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_download_one, j) for j in jobs]
                for fut in as_completed(futures):
                    _note(fut.result())
    finally:
        bar.close()

    for status, dest, err in results:
        if status == 'ok':
            summary['downloaded'] += 1
            summary['files'].append(str(dest))
        elif status == 'skip':
            summary['skipped_existing'] += 1
        else:
            summary['failed'] += 1
            row_id, object_key, exc = err
            log.error(
                f"[oss_download] failed id={row_id} key={object_key!r}: {exc}"
            )

    log.info(
        f"[oss_download] done: downloaded={summary['downloaded']} "
        f"skipped_existing={summary['skipped_existing']} "
        f"skipped_duplicate={summary['skipped_duplicate']} "
        f"skipped_invalid={summary['skipped_invalid']} "
        f"failed={summary['failed']} / total={summary['total']} "
        f"unique_keys={len(seen_keys)} threads={workers}"
    )
    return summary
