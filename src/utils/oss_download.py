"""
Query spider_product from MySQL and download files from Aliyun OSS.
Used by DHMF.download_from_oss(); does not run PDF recognition.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger_default = logging.getLogger(__name__)

# Characters illegal in filenames on common OSes
_ILLEGAL_FILENAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

def sanitize_product_filename(product_name: str, row_id: Any) -> Optional[str]:
    """
    Build local filename: {sanitized_product_name}_{id}.pdf
    - empty / None product_name → null_name_{id}.pdf（仍下载，不用产品名）
    - illegal path chars → '_'
    - trailing .pdf stripped before appending _{id}.pdf (no double suffix)
    - row_id 为空时无法生成文件名 → None
    """
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

def fetch_spider_products(
    mysql_conf: Any,
    *,
    table: str = 'spider_product',
    file_type: Union[str, int, None] = 'all',
    limit: int = 0,
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """
    SELECT from spider_product where is_delete=0, optional type filter,
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

    sql = (
        f"SELECT id, product_name, oss_url, type, is_delete "
        f"FROM `{table}` WHERE is_delete = 0"
    )
    params: list = []
    if type_filter is not None:
        sql += " AND type = %s"
        params.append(type_filter)
    sql += " ORDER BY id ASC"
    if limit is not None and int(limit) > 0:
        sql += " LIMIT %s"
        params.append(int(limit))

    log.info(
        f"[oss_download] query {table}: type={type_filter if type_filter is not None else 'all'}, "
        f"limit={limit if limit and int(limit) > 0 else 'none'}, host={host}/{db}"
    )

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
    logger: logging.Logger = None,
) -> dict:
    """
    Download each row's oss_url object key into download_dir as {product_name}_{id}.pdf.
    product_name 为空时用 null_name_{id}.pdf，不跳过。
    Skip when id 无效、oss_url empty, or local file already exists.
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

    auth = oss2.Auth(access_key, secret_key)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'total': len(rows),
        'downloaded': 0,
        'skipped_existing': 0,
        'skipped_invalid': 0,
        'null_name': 0,
        'failed': 0,
        'files': [],
    }

    for row in rows:
        row_id = row.get('id')
        product_name = row.get('product_name')
        oss_url = (row.get('oss_url') or '').strip()
        filename = sanitize_product_filename(product_name, row_id)

        if not filename:
            log.warning(
                f"[oss_download] skip id={row_id!r}: missing id (cannot build filename)"
            )
            summary['skipped_invalid'] += 1
            continue
        if filename.startswith('null_name_'):
            summary['null_name'] = summary.get('null_name', 0) + 1
            log.info(
                f"[oss_download] empty product_name id={row_id} -> {filename}"
            )
        if not oss_url:
            log.warning(
                f"[oss_download] skip id={row_id}: empty oss_url"
            )
            summary['skipped_invalid'] += 1
            continue

        # oss_url is object key, e.g. tds/4e6495f1....pdf
        object_key = oss_url.lstrip('/')
        dest = download_dir / filename

        if skip_existing and dest.exists():
            log.info(f"[oss_download] skip existing: {dest.name}")
            summary['skipped_existing'] += 1
            continue

        try:
            log.info(
                f"[oss_download] download id={row_id} key={object_key!r} -> {dest.name}"
            )
            bucket.get_object_to_file(object_key, str(dest))
            summary['downloaded'] += 1
            summary['files'].append(str(dest))
        except Exception as e:
            summary['failed'] += 1
            log.error(
                f"[oss_download] failed id={row_id} key={object_key!r}: {e}"
            )

    log.info(
        f"[oss_download] done: downloaded={summary['downloaded']} "
        f"skipped_existing={summary['skipped_existing']} "
        f"skipped_invalid={summary['skipped_invalid']} "
        f"null_name={summary.get('null_name', 0)} "
        f"failed={summary['failed']} / total={summary['total']}"
    )
    return summary
