#!/usr/bin/env python3
"""
从 DHMF 主库（SQLite）绘制超图可视化。

层级约定：
  - 大圆 = 超边（= 文档头块；不重复画 head chunk）
  - 中方块 = 正文块（body chunks only），环绕所属超边
  - 小圆 = 实体节点（同名合并）；头块上的节点挂在超边上，正文块节点挂在块上

示例：
  python scripts/draw_hypergraph.py --db example/b/DB/main.db
  python scripts/draw_hypergraph.py --db example/a/DB/main.db --max-hyperedges 20 --output outputs/hypergraph_20.png
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _configure_chinese_font():
    try:
        from matplotlib import font_manager
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    local_font_dirs = [
        _project_root() / "assets" / "fonts",
        Path(__file__).resolve().parent / "fonts",
    ]
    local_patterns = (
        "NotoSansSC-Regular.otf",
        "NotoSansSC-Regular.ttf",
        "NotoSansCJKsc-Regular.otf",
        "SourceHanSansSC-Regular.otf",
        "*.otf",
        "*.ttf",
        "*.ttc",
    )
    for font_dir in local_font_dirs:
        if not font_dir.is_dir():
            continue
        for pattern in local_patterns:
            for font_path in sorted(font_dir.glob(pattern)):
                try:
                    font_manager.fontManager.addfont(str(font_path))
                    prop = font_manager.FontProperties(fname=str(font_path))
                    family = prop.get_name()
                    plt.rcParams["font.family"] = "sans-serif"
                    plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
                    plt.rcParams["axes.unicode_minus"] = False
                    return f"{family} ({font_path.name})"
                except Exception:
                    continue

    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Arial Unicode MS",
        "Droid Sans Fallback",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name

    for f in font_manager.fontManager.ttflist:
        low = f.name.lower()
        if any(k in low for k in ("cjk", "noto sans sc", "wqy", "hei", "yahei", "simsun")):
            plt.rcParams["font.sans-serif"] = [f.name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return f.name

    plt.rcParams["axes.unicode_minus"] = False
    return None


def _parse_extra(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def load_hypergraph(db_path: Path):
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for required in ("hyperedge", "node", "chunk"):
        if required not in tables:
            con.close()
            raise ValueError(f"数据库缺少表 `{required}`: {db_path}")

    hyperedges = [dict(r) for r in cur.execute(
        "SELECT id, doc_id, chunk_id, name, content FROM hyperedge ORDER BY id"
    )]
    chunks = [dict(r) for r in cur.execute(
        "SELECT id, doc_id, name, content, extra FROM chunk ORDER BY id"
    )]
    nodes = [dict(r) for r in cur.execute(
        "SELECT id, doc_id, chunk_id, hyperedge_id, name, content FROM node ORDER BY id"
    )]

    doc_names: Dict[int, str] = {}
    if "doc" in tables:
        for r in cur.execute("SELECT id, name FROM doc"):
            doc_names[int(r["id"])] = r["name"] or f"doc_{r['id']}"

    con.close()
    return hyperedges, chunks, nodes, doc_names


def _is_head_chunk(chunk: dict, head_chunk_ids: Set[int]) -> bool:
    cid = chunk.get("id")
    if cid is not None and int(cid) in head_chunk_ids:
        return True
    name = (chunk.get("name") or "").strip().lower()
    if name == "head":
        return True
    extra = _parse_extra(chunk.get("extra"))
    return bool(extra.get("is_head") or extra.get("role") == "head")


def _chunk_order_key(chunk: dict, head_chunk_ids: Set[int]):
    extra = _parse_extra(chunk.get("extra"))
    idx = extra.get("chunk_index")
    if idx is None:
        name = (chunk.get("name") or "").strip().lower()
        if name == "head" or _is_head_chunk(chunk, head_chunk_ids):
            idx = 0
        elif name.startswith("body_"):
            try:
                idx = int(name.split("_", 1)[1])
            except Exception:
                idx = 10**9
        else:
            idx = 10**9
    try:
        idx = int(idx)
    except Exception:
        idx = 10**9
    return (idx, chunk.get("id") or 0)


def build_graph(
    hyperedges: List[dict],
    chunks: List[dict],
    nodes: List[dict],
    *,
    max_hyperedges: Optional[int] = None,
    hyperedge_ids: Optional[Set[int]] = None,
    max_nodes: Optional[int] = None,
    name_contains: Optional[str] = None,
    min_degree: int = 1,
):
    """
    返回：
      he_ids, he_meta
      body_chunk_ids（不含头块；超边即头块，不再单独画头块）
      chunk_meta, he_to_body_chunks
      entity_names, entity_meta
      membership: (entity_name, target)  target 为 ('he', hid) 或 ('ch', cid)
    """
    he_by_id = {int(h["id"]): h for h in hyperedges}
    ordered_he = sorted(he_by_id.keys())
    if hyperedge_ids is not None:
        ordered_he = [hid for hid in ordered_he if hid in hyperedge_ids]
    if max_hyperedges is not None and max_hyperedges > 0:
        ordered_he = ordered_he[:max_hyperedges]
    he_keep = set(ordered_he)
    he_meta = {hid: he_by_id[hid] for hid in ordered_he}

    head_chunk_ids: Set[int] = set()
    head_chunk_to_he: Dict[int, int] = {}
    for hid, h in he_meta.items():
        if h.get("chunk_id") is not None:
            hcid = int(h["chunk_id"])
            head_chunk_ids.add(hcid)
            head_chunk_to_he[hcid] = hid

    # doc_id -> hyperedge_id（一文档一超边）
    doc_to_he: Dict[int, int] = {}
    for hid, h in he_meta.items():
        did = h.get("doc_id")
        if did is not None and int(did) not in doc_to_he:
            doc_to_he[int(did)] = hid

    chunk_by_id = {int(c["id"]): c for c in chunks if c.get("id") is not None}

    # 所有属于保留超边的 chunk（含头块，用于节点归属判断）
    he_to_all_chunks: Dict[int, List[int]] = defaultdict(list)
    for c in chunks:
        cid = c.get("id")
        did = c.get("doc_id")
        if cid is None or did is None:
            continue
        hid = doc_to_he.get(int(did))
        if hid is None or hid not in he_keep:
            continue
        he_to_all_chunks[hid].append(int(cid))

    for hid in he_to_all_chunks:
        he_to_all_chunks[hid].sort(
            key=lambda cid: _chunk_order_key(chunk_by_id[cid], head_chunk_ids)
        )

    all_kept_set = set()
    for cids in he_to_all_chunks.values():
        all_kept_set.update(cids)

    # 画图用的块：去掉头块（超边 = 头块，不重复画）
    he_to_body_chunks: Dict[int, List[int]] = {}
    body_chunk_ids: List[int] = []
    chunk_meta = {}
    for hid in ordered_he:
        bodies = []
        for cid in he_to_all_chunks.get(hid, []):
            c = chunk_by_id.get(cid)
            if c is None:
                continue
            is_head = _is_head_chunk(c, head_chunk_ids)
            chunk_meta[cid] = {
                "chunk": c,
                "hyperedge_id": hid,
                "is_head": is_head,
                "label": (c.get("name") or f"chunk_{cid}"),
            }
            if not is_head:
                bodies.append(cid)
                body_chunk_ids.append(cid)
        he_to_body_chunks[hid] = bodies

    # 实体：同名合并
    name_to_chunks: Dict[str, Set[int]] = defaultdict(set)
    name_to_hes: Dict[str, Set[int]] = defaultdict(set)
    name_to_count: Dict[str, int] = defaultdict(int)
    name_to_sample: Dict[str, str] = {}

    for n in nodes:
        name = (n.get("name") or "").strip()
        if not name:
            continue
        if name_contains and name_contains not in name:
            continue
        cid = n.get("chunk_id")
        if cid is None:
            continue
        cid = int(cid)
        if cid not in all_kept_set:
            continue
        name_to_chunks[name].add(cid)
        name_to_count[name] += 1
        hid = chunk_meta.get(cid, {}).get("hyperedge_id")
        if hid is None and cid in head_chunk_to_he:
            hid = head_chunk_to_he[cid]
        if hid is not None:
            name_to_hes[name].add(hid)
        if name not in name_to_sample and n.get("content"):
            name_to_sample[name] = str(n["content"])

    entities = list(name_to_chunks.keys())
    if min_degree > 1:
        entities = [n for n in entities if len(name_to_hes[n]) >= min_degree]

    entities.sort(key=lambda nm: (-len(name_to_hes[nm]), -name_to_count[nm], nm))
    if max_nodes is not None and max_nodes > 0:
        entities = entities[:max_nodes]

    # membership：头块上的节点挂到超边，正文块节点挂到块
    membership: List[Tuple[str, Tuple[str, int]]] = []
    for name in entities:
        for cid in sorted(name_to_chunks[name]):
            if cid in head_chunk_ids or chunk_meta.get(cid, {}).get("is_head"):
                hid = head_chunk_to_he.get(cid) or chunk_meta.get(cid, {}).get("hyperedge_id")
                if hid is not None:
                    membership.append((name, ("he", int(hid))))
            else:
                membership.append((name, ("ch", cid)))

    entity_meta = {}
    for name in entities:
        anchors = []
        seen = set()
        for cid in sorted(name_to_chunks[name]):
            if cid in head_chunk_ids or chunk_meta.get(cid, {}).get("is_head"):
                hid = head_chunk_to_he.get(cid) or chunk_meta.get(cid, {}).get("hyperedge_id")
                if hid is None:
                    continue
                key = ("he", int(hid))
            else:
                key = ("ch", cid)
            if key not in seen:
                seen.add(key)
                anchors.append(key)
        entity_meta[name] = {
            "degree": len(name_to_hes[name]),
            "chunk_degree": len(name_to_chunks[name]),
            "count": name_to_count[name],
            "content": name_to_sample.get(name, ""),
            "chunk_ids": sorted(name_to_chunks[name]),
            "hyperedge_ids": sorted(name_to_hes[name]),
            "anchors": anchors,
        }

    return (
        ordered_he,
        he_meta,
        body_chunk_ids,
        chunk_meta,
        he_to_body_chunks,
        entities,
        entity_meta,
        membership,
    )


def _clip(s: str, max_len: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _he_label(hid: int, he_meta: dict, doc_names: Dict[int, str], max_len: int = 16) -> str:
    """超边 = 文档头块标记。"""
    he = he_meta.get(hid, {})
    doc_id = he.get("doc_id")
    if doc_id is not None and int(doc_id) in doc_names:
        base = doc_names[int(doc_id)]
    elif he.get("name") and str(he["name"]).lower() != "head":
        base = str(he["name"])
    else:
        base = f"HE-{hid}"
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    return f"{_clip(base, max_len)}\n头块/超边\n(he={hid})"


def _chunk_label(cid: int, chunk_meta: dict, max_len: int = 10) -> str:
    meta = chunk_meta.get(cid, {})
    label = meta.get("label") or f"c{cid}"
    return _clip(str(label), max_len) + f"\n{cid}"


def _ring_plan(n: int, inner_r: float, spacing: float) -> List[Tuple[float, int]]:
    if n <= 0:
        return []
    rings: List[Tuple[float, int]] = []
    left = n
    r = inner_r
    while left > 0:
        cap = max(3, int(2 * math.pi * r / max(spacing, 1e-6)))
        take = min(left, cap)
        rings.append((r, take))
        left -= take
        r += spacing
    return rings


def _place_on_rings(
    keys: List,
    center: Tuple[float, float],
    *,
    inner_r: float,
    spacing: float,
    angle_offset: float = 0.0,
) -> Dict:
    cx, cy = center
    out = {}
    rings = _ring_plan(len(keys), inner_r, spacing)
    idx = 0
    for ring_i, (r, count) in enumerate(rings):
        batch = keys[idx: idx + count]
        idx += count
        m = len(batch)
        phase = angle_offset + (math.pi / m if (ring_i % 2 and m) else 0.0)
        for j, key in enumerate(batch):
            ang = phase + 2 * math.pi * j / m
            out[key] = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    return out


def _separate_positions(
    movable: Dict[str, Tuple[float, float]],
    fixed: Dict[str, Tuple[float, float]],
    *,
    min_dist: float,
    iterations: int = 80,
    step: float = 0.12,
) -> Dict[str, Tuple[float, float]]:
    pos = {k: (float(v[0]), float(v[1])) for k, v in movable.items()}
    if not pos:
        return pos
    keys = list(pos.keys())
    fixed_items = list(fixed.items())

    for _ in range(iterations):
        deltas = {k: [0.0, 0.0] for k in keys}
        for i in range(len(keys)):
            ki = keys[i]
            xi, yi = pos[ki]
            for j in range(i + 1, len(keys)):
                kj = keys[j]
                xj, yj = pos[kj]
                dx, dy = xi - xj, yi - yj
                dist = math.hypot(dx, dy)
                if dist < 1e-9:
                    dx, dy = 0.01 * ((i + 1) % 7 + 1), 0.01 * ((j + 3) % 5 + 1)
                    dist = math.hypot(dx, dy)
                if dist < min_dist:
                    push = (min_dist - dist) / dist
                    fx, fy = dx * push * 0.5, dy * push * 0.5
                    deltas[ki][0] += fx
                    deltas[ki][1] += fy
                    deltas[kj][0] -= fx
                    deltas[kj][1] -= fy
        for ki in keys:
            xi, yi = pos[ki]
            for _, (xf, yf) in fixed_items:
                dx, dy = xi - xf, yi - yf
                dist = math.hypot(dx, dy)
                if dist < 1e-9:
                    dx, dy = 0.02, 0.01
                    dist = math.hypot(dx, dy)
                need = min_dist * 1.05
                if dist < need:
                    push = (need - dist) / dist
                    deltas[ki][0] += dx * push
                    deltas[ki][1] += dy * push

        max_move = 0.0
        for k in keys:
            dx, dy = deltas[k]
            mag = math.hypot(dx, dy)
            if mag > step:
                dx, dy = dx / mag * step, dy / mag * step
                mag = step
            pos[k] = (pos[k][0] + dx, pos[k][1] + dy)
            max_move = max(max_move, mag)
        if max_move < 1e-3:
            break
    return pos


def layout_positions(
    he_ids: List[int],
    he_to_body_chunks: Dict[int, List[int]],
    chunk_meta: dict,
    entity_names: List[str],
    entity_meta: dict,
    *,
    he_r: float = 0.55,
    chunk_half: float = 0.28,
    node_spacing: float = 0.42,
) -> Dict[str, Tuple[float, float]]:
    """
    布局（超边 = 头块，不另画头块方块）：
      1. 超边大圆中心
      2. 正文块方块环绕超边
      3. 仅属一个锚点的实体：环绕所属超边或正文块
      4. 跨锚点实体：锚点质心附近 + 斥力分离
    """
    def _cluster_radius(n_body: int, n_excl_on_he: int, n_excl_on_bodies: int) -> float:
        base = he_r + 0.55
        if n_body > 0:
            base = he_r + 1.0 + 0.15 * max(0, n_body - 4)
        node_extra = 0.0
        n_excl = n_excl_on_he + n_excl_on_bodies
        if n_excl > 0:
            rings = _ring_plan(
                max(n_excl_on_he, n_excl_on_bodies, 1),
                inner_r=chunk_half + node_spacing,
                spacing=node_spacing,
            )
            node_extra = rings[-1][0] if rings else node_spacing
        return base + node_extra + 0.35

    # exclusive / bridge 按 anchors（he 或 ch）
    exclusive_by_anchor: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    bridges: List[str] = []
    for name in entity_names:
        anchors = entity_meta[name].get("anchors") or []
        if len(anchors) <= 1:
            if anchors:
                exclusive_by_anchor[anchors[0]].append(name)
        else:
            bridges.append(name)
    for k in exclusive_by_anchor:
        exclusive_by_anchor[k] = sorted(exclusive_by_anchor[k])
    bridges = sorted(bridges, key=lambda n: (-entity_meta[n]["chunk_degree"], n))

    bubble = {}
    for hid in he_ids:
        bodies = he_to_body_chunks.get(hid, [])
        n_he_excl = len(exclusive_by_anchor.get(("he", hid), []))
        n_body_excl = sum(
            len(exclusive_by_anchor.get(("ch", cid), [])) for cid in bodies
        )
        bubble[hid] = _cluster_radius(len(bodies), n_he_excl, n_body_excl)

    # --- 超边位置 ---
    he_pos: Dict[int, Tuple[float, float]] = {}
    n_he = len(he_ids)
    if n_he == 0:
        return {}
    if n_he == 1:
        he_pos[he_ids[0]] = (0.0, 0.0)
    else:
        max_b = max(bubble.values()) if bubble else 1.0
        min_chord = 2 * max_b + node_spacing * 1.5
        R = min_chord / (2 * math.sin(math.pi / n_he))
        weights = [max(bubble[h], 0.5) for h in he_ids]
        total_w = sum(weights)
        angles = []
        acc = 0.0
        for w in weights:
            acc += w / total_w * 2 * math.pi
            angles.append(acc - (w / total_w * math.pi))
        rot = -math.pi / 2 - angles[0]
        for hid, ang in zip(he_ids, angles):
            a = ang + rot
            he_pos[hid] = (R * math.cos(a), R * math.sin(a))

        for _ in range(50):
            moved = False
            for i in range(len(he_ids)):
                for j in range(i + 1, len(he_ids)):
                    hi, hj = he_ids[i], he_ids[j]
                    xi, yi = he_pos[hi]
                    xj, yj = he_pos[hj]
                    dx, dy = xi - xj, yi - yj
                    dist = math.hypot(dx, dy)
                    need = bubble[hi] + bubble[hj] + node_spacing * 0.6
                    if dist < 1e-9:
                        dx, dy, dist = 0.1, 0.0, 0.1
                    if dist < need:
                        push = (need - dist) * 0.5
                        ux, uy = dx / dist, dy / dist
                        he_pos[hi] = (xi + ux * push, yi + uy * push)
                        he_pos[hj] = (xj - ux * push, yj - uy * push)
                        moved = True
            if not moved:
                break

    pos: Dict[str, Tuple[float, float]] = {
        f"he:{hid}": he_pos[hid] for hid in he_ids
    }

    # --- 正文块环绕超边（无头块）---
    chunk_pos: Dict[int, Tuple[float, float]] = {}
    for i, hid in enumerate(he_ids):
        bodies = he_to_body_chunks.get(hid, [])
        if not bodies:
            continue
        cx, cy = he_pos[hid]
        outward = math.atan2(cy, cx) if (cx or cy) else (-math.pi / 2)
        placed = _place_on_rings(
            bodies,
            (cx, cy),
            inner_r=he_r + chunk_half + 0.75,
            spacing=max(node_spacing * 1.5, chunk_half * 2.8),
            angle_offset=outward + i * 0.05,
        )
        for cid, xy in placed.items():
            chunk_pos[cid] = xy
            pos[f"ch:{cid}"] = xy

    # --- 单属实体环绕锚点（超边或正文块）---
    exclusive_pos: Dict[str, Tuple[float, float]] = {}
    for (kind, aid), names in exclusive_by_anchor.items():
        if not names:
            continue
        if kind == "he":
            if aid not in he_pos:
                continue
            cx, cy = he_pos[aid]
            outward = math.atan2(cy, cx) if (cx or cy) else 0.0
            inner = he_r + node_spacing * 0.7
        else:
            if aid not in chunk_pos:
                continue
            cx, cy = chunk_pos[aid]
            hid = chunk_meta.get(aid, {}).get("hyperedge_id")
            if hid is not None and hid in he_pos:
                hx, hy = he_pos[hid]
                outward = math.atan2(cy - hy, cx - hx)
            else:
                outward = 0.0
            inner = chunk_half + node_spacing * 0.85
        placed = _place_on_rings(
            names,
            (cx, cy),
            inner_r=inner,
            spacing=node_spacing,
            angle_offset=outward,
        )
        exclusive_pos.update(placed)

    for name, xy in exclusive_pos.items():
        pos[f"ent:{name}"] = xy

    # --- 跨锚点实体 ---
    def _anchor_xy(anchor: Tuple[str, int]) -> Optional[Tuple[float, float]]:
        kind, aid = anchor
        if kind == "he":
            return he_pos.get(aid)
        return chunk_pos.get(aid)

    group_members: Dict[Tuple[Tuple[str, int], ...], List[str]] = defaultdict(list)
    for name in bridges:
        key = tuple(sorted(entity_meta[name]["anchors"]))
        group_members[key].append(name)

    bridge_init: Dict[str, Tuple[float, float]] = {}
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for g_i, (akey, members) in enumerate(sorted(group_members.items(), key=lambda x: str(x[0]))):
        pts = [p for a in akey if (p := _anchor_xy(a)) is not None]
        if not pts:
            continue
        gcx = sum(p[0] for p in pts) / len(pts)
        gcy = sum(p[1] for p in pts) / len(pts)
        for j, name in enumerate(members):
            rad = node_spacing * (0.55 + 0.45 * math.sqrt(j + 0.5))
            ang = j * golden + g_i * 0.37
            bridge_init[name] = (gcx + rad * math.cos(ang), gcy + rad * math.sin(ang))

    obstacles: Dict[str, Tuple[float, float]] = {}
    for hid, xy in he_pos.items():
        obstacles[f"he:{hid}"] = xy
    for cid, xy in chunk_pos.items():
        obstacles[f"ch:{cid}"] = xy
    for name, xy in exclusive_pos.items():
        obstacles[f"ex:{name}"] = xy

    bridge_sep = _separate_positions(
        bridge_init,
        obstacles,
        min_dist=node_spacing * 0.95,
        iterations=120,
        step=node_spacing * 0.35,
    )
    for name, xy in bridge_sep.items():
        pos[f"ent:{name}"] = xy

    return pos


def draw_hypergraph(
    db_path: Path,
    output: Path,
    *,
    max_hyperedges: Optional[int] = None,
    hyperedge_ids: Optional[Iterable[int]] = None,
    min_degree: int = 1,
    max_nodes: Optional[int] = None,
    name_contains: Optional[str] = None,
    show_labels: bool = True,
    label_min_degree: int = 1,
    figsize: Tuple[float, float] = (16, 12),
    dpi: int = 160,
    title: Optional[str] = None,
    seed: int = 42,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, FancyBboxPatch
    except ImportError as e:
        raise ImportError(
            "需要安装 matplotlib：\n  pip install matplotlib"
        ) from e

    font_name = _configure_chinese_font()
    hyperedges, chunks, nodes, doc_names = load_hypergraph(db_path)
    he_id_filter = set(hyperedge_ids) if hyperedge_ids is not None else None

    (
        he_ids,
        he_meta,
        body_chunk_ids,
        chunk_meta,
        he_to_body_chunks,
        entity_names,
        entity_meta,
        membership,
    ) = build_graph(
        hyperedges,
        chunks,
        nodes,
        max_hyperedges=max_hyperedges,
        hyperedge_ids=he_id_filter,
        max_nodes=max_nodes,
        name_contains=name_contains,
        min_degree=min_degree,
    )

    if not he_ids:
        raise ValueError("没有可绘制的超边，请检查数据库是否已 build。")

    n_ent = max(len(entity_names), 1)
    node_spacing = 0.40 + min(0.12, n_ent / 600.0)
    he_r = 0.55
    chunk_half = 0.30

    pos = layout_positions(
        he_ids,
        he_to_body_chunks,
        chunk_meta,
        entity_names,
        entity_meta,
        he_r=he_r,
        chunk_half=chunk_half,
        node_spacing=node_spacing,
    )

    xs_all = [p[0] for p in pos.values()]
    ys_all = [p[1] for p in pos.values()]
    span = max(max(xs_all) - min(xs_all), max(ys_all) - min(ys_all), 1.0)
    auto_w = max(figsize[0], min(32.0, 8.0 + span * 0.5))
    auto_h = max(figsize[1], min(32.0, 8.0 + span * 0.5))
    fig, ax = plt.subplots(figsize=(auto_w, auto_h), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- 超边 → 正文块连线 + 正文块环 ---
    for hid in he_ids:
        cids = he_to_body_chunks.get(hid, [])
        he_xy = pos.get(f"he:{hid}")
        if he_xy is None:
            continue
        pts = []
        for cid in cids:
            p = pos.get(f"ch:{cid}")
            if p is not None:
                pts.append(p)
        for p in pts:
            ax.plot(
                [he_xy[0], p[0]],
                [he_xy[1], p[1]],
                color="#93c5fd",
                linewidth=1.4,
                alpha=0.55,
                zorder=1,
                solid_capstyle="round",
            )
        if len(pts) >= 2:
            ring = pts + [pts[0]]
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            ax.plot(
                xs,
                ys,
                color="#3b82f6",
                linewidth=1.8,
                alpha=0.45,
                zorder=1,
                linestyle="--",
            )

    # --- 实体 → 锚点（超边 或 正文块）---
    for name, target in membership:
        p1 = pos.get(f"ent:{name}")
        if p1 is None:
            continue
        kind, aid = target
        p2 = pos.get(f"he:{aid}" if kind == "he" else f"ch:{aid}")
        if p2 is None:
            continue
        deg = entity_meta[name]["chunk_degree"]
        alpha = 0.25 if deg <= 1 else min(0.55, 0.25 + 0.08 * deg)
        lw = 0.5 if deg <= 1 else min(1.5, 0.5 + 0.12 * deg)
        color = "#93c5fd" if kind == "he" else ("#cbd5e1" if deg <= 1 else "#64748b")
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            linewidth=lw,
            alpha=alpha,
            zorder=2,
            solid_capstyle="round",
        )

    # --- 超边（= 头块）：大圆 ---
    for hid in he_ids:
        x, y = pos[f"he:{hid}"]
        circle = Circle(
            (x, y),
            he_r,
            facecolor="#dbeafe",
            edgecolor="#1d4ed8",
            linewidth=2.4,
            alpha=0.96,
            zorder=3,
        )
        ax.add_patch(circle)
        if show_labels:
            ax.text(
                x,
                y,
                _he_label(hid, he_meta, doc_names),
                ha="center",
                va="center",
                fontsize=7.0,
                color="#1e3a8a",
                zorder=4,
                fontweight="bold",
            )

    # --- 正文块：中等方块（不含头块）---
    for cid in body_chunk_ids:
        key = f"ch:{cid}"
        if key not in pos:
            continue
        x, y = pos[key]
        face, edge = "#e2e8f0", "#475569"
        half = chunk_half
        box = FancyBboxPatch(
            (x - half, y - half),
            half * 2,
            half * 2,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.6,
            alpha=0.96,
            zorder=5,
        )
        ax.add_patch(box)
        if show_labels:
            ax.text(
                x,
                y,
                _chunk_label(cid, chunk_meta),
                ha="center",
                va="center",
                fontsize=5.5,
                color="#0f172a",
                zorder=6,
            )

    # --- 实体：小圆（同名已合并）---
    for name in entity_names:
        key = f"ent:{name}"
        if key not in pos:
            continue
        x, y = pos[key]
        deg = entity_meta[name]["chunk_degree"]
        r = 0.09 + 0.012 * min(max(deg, 1), 8)
        if deg <= 1:
            face, edge = "#fde68a", "#d97706"
        elif deg == 2:
            face, edge = "#fdba74", "#c2410c"
        else:
            face, edge = "#fca5a5", "#b91c1c"
        circle = Circle(
            (x, y),
            r,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            alpha=0.95,
            zorder=7,
        )
        ax.add_patch(circle)
        if show_labels and deg >= label_min_degree:
            ax.text(
                x,
                y + r + node_spacing * 0.15,
                _clip(name, 12),
                ha="center",
                va="bottom",
                fontsize=5.5 if deg <= 1 else 6.5,
                color="#334155",
                zorder=8,
                clip_on=False,
            )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = max(1.2, node_spacing * 2.5)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    n_raw_nodes = len(nodes)
    n_merged = len(entity_names)
    n_he = len(he_ids)
    n_ch = len(body_chunk_ids)
    n_links = len(membership)
    if title is None:
        title = (
            f"Hypergraph  |  超边/头块={n_he}  正文块={n_ch}  "
            f"nodes(raw={n_raw_nodes} → merged={n_merged})  links={n_links}"
        )
    ax.set_title(title, fontsize=12, pad=12)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#dbeafe",
               markeredgecolor="#1d4ed8", markersize=16, label="超边 (= 头块)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e2e8f0",
               markeredgecolor="#475569", markersize=11, label="正文块 (body chunk)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#fde68a",
               markeredgecolor="#d97706", markersize=8, label="实体节点 (单锚点)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#fca5a5",
               markeredgecolor="#b91c1c", markersize=10, label="跨块/跨超边同名实体"),
        Line2D([0], [0], color="#3b82f6", linewidth=1.8, linestyle="--",
               label="超边连接正文块环"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9, fontsize=8)

    subtitle = f"db={db_path}"
    if font_name is None:
        subtitle += (
            "  |  未检测到中文字体（可将 NotoSansSC-Regular.otf 放到 assets/fonts/）"
        )
    else:
        subtitle += f"  |  font={font_name}"
    fig.text(0.5, 0.02, subtitle, ha="center", fontsize=8, color="#64748b")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="绘制三层超图：超边大圆 → 块方块 → 同名合并实体圆"
    )
    p.add_argument("--db", type=str, required=True, help="main.db 路径")
    p.add_argument("--output", "-o", type=str, default="outputs/hypergraph.png")
    p.add_argument("--max-hyperedges", type=int, default=None)
    p.add_argument("--hyperedge-ids", type=str, default=None, help="逗号分隔超边 id")
    p.add_argument(
        "--min-degree",
        type=int,
        default=1,
        help="实体至少连接多少条超边才显示",
    )
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--name-contains", type=str, default=None)
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--label-min-degree", type=int, default=1)
    p.add_argument("--figsize", type=str, default="16,12")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--title", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    he_ids = None
    if args.hyperedge_ids:
        he_ids = [int(x.strip()) for x in args.hyperedge_ids.split(",") if x.strip()]

    fw, fh = (float(x) for x in args.figsize.split(","))
    out = draw_hypergraph(
        db_path=Path(args.db),
        output=Path(args.output),
        max_hyperedges=args.max_hyperedges,
        hyperedge_ids=he_ids,
        min_degree=args.min_degree,
        max_nodes=args.max_nodes,
        name_contains=args.name_contains,
        show_labels=not args.no_labels,
        label_min_degree=args.label_min_degree,
        figsize=(fw, fh),
        dpi=args.dpi,
        title=args.title,
        seed=args.seed,
    )
    print(f"Saved hypergraph figure → {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
