#!/usr/bin/env python3
"""
根据 hyperedge 表的 id 与 recommendation 列绘制「相似产品 / 推荐」关联图。

约定：
  - 每个超边条目 = 一个节点（默认只画有推荐关系的超边；可用 --all-nodes 画全库）
  - recommendation 字段为逗号分隔的目标超边 id（如 "3,7,13"）
  - 边为无向（A→B 与 B→A 只画一次），指向不存在的 id 会跳过并告警

示例：
  python scripts/draw_recommendation_graph.py --db example/b/DB/main.db
  python scripts/draw_recommendation_graph.py --db example/b/DB/main.db -o outputs/recommendation.png
  python scripts/draw_recommendation_graph.py --db example/b/DB/main.db --all-nodes --label id
"""

from __future__ import annotations

import argparse
import math
import re
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

    # 字体目录：scripts/assets/fonts（随 scripts 一起维护）
    scripts_dir = Path(__file__).resolve().parent
    local_font_dirs = [
        scripts_dir / "assets" / "fonts",
        scripts_dir / "fonts",
        _project_root() / "assets" / "fonts",  # 兼容旧位置
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


def parse_recommendation_ids(raw) -> List[int]:
    """Parse '1,2,3' / list → int ids（去空、去重、保序）。"""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        parts = list(raw)
    else:
        text = str(raw).strip()
        if not text:
            return []
        parts = re.split(r"[,，;\s]+", text)
    out: List[int] = []
    seen: Set[int] = set()
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if not s:
            continue
        try:
            i = int(s)
        except Exception:
            continue
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def load_hyperedges(db_path: Path) -> Tuple[List[dict], Dict[int, str]]:
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "hyperedge" not in tables:
        con.close()
        raise ValueError(f"数据库缺少表 `hyperedge`: {db_path}")

    cols = {r[1] for r in cur.execute("PRAGMA table_info(hyperedge)")}
    if "recommendation" not in cols:
        con.close()
        raise ValueError("hyperedge 表无 recommendation 列，请先跑 dhmf.recommend()")

    rows = [
        dict(r)
        for r in cur.execute(
            "SELECT id, doc_id, name, content, recommendation FROM hyperedge ORDER BY id"
        )
    ]

    doc_names: Dict[int, str] = {}
    if "doc" in tables:
        for r in cur.execute("SELECT id, name FROM doc"):
            if r["id"] is not None:
                doc_names[int(r["id"])] = r["name"] or f"doc_{r['id']}"

    con.close()
    return rows, doc_names


def build_recommendation_graph(
    hyperedges: List[dict],
    *,
    all_nodes: bool = False,
    hyperedge_ids: Optional[Set[int]] = None,
    min_degree: int = 0,
) -> Tuple[List[int], List[Tuple[int, int]], Dict[int, dict], List[str]]:
    """
    返回：
      node_ids, undirected_edges, meta_by_id, warnings
    """
    he_by_id: Dict[int, dict] = {}
    for h in hyperedges:
        if h.get("id") is None:
            continue
        he_by_id[int(h["id"])] = h

    if hyperedge_ids is not None:
        he_by_id = {hid: he_by_id[hid] for hid in hyperedge_ids if hid in he_by_id}

    all_ids = set(he_by_id.keys())
    directed: List[Tuple[int, int]] = []
    warnings: List[str] = []
    missing_targets: Dict[int, List[int]] = defaultdict(list)

    for hid, h in he_by_id.items():
        for rid in parse_recommendation_ids(h.get("recommendation")):
            if rid not in all_ids:
                # 若过滤了 hyperedge_ids，目标可能只是被过滤掉；全库里也可能真的不存在
                missing_targets[hid].append(rid)
                continue
            if rid == hid:
                continue
            directed.append((hid, rid))

    for src, bad_ids in sorted(missing_targets.items()):
        uniq = sorted(set(bad_ids))
        if len(uniq) <= 5:
            detail = ",".join(str(x) for x in uniq)
        else:
            detail = ",".join(str(x) for x in uniq[:5]) + f",…(+{len(uniq) - 5})"
        warnings.append(f"he={src} recommendation 指向不存在/未选中的 id: {detail}")

    # 无向去重
    edge_set: Set[Tuple[int, int]] = set()
    for a, b in directed:
        edge_set.add((a, b) if a < b else (b, a))
    edges = sorted(edge_set)

    degree: Dict[int, int] = defaultdict(int)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1

    if all_nodes:
        node_ids = sorted(all_ids)
    else:
        # 只保留出现在至少一条推荐边上的节点
        node_ids = sorted({n for e in edges for n in e})

    if min_degree > 0:
        keep = {n for n in node_ids if degree.get(n, 0) >= min_degree}
        # 边两端都保留才画
        edges = [(a, b) for a, b in edges if a in keep and b in keep]
        if all_nodes:
            node_ids = sorted(keep | (all_ids & keep))
            # all_nodes + min_degree: 仍只保留满足度的点
            node_ids = sorted(keep)
        else:
            node_ids = sorted({n for e in edges for n in e})

    meta = {hid: he_by_id[hid] for hid in node_ids if hid in he_by_id}
    return node_ids, edges, meta, warnings


def _clip(s: str, max_len: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _product_label(
    hid: int,
    meta: dict,
    doc_names: Dict[int, str],
    *,
    mode: str = "auto",
    max_len: int = 18,
) -> str:
    """
    mode:
      id     → 只显示 he id
      name   → name / 文档名
      auto   → 优先文档名，否则 name，否则 HE-id
      both   → 文档名 + id
    """
    he = meta.get(hid, {})
    doc_id = he.get("doc_id")
    doc_name = ""
    if doc_id is not None and int(doc_id) in doc_names:
        doc_name = doc_names[int(doc_id)]
        if doc_name.lower().endswith(".pdf"):
            doc_name = doc_name[:-4]

    raw_name = (he.get("name") or "").strip()
    if raw_name.lower() in ("", "head"):
        raw_name = ""

    content_hint = ""
    content = (he.get("content") or "").strip()
    if content:
        # 尝试从头块文本截一段产品名感
        m = re.search(r"(?:产品名称|Product(?:\s+name)?|Name)\s*[:：]\s*([^\n;；]{2,60})", content, re.I)
        if m:
            content_hint = m.group(1).strip()

    if mode == "id":
        return str(hid)
    if mode == "name":
        base = doc_name or raw_name or content_hint or f"HE-{hid}"
        return _clip(base, max_len)
    if mode == "both":
        base = doc_name or raw_name or content_hint or f"HE-{hid}"
        return f"{_clip(base, max(4, max_len - 6))}\n({hid})"
    # auto
    base = doc_name or raw_name or content_hint or f"HE-{hid}"
    return f"{_clip(base, max_len)}\n(he={hid})"


def _min_nearest_neighbor(coords: List[Tuple[float, float]]) -> Optional[float]:
    min_nn = None
    for i in range(len(coords)):
        xi, yi = coords[i]
        for j in range(i + 1, len(coords)):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < 1e-9:
                continue
            if min_nn is None or d < min_nn:
                min_nn = d
    return min_nn


def _component_local_layout(
    comp: List[int],
    edge_list: List[Tuple[int, int]],
    *,
    seed: int,
    iterations: int,
    scale: float,
) -> Dict[int, Tuple[float, float]]:
    """
    单个连通分量内部布局，坐标以 (0,0) 为质心。
    scale 控制节点间距：越大越松。
    """
    n = len(comp)
    # 目标最近邻间距：给节点圆 + 外侧标签留空
    target_nn = 1.35 * scale

    if n == 1:
        return {comp[0]: (0.0, 0.0)}
    if n == 2:
        d = target_nn * 0.55
        return {comp[0]: (-d, 0.0), comp[1]: (d, 0.0)}
    if n == 3:
        # 等边三角
        r = target_nn / math.sqrt(3)
        out = {}
        for i, nid in enumerate(comp):
            ang = -math.pi / 2 + 2 * math.pi * i / 3
            out[nid] = (r * math.cos(ang), r * math.sin(ang))
        return out
    if n == 4:
        # 优先方形，比 spring 更稳
        half = target_nn * 0.55
        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        return {nid: corners[i] for i, nid in enumerate(comp)}

    try:
        import networkx as nx
    except ImportError:
        nx = None

    if nx is not None:
        sub = nx.Graph()
        sub.add_nodes_from(comp)
        for a, b in edge_list:
            if a in sub and b in sub:
                sub.add_edge(a, b)
        k = max(2.2, 3.4 / math.sqrt(n)) * (scale / 2.2)
        sub_pos = nx.spring_layout(
            sub,
            seed=seed,
            k=k,
            iterations=max(iterations, 160),
            weight=None,
            scale=1.0,
        )
        min_nn = _min_nearest_neighbor(list(sub_pos.values()))
        if min_nn and min_nn > 0:
            factor = target_nn / min_nn
            factor = max(1.15, min(factor, 6.0))
            sub_pos = {nid: (x * factor, y * factor) for nid, (x, y) in sub_pos.items()}

        # 节点间再做一轮硬性分离，防止 spring 局部重叠
        ids = list(sub_pos.keys())
        for _ in range(60):
            deltas = {nid: [0.0, 0.0] for nid in ids}
            moved = False
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    xa, ya = sub_pos[a]
                    xb, yb = sub_pos[b]
                    dx, dy = xa - xb, ya - yb
                    dist = math.hypot(dx, dy)
                    need = target_nn * 0.92
                    if dist < 1e-9:
                        dx, dy, dist = 0.1, 0.05, 0.11
                    if dist < need:
                        push = (need - dist) * 0.5
                        ux, uy = dx / dist, dy / dist
                        deltas[a][0] += ux * push
                        deltas[a][1] += uy * push
                        deltas[b][0] -= ux * push
                        deltas[b][1] -= uy * push
                        moved = True
            if not moved:
                break
            for nid in ids:
                sub_pos[nid] = (
                    sub_pos[nid][0] + deltas[nid][0],
                    sub_pos[nid][1] + deltas[nid][1],
                )
    else:
        r = max(target_nn, 0.5 * n * target_nn / (2 * math.pi))
        sub_pos = {}
        for i, nid in enumerate(comp):
            ang = -math.pi / 2 + 2 * math.pi * i / n
            sub_pos[nid] = (r * math.cos(ang), r * math.sin(ang))

    cx = sum(p[0] for p in sub_pos.values()) / n
    cy = sum(p[1] for p in sub_pos.values()) / n
    return {nid: (x - cx, y - cy) for nid, (x, y) in sub_pos.items()}


def _component_radius(local_pos: Dict[int, Tuple[float, float]], pad: float = 0.35) -> float:
    if not local_pos:
        return pad
    r = max(math.hypot(x, y) for x, y in local_pos.values())
    return r + pad


def _pack_components_radial(
    components: List[List[int]],
    local_layouts: Dict[int, Dict[int, Tuple[float, float]]],
    radii: Dict[int, float],
    *,
    gap: float = 1.35,
) -> Dict[int, Tuple[float, float]]:
    """
    将多个连通分量环绕图中心排列（非一字排开）。
    - 最大分量尽量放中心
    - 其余按大小在外圈环绕；圈放不下则扩到下一环
    """
    if not components:
        return {}

    # 索引用 list 位置
    order = list(range(len(components)))
    order.sort(key=lambda i: (-len(components[i]), -radii[i], components[i][0]))

    pos: Dict[int, Tuple[float, float]] = {}
    center_idx = order[0]
    # 中心分量
    for nid, (x, y) in local_layouts[center_idx].items():
        pos[nid] = (x, y)

    remaining = order[1:]
    if not remaining:
        return pos

    # 按环放置：每环容纳的分量数量随半径增加
    ring = 0
    idx = 0
    center_r = radii[center_idx]
    while idx < len(remaining):
        ring += 1
        # 本环半径：在中心分量外包络外再叠一层
        # 先估计本环要放多少：随环号递增
        approx_count = min(len(remaining) - idx, max(6, 6 + 3 * (ring - 1)))
        # 取本批中最大半径，决定环间距
        batch_ids = remaining[idx: idx + approx_count]
        max_comp_r = max(radii[i] for i in batch_ids)
        # 环中心到原点的距离
        if ring == 1:
            ring_R = center_r + max_comp_r + gap
        else:
            # 用上一环估算：上一环半径 + 间隙
            # 简化：随环累加
            ring_R = center_r + gap
            for r_i in range(1, ring):
                # 粗略累加：每环至少 2*max 分量半径级间距
                ring_R += max_comp_r * 1.15 + gap

        # 根据圆周长度决定本环实际可放数量，避免分量互相撞
        circ = 2 * math.pi * max(ring_R, 1e-6)
        slot = 2 * max_comp_r + gap * 0.85
        capacity = max(1, int(circ / max(slot, 1e-6)))
        batch_ids = remaining[idx: idx + capacity]
        m = len(batch_ids)
        if m == 0:
            break

        # 再按本批真实最大半径微调 ring_R
        max_comp_r = max(radii[i] for i in batch_ids)
        if ring == 1:
            ring_R = center_r + max_comp_r + gap
        else:
            # 用已放置点的最远距离 + 新分量半径
            if pos:
                placed_r = max(math.hypot(x, y) for x, y in pos.values())
            else:
                placed_r = center_r
            # 还要考虑已放置外圈分量的外包络
            # 用 radii 近似：已放分量中心距 + 其半径
            outer = center_r
            for j in order:
                if j == center_idx:
                    continue
                # 若该分量已有点
                sample = next(iter(local_layouts[j]))
                if sample in pos:
                    # 分量中心 ≈ 其节点均值
                    pts = [pos[nid] for nid in components[j] if nid in pos]
                    if pts:
                        cx = sum(p[0] for p in pts) / len(pts)
                        cy = sum(p[1] for p in pts) / len(pts)
                        outer = max(outer, math.hypot(cx, cy) + radii[j])
            ring_R = outer + max_comp_r + gap

        # 角度：从正上方开始，均匀分布；大分量优先（batch 已按 order）
        # 按半径加权分角度，大块多分一点弧
        weights = [max(radii[i], 0.3) for i in batch_ids]
        total_w = sum(weights)
        angles = []
        acc = 0.0
        for w in weights:
            acc += w / total_w * 2 * math.pi
            angles.append(acc - (w / total_w * math.pi))
        rot = -math.pi / 2 - angles[0]

        for bi, ang0 in zip(batch_ids, angles):
            ang = ang0 + rot
            ox = ring_R * math.cos(ang)
            oy = ring_R * math.sin(ang)
            for nid, (x, y) in local_layouts[bi].items():
                pos[nid] = (x + ox, y + oy)

        idx += m

    # 分量中心之间若仍过近，再做几次排斥推开
    centers: Dict[int, Tuple[float, float]] = {}
    for i, comp in enumerate(components):
        pts = [pos[nid] for nid in comp if nid in pos]
        if not pts:
            continue
        centers[i] = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
        )

    keys = list(centers.keys())
    for _ in range(40):
        deltas = {i: [0.0, 0.0] for i in keys}
        moved = False
        for a_i in range(len(keys)):
            for b_i in range(a_i + 1, len(keys)):
                ia, ib = keys[a_i], keys[b_i]
                xa, ya = centers[ia]
                xb, yb = centers[ib]
                dx, dy = xa - xb, ya - yb
                dist = math.hypot(dx, dy)
                need = radii[ia] + radii[ib] + gap * 0.9
                if dist < 1e-9:
                    dx, dy, dist = 0.15, 0.05, 0.16
                if dist < need:
                    push = (need - dist) * 0.5
                    ux, uy = dx / dist, dy / dist
                    deltas[ia][0] += ux * push
                    deltas[ia][1] += uy * push
                    deltas[ib][0] -= ux * push
                    deltas[ib][1] -= uy * push
                    moved = True
        if not moved:
            break
        for i in keys:
            dx, dy = deltas[i]
            mag = math.hypot(dx, dy)
            if mag > 1.2:
                dx, dy = dx / mag * 1.2, dy / mag * 1.2
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            centers[i] = (centers[i][0] + dx, centers[i][1] + dy)
            for nid in components[i]:
                if nid in pos:
                    pos[nid] = (pos[nid][0] + dx, pos[nid][1] + dy)

    return pos


def _layout_positions(
    node_ids: List[int],
    edges: List[Tuple[int, int]],
    *,
    seed: int = 42,
    iterations: int = 140,
    scale: float = 2.6,
    component_gap: float = 1.5,
) -> Dict[int, Tuple[float, float]]:
    """
    连通分量环绕中心排列；分量内节点用更大的 spring 间距。
    """
    if not node_ids:
        return {}

    try:
        import networkx as nx
    except ImportError:
        nx = None

    if nx is None:
        n = len(node_ids)
        if n == 1:
            return {node_ids[0]: (0.0, 0.0)}
        R = max(2.0, n * 0.55) * (scale / 2.0)
        out = {}
        for i, nid in enumerate(node_ids):
            ang = -math.pi / 2 + 2 * math.pi * i / n
            out[nid] = (R * math.cos(ang), R * math.sin(ang))
        return out

    G = nx.Graph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(edges)
    components = [sorted(c) for c in nx.connected_components(G)]
    components.sort(key=lambda c: (-len(c), c[0] if c else 0))

    local_layouts: Dict[int, Dict[int, Tuple[float, float]]] = {}
    radii: Dict[int, float] = {}
    for i, comp in enumerate(components):
        local = _component_local_layout(
            comp,
            edges,
            seed=seed + i * 17,
            iterations=iterations,
            scale=scale,
        )
        local_layouts[i] = local
        # 半径预留节点半径 + 标签外伸空间
        radii[i] = _component_radius(local, pad=0.55 * scale)

    return _pack_components_radial(
        components,
        local_layouts,
        radii,
        gap=component_gap * scale * 0.55,
    )


def _node_radius(deg: int, node_size_scale: float) -> float:
    return (0.22 + 0.045 * min(max(deg, 0), 8)) * node_size_scale


def _estimate_label_size(text: str, fontsize: float) -> Tuple[float, float]:
    """粗估标签宽高（数据坐标近似，后续会按画布比例再调）。"""
    lines = (text or "").split("\n")
    max_chars = max((len(ln) for ln in lines), default=1)
    # 经验：字号 7 ≈ 0.09 数据单位/字（后续会按 span 再缩放）
    char_w = 0.075 * (fontsize / 7.0)
    line_h = 0.16 * (fontsize / 7.0)
    return max_chars * char_w, max(1, len(lines)) * line_h


def _place_labels(
    node_ids: List[int],
    pos: Dict[int, Tuple[float, float]],
    labels: Dict[int, str],
    radii: Dict[int, float],
    *,
    fontsizes: Dict[int, float],
    iterations: int = 140,
) -> Dict[int, Tuple[float, float]]:
    """
    标签放在节点外侧：优先沿「远离邻居密集方向」外推，
    再通过标签-节点 / 标签-标签排斥减少重叠。
    """
    if not node_ids:
        return {}

    gx = sum(pos[n][0] for n in node_ids) / len(node_ids)
    gy = sum(pos[n][1] for n in node_ids) / len(node_ids)

    # 每个节点：邻居斥力方向 + 全局外侧方向 合成初始标签方位
    label_pos: Dict[int, Tuple[float, float]] = {}
    half_sizes: Dict[int, Tuple[float, float]] = {}
    preferred_dir: Dict[int, Tuple[float, float]] = {}

    for nid in node_ids:
        x, y = pos[nid]
        r = radii[nid]
        text = labels.get(nid, "")
        fs = fontsizes.get(nid, 7.0)
        tw, th = _estimate_label_size(text, fs)
        # 标签框略放大，给排斥留余量
        half_sizes[nid] = (tw / 2 * 1.08, th / 2 * 1.15)

        # 远离近邻的方向（避免标签挤进簇中心）
        rx, ry = 0.0, 0.0
        for mid in node_ids:
            if mid == nid:
                continue
            mx, my = pos[mid]
            dx, dy = x - mx, y - my
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                continue
            # 近邻权重大
            w = 1.0 / (dist * dist + 0.15)
            rx += dx / dist * w
            ry += dy / dist * w
        # 再混一点远离全局质心
        dxg, dyg = x - gx, y - gy
        dg = math.hypot(dxg, dyg)
        if dg > 1e-9:
            rx += 0.35 * dxg / dg
            ry += 0.35 * dyg / dg
        rm = math.hypot(rx, ry)
        if rm < 1e-9:
            ang = (nid * 47) % 360 * math.pi / 180.0
            ux, uy = math.cos(ang), math.sin(ang)
        else:
            ux, uy = rx / rm, ry / rm
        preferred_dir[nid] = (ux, uy)
        offset = r + th / 2 + 0.28
        label_pos[nid] = (x + ux * offset, y + uy * offset)

    def _overlap_push(
        ax: float, ay: float, aw: float, ah: float,
        bx: float, by: float, bw: float, bh: float,
        pad: float = 0.1,
    ) -> Optional[Tuple[float, float]]:
        dx = ax - bx
        dy = ay - by
        ox = (aw + bw) / 2 + pad - abs(dx)
        oy = (ah + bh) / 2 + pad - abs(dy)
        if ox <= 0 or oy <= 0:
            return None
        if ox < oy:
            return (math.copysign(ox, dx if abs(dx) > 1e-9 else 1.0), 0.0)
        return (0.0, math.copysign(oy, dy if abs(dy) > 1e-9 else 1.0))

    for it in range(iterations):
        deltas = {nid: [0.0, 0.0] for nid in node_ids}
        max_move = 0.0

        # 标签 ↔ 所有节点圆
        for nid in node_ids:
            lx, ly = label_pos[nid]
            tw, th = half_sizes[nid]
            lr = max(tw, th) * 0.85
            for mid in node_ids:
                nx_, ny_ = pos[mid]
                nr = radii[mid] + 0.14
                dx, dy = lx - nx_, ly - ny_
                dist = math.hypot(dx, dy)
                need = nr + lr
                if dist < 1e-9:
                    ux, uy = preferred_dir[nid]
                    dx, dy, dist = ux, uy, 1.0
                if dist < need:
                    push = (need - dist) / dist
                    force = 1.15 if mid == nid else 0.95
                    deltas[nid][0] += dx * push * force
                    deltas[nid][1] += dy * push * force

        # 标签 ↔ 标签（AABB）
        ids = node_ids
        for i in range(len(ids)):
            a = ids[i]
            ax, ay = label_pos[a]
            aw, ah = half_sizes[a][0] * 2, half_sizes[a][1] * 2
            for j in range(i + 1, len(ids)):
                b = ids[j]
                bx, by = label_pos[b]
                bw, bh = half_sizes[b][0] * 2, half_sizes[b][1] * 2
                push = _overlap_push(ax, ay, aw, ah, bx, by, bw, bh, pad=0.12)
                if push is None:
                    continue
                px, py = push
                deltas[a][0] += px * 0.55
                deltas[a][1] += py * 0.55
                deltas[b][0] -= px * 0.55
                deltas[b][1] -= py * 0.55

        # 轻微拉回：标签不要离所属节点太远
        for nid in node_ids:
            x, y = pos[nid]
            lx, ly = label_pos[nid]
            dx, dy = lx - x, ly - y
            dist = math.hypot(dx, dy)
            max_dist = radii[nid] + max(half_sizes[nid]) * 2.8 + 1.1
            if dist > max_dist and dist > 1e-9:
                pull = (dist - max_dist) * 0.25
                deltas[nid][0] -= dx / dist * pull
                deltas[nid][1] -= dy / dist * pull

        step = 0.28 if it < iterations // 2 else 0.16
        for nid in node_ids:
            dx, dy = deltas[nid]
            mag = math.hypot(dx, dy)
            if mag > step:
                dx, dy = dx / mag * step, dy / mag * step
                mag = step
            if mag < 1e-4:
                continue
            label_pos[nid] = (label_pos[nid][0] + dx, label_pos[nid][1] + dy)
            max_move = max(max_move, mag)
        if max_move < 8e-4:
            break

    return label_pos


def draw_recommendation_graph(
    db_path: Path,
    output: Path,
    *,
    all_nodes: bool = False,
    hyperedge_ids: Optional[Iterable[int]] = None,
    min_degree: int = 0,
    label: str = "auto",
    label_max_len: int = 18,
    show_labels: bool = True,
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 160,
    title: Optional[str] = None,
    seed: int = 42,
    node_size_scale: float = 1.0,
    layout_scale: float = 3.2,
    component_gap: float = 1.9,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, FancyBboxPatch
    except ImportError as e:
        raise ImportError("需要安装 matplotlib：\n  pip install matplotlib") from e

    font_name = _configure_chinese_font()
    hyperedges, doc_names = load_hyperedges(db_path)
    he_filter = set(hyperedge_ids) if hyperedge_ids is not None else None

    node_ids, edges, meta, warnings = build_recommendation_graph(
        hyperedges,
        all_nodes=all_nodes,
        hyperedge_ids=he_filter,
        min_degree=min_degree,
    )

    for w in warnings[:20]:
        print(f"[warn] {w}", file=sys.stderr)
    if len(warnings) > 20:
        print(f"[warn] …另有 {len(warnings) - 20} 条缺失目标警告", file=sys.stderr)

    if not node_ids:
        n_he = len(hyperedges)
        n_with = sum(1 for h in hyperedges if parse_recommendation_ids(h.get("recommendation")))
        raise ValueError(
            "没有可绘制的推荐关系节点。\n"
            f"  hyperedge 总数={n_he}，recommendation 非空={n_with}。\n"
            "  请先运行 dhmf.recommend() 写入推荐，或使用 --all-nodes 查看孤立点。"
        )

    degree: Dict[int, int] = defaultdict(int)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1

    pos = _layout_positions(
        node_ids,
        edges,
        seed=seed,
        scale=layout_scale,
        component_gap=component_gap,
    )
    if not pos:
        raise ValueError("布局失败：无节点坐标")

    node_radii = {
        hid: _node_radius(degree.get(hid, 0), node_size_scale) for hid in node_ids
    }

    label_texts: Dict[int, str] = {}
    fontsizes: Dict[int, float] = {}
    if show_labels:
        for hid in node_ids:
            deg = degree.get(hid, 0)
            label_texts[hid] = _product_label(
                hid, meta, doc_names, mode=label, max_len=label_max_len
            )
            fontsizes[hid] = 6.0 if deg <= 2 else 6.5

        label_pos = _place_labels(
            node_ids,
            pos,
            label_texts,
            node_radii,
            fontsizes=fontsizes,
        )
    else:
        label_pos = {}

    # 画布范围：节点 + 标签
    xs_all = [pos[n][0] for n in node_ids]
    ys_all = [pos[n][1] for n in node_ids]
    for nid, (lx, ly) in label_pos.items():
        tw, th = _estimate_label_size(label_texts[nid], fontsizes[nid])
        xs_all.extend([lx - tw / 2, lx + tw / 2])
        ys_all.extend([ly - th / 2, ly + th / 2])
    span = max(max(xs_all) - min(xs_all), max(ys_all) - min(ys_all), 1.0)
    auto_w = max(figsize[0], min(40.0, 10.0 + span * 0.95 + len(node_ids) * 0.06))
    auto_h = max(figsize[1], min(40.0, 10.0 + span * 0.95 + len(node_ids) * 0.05))
    fig, ax = plt.subplots(figsize=(auto_w, auto_h), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- edges（画到圆边界，避免压住圆心）---
    for a, b in edges:
        if a not in pos or b not in pos:
            continue
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            continue
        ux, uy = dx / dist, dy / dist
        ra, rb = node_radii[a], node_radii[b]
        ax.plot(
            [x1 + ux * ra, x2 - ux * rb],
            [y1 + uy * ra, y2 - uy * rb],
            color="#94a3b8",
            linewidth=1.15,
            alpha=0.6,
            zorder=1,
            solid_capstyle="round",
        )

    # --- soft leader lines: node → label ---
    if show_labels:
        for hid in node_ids:
            if hid not in label_pos:
                continue
            x, y = pos[hid]
            lx, ly = label_pos[hid]
            dx, dy = lx - x, ly - y
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                continue
            ux, uy = dx / dist, dy / dist
            r = node_radii[hid]
            ax.plot(
                [x + ux * r, lx],
                [y + uy * r, ly],
                color="#cbd5e1",
                linewidth=0.7,
                alpha=0.55,
                zorder=2,
                linestyle=":",
            )

    # --- nodes ---
    max_deg = max((degree.get(n, 0) for n in node_ids), default=0)
    for hid in node_ids:
        x, y = pos[hid]
        deg = degree.get(hid, 0)
        r = node_radii[hid]
        if deg == 0:
            face, edge_c = "#f1f5f9", "#94a3b8"
        elif deg == 1:
            face, edge_c = "#dbeafe", "#3b82f6"
        elif deg <= 3:
            face, edge_c = "#bfdbfe", "#2563eb"
        else:
            face, edge_c = "#93c5fd", "#1d4ed8"
        circle = Circle(
            (x, y),
            r,
            facecolor=face,
            edgecolor=edge_c,
            linewidth=1.8,
            alpha=0.96,
            zorder=3,
        )
        ax.add_patch(circle)
        # 圆内只放短 id，避免与外侧产品名抢位
        ax.text(
            x,
            y,
            str(hid),
            ha="center",
            va="center",
            fontsize=7.0,
            color="#1e3a8a",
            zorder=4,
        )

    # --- outer labels (name) ---
    if show_labels:
        for hid in node_ids:
            if hid not in label_pos:
                continue
            lx, ly = label_pos[hid]
            text = label_texts[hid]
            # auto/both 若含换行且末行是 (he=xx)，圆内已有 id 时可只保留名称行
            if label in ("auto", "both") and "\n" in text:
                lines = text.split("\n")
                # 去掉纯 (he=N) / (N)
                lines = [
                    ln for ln in lines
                    if not re.fullmatch(r"\(?(?:he=)?\d+\)?", ln.strip())
                ]
                text = "\n".join(lines) if lines else label_texts[hid]

            fs = fontsizes[hid]
            tw, th = _estimate_label_size(text, fs)
            # 半透明底，减少与边线交叉时的阅读干扰
            box = FancyBboxPatch(
                (lx - tw / 2 - 0.04, ly - th / 2 - 0.03),
                tw + 0.08,
                th + 0.06,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                facecolor="white",
                edgecolor="#e2e8f0",
                linewidth=0.6,
                alpha=0.88,
                zorder=5,
            )
            ax.add_patch(box)
            ax.text(
                lx,
                ly,
                text,
                ha="center",
                va="center",
                fontsize=fs,
                color="#334155",
                zorder=6,
            )

    xs = list(xs_all)
    ys = list(ys_all)
    pad = max(1.0, 0.55 * layout_scale)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    n_nodes = len(node_ids)
    n_edges = len(edges)
    n_iso = sum(1 for n in node_ids if degree.get(n, 0) == 0)
    if title is None:
        title = (
            f"Recommendation Graph  |  nodes={n_nodes}  edges={n_edges}  "
            f"isolated={n_iso}  max_degree={max_deg}"
        )
    ax.set_title(title, fontsize=12, pad=14)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="#dbeafe", markeredgecolor="#3b82f6",
            markersize=10, label="超边（圆内=id）",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="#93c5fd", markeredgecolor="#1d4ed8",
            markersize=14, label="高连接度超边",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="#f1f5f9", markeredgecolor="#94a3b8",
            markersize=9, label="孤立点（无 recommendation）",
        ),
        Line2D([0], [0], color="#94a3b8", linewidth=1.5, label="推荐关系（无向）"),
        Line2D([0], [0], color="#cbd5e1", linewidth=1.0, linestyle=":", label="名称标注引线"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.92, fontsize=8)

    subtitle = f"db={db_path}  |  source=hyperedge.id ↔ hyperedge.recommendation"
    if font_name is None:
        subtitle += "  |  未检测到中文字体（可将字体放到 scripts/assets/fonts/）"
    else:
        subtitle += f"  |  font={font_name}"
    fig.text(0.5, 0.015, subtitle, ha="center", fontsize=8, color="#64748b")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        f"[ok] wrote {output}  nodes={n_nodes} edges={n_edges} isolated={n_iso}",
        file=sys.stderr,
    )
    return output


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="根据 hyperedge.recommendation 绘制推荐关联图"
    )
    p.add_argument("--db", type=str, required=True, help="main.db 路径")
    p.add_argument(
        "--output", "-o", type=str, default="outputs/recommendation_graph.png"
    )
    p.add_argument(
        "--all-nodes",
        action="store_true",
        help="画出全部超边（含无 recommendation 的孤立点）；默认只画有边的节点",
    )
    p.add_argument(
        "--hyperedge-ids",
        type=str,
        default=None,
        help="只绘制这些超边 id（逗号分隔），及其之间的推荐边",
    )
    p.add_argument(
        "--min-degree",
        type=int,
        default=0,
        help="节点至少有多少条推荐边才显示（默认 0）",
    )
    p.add_argument(
        "--label",
        type=str,
        default="auto",
        choices=("auto", "id", "name", "both"),
        help="节点标签：id / name / auto / both",
    )
    p.add_argument("--label-max-len", type=int, default=18)
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--figsize", type=str, default="14,10")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--node-size-scale", type=float, default=1.0)
    p.add_argument(
        "--layout-scale",
        type=float,
        default=3.2,
        help="分量内节点间距系数，越大越松（默认 3.2）",
    )
    p.add_argument(
        "--component-gap",
        type=float,
        default=1.9,
        help="连通分量之间的间隙系数（默认 1.9）",
    )
    p.add_argument("--title", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    he_ids = None
    if args.hyperedge_ids:
        he_ids = [
            int(x.strip()) for x in args.hyperedge_ids.split(",") if x.strip()
        ]

    fw, fh = (float(x) for x in args.figsize.split(","))
    try:
        draw_recommendation_graph(
            db_path=Path(args.db),
            output=Path(args.output),
            all_nodes=args.all_nodes,
            hyperedge_ids=he_ids,
            min_degree=args.min_degree,
            label=args.label,
            label_max_len=args.label_max_len,
            show_labels=not args.no_labels,
            figsize=(fw, fh),
            dpi=args.dpi,
            title=args.title,
            seed=args.seed,
            node_size_scale=args.node_size_scale,
            layout_scale=args.layout_scale,
            component_gap=args.component_gap,
        )
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
