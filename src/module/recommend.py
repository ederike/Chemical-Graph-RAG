"""
相似超边推荐（离线、可单独运行，不参与 build 流水线）。

流程：
  1) 在 node.name 列用配置关键词做「包含」多次查找，合并去重
  2) 从 node.vdb 取已有 embedding（无向量则按配置跳过）
  3) HDBSCAN 聚类（簇个数不限；min_cluster_size 控制最小簇）
  4) 将超过 max_cluster_size 的簇再拆成「每簇最多 N 个元素」的子簇
     （sklearn HDBSCAN 的 max_cluster_size 会把过大簇直接标噪声，
      不会拆分，故用二次切分落实「最大簇内元素数」语义）
  5) 簇内节点映射到超边；≥2 个不同超边时互写 recommendation（id 逗号分隔）
  6) 每次运行先清空全部 hyperedge.recommendation 再重写
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans

from ..utils.config import Config
from ..utils.database import BaseDB, BaseVDB

class Recommend:
    def __init__(
        self,
        db: Dict[str, BaseDB],
        vdb: Dict[str, BaseVDB],
        logger: logging.Logger,
        config: Config,
    ):
        self.db = db
        self.vdb = vdb
        self.logger = logger
        self.config = config

    def run(self) -> dict:
        """
        完整跑一遍推荐计算并写库。返回统计摘要。
        """
        cfg = self.config.recommend
        self.logger.info(
            f"[recommend] start keywords={cfg.keywords} "
            f"max_cluster_size={cfg.max_cluster_size} "
            f"min_cluster_size={cfg.min_cluster_size} "
            f"metric={cfg.metric} seed={cfg.random_seed}"
        )

        # 可复现：固定 numpy 种子（HDBSCAN 本身近似确定）
        np.random.seed(int(cfg.random_seed))

        nodes = self._collect_nodes_by_keywords(cfg.keywords)
        self.logger.info(f"[recommend] keyword-matched nodes={len(nodes)}")
        if not nodes:
            self._clear_all_recommendations()
            return {'matched_nodes': 0, 'clustered_nodes': 0, 'clusters': 0, 'updated_hyperedges': 0}

        vectors, used_nodes, skipped = self._attach_vectors(
            nodes, skip_missing=bool(cfg.skip_missing_embedding)
        )
        self.logger.info(
            f"[recommend] with_embedding={len(used_nodes)} "
            f"skipped_missing={skipped}"
        )
        if len(used_nodes) < int(cfg.min_cluster_size):
            self.logger.warning(
                f"[recommend] too few nodes with embedding "
                f"({len(used_nodes)} < min_cluster_size={cfg.min_cluster_size}); "
                f"clear recommendations and exit"
            )
            self._clear_all_recommendations()
            return {
                'matched_nodes': len(nodes),
                'clustered_nodes': 0,
                'clusters': 0,
                'updated_hyperedges': 0,
                'skipped_missing': skipped,
            }

        labels = self._cluster(vectors, cfg)
        rec_map, n_clusters, n_noise = self._labels_to_recommendations(used_nodes, labels)
        self.logger.info(
            f"[recommend] clusters={n_clusters} noise={n_noise} "
            f"hyperedges_with_rec={len(rec_map)}"
        )

        updated = self._write_recommendations(rec_map)
        self.logger.info(f"[recommend] done updated_hyperedges={updated}")
        return {
            'matched_nodes': len(nodes),
            'clustered_nodes': len(used_nodes),
            'clusters': n_clusters,
            'noise': n_noise,
            'updated_hyperedges': updated,
            'skipped_missing': skipped,
        }

    def _collect_nodes_by_keywords(self, keywords: List[str]) -> List[dict]:
        node_db = self.db['node']
        by_id: Dict[int, dict] = {}
        for kw in keywords or []:
            kw = (kw or '').strip()
            if not kw:
                continue
            # 包含匹配，仅 name 列
            sql = f"SELECT * FROM {node_db.table} WHERE name LIKE ?"
            rows = node_db.db.execute(sql, (f'%{kw}%',)) or []
            for row in rows:
                nid = row.get('id')
                if nid is None:
                    continue
                by_id[int(nid)] = row
            self.logger.info(f"[recommend] keyword={kw!r} hits={len(rows)}")
        return list(by_id.values())

    def _faiss_id_to_vector_map(self) -> Dict[int, np.ndarray]:
        """Build id → vector from FAISS (supports sharded node VDB)."""
        base_vdb = self.vdb.get('node')
        if base_vdb is None:
            return {}
        # Prefer BaseVDB helper (mono + sharded).
        if hasattr(base_vdb, 'id_to_vector_map'):
            try:
                return base_vdb.id_to_vector_map()
            except Exception as e:
                self.logger.error(f"[recommend] id_to_vector_map failed: {e}")
                return {}
        index = getattr(getattr(base_vdb, 'vdb', None), 'vdb', None)
        if index is None or getattr(index, 'ntotal', 0) <= 0:
            return {}

        try:
            id_map = faiss.vector_to_array(index.id_map)
        except Exception as e:
            self.logger.error(f"[recommend] cannot read faiss id_map: {e}")
            return {}

        # IndexIDMap.reconstruct 不可用，走底层 index
        inner = getattr(index, 'index', None)
        if inner is None:
            self.logger.error("[recommend] faiss IndexIDMap has no inner index")
            return {}

        out: Dict[int, np.ndarray] = {}
        for pos, nid in enumerate(id_map):
            try:
                vec = inner.reconstruct(int(pos))
            except Exception:
                continue
            out[int(nid)] = np.asarray(vec, dtype=np.float64)
        return out

    def _attach_vectors(
        self, nodes: List[dict], skip_missing: bool = True
    ) -> Tuple[np.ndarray, List[dict], int]:
        id2vec = self._faiss_id_to_vector_map()
        used: List[dict] = []
        vecs: List[np.ndarray] = []
        skipped = 0
        for n in nodes:
            nid = n.get('id')
            if nid is None:
                skipped += 1
                continue
            vec = id2vec.get(int(nid))
            if vec is None:
                skipped += 1
                if not skip_missing:
                    self.logger.warning(
                        f"[recommend] node id={nid} has no embedding"
                    )
                continue
            used.append(n)
            vecs.append(vec)

        if not vecs:
            return np.zeros((0, 0), dtype=np.float64), [], skipped
        return np.vstack(vecs), used, skipped

    def _prepare_matrix(self, vectors: np.ndarray, metric: str) -> Tuple[np.ndarray, str]:
        """
        cosine → L2 归一化后用 euclidean（与角度距离单调一致，数值更稳）。
        返回 (X, sklearn_metric)。
        """
        X = np.asarray(vectors, dtype=np.float64)
        m = (metric or 'euclidean').strip().lower()
        if m == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X = X / norms
            return X, 'euclidean'
        return X, m if m else 'euclidean'

    def _chunk_indices(self, indices: np.ndarray, max_size: int) -> List[np.ndarray]:
        """按原序硬切成长度 <= max_size 的块。"""
        indices = np.asarray(indices, dtype=int)
        return [
            indices[i:i + max_size]
            for i in range(0, len(indices), max_size)
        ]

    def _split_indices_by_max_size(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        max_size: int,
        metric: str,
        random_seed: int = 42,
        _depth: int = 0,
    ) -> List[np.ndarray]:
        """
        将一组索引切成每组长度 <= max_size 的子组。

        高维 embedding 上层次聚类极易极不均衡，故用 KMeans 做均衡切分；
        若仍有超限子簇则递归再切。遇到重复向量 / 切分失败 / 过深时，
        回退到按原序硬切块，避免死递归。
        """
        indices = np.asarray(indices, dtype=int)
        n = len(indices)
        if n <= max_size:
            return [indices]
        if n == 0:
            return []

        # 深度保护 + 重复向量：无法再分时硬切
        if _depth >= 8:
            return self._chunk_indices(indices, max_size)

        sub = X[indices]
        # 唯一行数不足时 KMeans 无法拆开
        try:
            # 粗略判重：四舍五入后看唯一行
            uniq = np.unique(np.round(sub, decimals=6), axis=0)
            n_unique = int(uniq.shape[0])
        except Exception:
            n_unique = n

        n_clusters = max(2, int(math.ceil(n / float(max_size))))
        if n_unique < 2:
            return self._chunk_indices(indices, max_size)
        n_clusters = min(n_clusters, n_unique, n)

        try:
            model = KMeans(
                n_clusters=n_clusters,
                random_state=int(random_seed),
                n_init=5,
            )
            labs = model.fit_predict(sub)
        except Exception as e:
            self.logger.warning(
                f"[recommend] kmeans split failed n={n}: {e}; "
                f"fall back to sequential chunks"
            )
            return self._chunk_indices(indices, max_size)

        groups: Dict[int, List[int]] = defaultdict(list)
        for local_i, lab in enumerate(labs):
            groups[int(lab)].append(int(indices[local_i]))

        # 若完全没拆开（全在一个标签），硬切
        if len(groups) <= 1:
            return self._chunk_indices(indices, max_size)

        out: List[np.ndarray] = []
        for g in groups.values():
            g_arr = np.asarray(g, dtype=int)
            if len(g_arr) <= max_size:
                out.append(g_arr)
            elif len(g_arr) == n:
                # 与父组同大：无法推进，硬切
                out.extend(self._chunk_indices(g_arr, max_size))
            else:
                out.extend(
                    self._split_indices_by_max_size(
                        X,
                        g_arr,
                        max_size,
                        metric,
                        random_seed=random_seed + 1,
                        _depth=_depth + 1,
                    )
                )
        return out

    def _enforce_max_cluster_size(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        max_size: int,
        metric: str,
        random_seed: int = 42,
    ) -> np.ndarray:
        """过大簇二次切分；结果标签重新编号，噪声保持 -1。"""
        labels = np.asarray(labels, dtype=int)
        new_labels = np.full(labels.shape, -1, dtype=int)
        next_lab = 0
        for lab in sorted(set(labels.tolist())):
            if lab < 0:
                continue
            idx = np.where(labels == lab)[0]
            for group in self._split_indices_by_max_size(
                X, idx, max_size, metric, random_seed=random_seed
            ):
                if len(group) < 2:
                    # 单点：保持噪声，后续跳过
                    continue
                new_labels[group] = next_lab
                next_lab += 1
        return new_labels

    def _cluster(self, vectors: np.ndarray, cfg) -> np.ndarray:
        n = vectors.shape[0]
        min_cs = max(2, int(cfg.min_cluster_size))
        max_cs = int(cfg.max_cluster_size)
        if max_cs < min_cs:
            self.logger.warning(
                f"[recommend] max_cluster_size={max_cs} < min_cluster_size={min_cs}; "
                f"raise max to {min_cs}"
            )
            max_cs = min_cs

        min_samples = cfg.min_samples
        if min_samples is None:
            min_samples = min_cs
        else:
            min_samples = max(1, int(min_samples))

        if n < min_cs:
            return np.full(n, -1, dtype=int)

        method = (cfg.cluster_selection_method or 'eom').strip().lower()
        if method not in ('eom', 'leaf'):
            method = 'eom'

        X, sk_metric = self._prepare_matrix(vectors, cfg.metric)
        seed = int(cfg.random_seed)

        # 注意：不把 max_cluster_size 直接传给 HDBSCAN。
        # sklearn 实现会把超过上限的自然簇整簇标为噪声，而不是拆成小簇。
        clusterer = HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=min_samples,
            metric=sk_metric,
            cluster_selection_method=method,
            copy=True,
        )
        labels = np.asarray(clusterer.fit_predict(X), dtype=int)
        n_raw = len({int(x) for x in labels if int(x) >= 0})
        n_noise_raw = int(np.sum(labels < 0))
        self.logger.info(
            f"[recommend] HDBSCAN raw clusters={n_raw} noise={n_noise_raw}"
        )

        labels = self._enforce_max_cluster_size(
            X, labels, max_cs, sk_metric, random_seed=seed
        )
        n_final = len({int(x) for x in labels if int(x) >= 0})
        n_noise_final = int(np.sum(labels < 0))
        sizes = []
        for lab in set(labels.tolist()):
            if lab < 0:
                continue
            sizes.append(int(np.sum(labels == lab)))
        max_obs = max(sizes) if sizes else 0
        self.logger.info(
            f"[recommend] after max_size={max_cs} split: "
            f"clusters={n_final} noise={n_noise_final} max_obs_size={max_obs}"
        )
        return labels

    def _labels_to_recommendations(
        self, nodes: List[dict], labels: np.ndarray
    ) -> Tuple[Dict[int, Set[int]], int, int]:
        """
        返回: (he_id -> 推荐 he_id 集合, 有效簇数, 噪声点数)
        规则：
          - 单点簇 / 噪声跳过
          - 簇内唯一超边 < 2 跳过
          - 不写自己 id
        """
        by_label: Dict[int, List[dict]] = defaultdict(list)
        noise = 0
        for node, lab in zip(nodes, labels):
            lab = int(lab)
            if lab < 0:
                noise += 1
                continue
            by_label[lab].append(node)

        rec_map: Dict[int, Set[int]] = defaultdict(set)
        n_clusters = 0
        for lab, members in by_label.items():
            if len(members) < 2:
                # 防御：min_cluster_size 应已保证，仍跳过
                continue
            he_ids: Set[int] = set()
            for m in members:
                hid = m.get('hyperedge_id')
                if hid is None:
                    continue
                he_ids.add(int(hid))
            if len(he_ids) < 2:
                continue
            n_clusters += 1
            for hid in he_ids:
                rec_map[hid].update(he_ids - {hid})

        return rec_map, n_clusters, noise

    def _clear_all_recommendations(self) -> None:
        """每次重跑先清空全部 recommendation。"""
        try:
            self.db['hyperedge'].update_key('recommendation', None)
        except Exception as e:
            self.logger.warning(f"[recommend] clear recommendation failed: {e}")
            # 回退：逐行写空
            rows = self.db['hyperedge'].search_all() or []
            if rows:
                self.db['hyperedge'].update([
                    {'id': r['id'], 'recommendation': None} for r in rows if r.get('id') is not None
                ])

    def _write_recommendations(self, rec_map: Dict[int, Set[int]]) -> int:
        self._clear_all_recommendations()
        if not rec_map:
            return 0

        updates = []
        for hid, others in rec_map.items():
            if not others:
                continue
            # 稳定排序，逗号分隔
            rec_str = ','.join(str(x) for x in sorted(others))
            updates.append({'id': int(hid), 'recommendation': rec_str})

        if updates:
            self.db['hyperedge'].update(updates)
        return len(updates)
