"""EMem graph — heterogeneous graph construction and PPR retrieval.

Builds a Session-EDU-Argument heterogeneous graph and provides
Personalized PageRank (PPR) for associative memory retrieval.

Uses ``igraph`` when available (via ``pip install summerclaw-ai[emem]``),
otherwise falls back to a pure-Python scipy-based implementation.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# igraph availability check
# ---------------------------------------------------------------------------

try:
    import igraph as ig

    _HAS_IGRAPH = True
except ImportError:
    _HAS_IGRAPH = False
    logger.debug("igraph not available; using scipy fallback for PPR")


# ---------------------------------------------------------------------------
# EMemGraph
# ---------------------------------------------------------------------------


class EMemGraph:
    """Heterogeneous graph of Session, EDU, and Argument nodes.

    Node types:
    - **Session**: A conversation session (batch of turns).
    - **EDU**: An Elementary Discourse Unit (atomic proposition).
    - **Argument**: An entity/value extracted from EDUs.

    Edge types:
    - **EDU-Argument**: Connects an EDU to its role-argument pairs.
    - **Session-EDU**: Connects a session to the EDUs it contains.
    - **Argument-Argument**: Synonymy edges between similar arguments.
    """

    def __init__(
        self,
        working_dir: Path,
        directed: bool = False,
        force_rebuild: bool = False,
    ):
        self.working_dir = working_dir
        self.directed = directed
        self.force_rebuild = force_rebuild

        self._pickle_path = working_dir / "emem_graph.pkl"

        # Node storage
        self._node_names: list[str] = []
        self._node_types: list[str] = []  # "Session", "EDU", "Argument"
        self._node_name_to_idx: dict[str, int] = {}

        # Edge storage (adjacency for PPR)
        self._edges: list[tuple[int, int, float]] = []  # (src, tgt, weight)

        # Indices for fast lookup
        self.session_node_idxs: list[int] = []
        self.edu_node_idxs: list[int] = []
        self.argument_node_idxs: list[int] = []

        # Graph object (igraph or fallback adjacency matrix)
        self._igraph: Any = None
        self._adj_matrix: Any = None  # scipy sparse matrix

        self._loaded = False

    # ------------------------------------------------------------------ load / save

    def load_or_create(self) -> None:
        """Load existing graph from pickle or create a new one."""
        if os.path.exists(self._pickle_path) and not self.force_rebuild:
            try:
                self._load()
                logger.info(
                    f"Loaded graph: {len(self._node_names)} nodes, "
                    f"{len(self._edges)} edges"
                )
                return
            except Exception:
                logger.exception("Failed to load graph pickle, rebuilding")
        self._loaded = True

    def _load(self) -> None:
        with open(self._pickle_path, "rb") as f:
            data = pickle.load(f)
        self._node_names = data["node_names"]
        self._node_types = data["node_types"]
        self._node_name_to_idx = data["node_name_to_idx"]
        self._edges = data["edges"]
        self.session_node_idxs = data.get("session_node_idxs", [])
        self.edu_node_idxs = data.get("edu_node_idxs", [])
        self.argument_node_idxs = data.get("argument_node_idxs", [])
        self._loaded = True

    def save(self) -> None:
        """Save graph to pickle."""
        working_dir = self.working_dir
        working_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "node_names": self._node_names,
            "node_types": self._node_types,
            "node_name_to_idx": self._node_name_to_idx,
            "edges": self._edges,
            "session_node_idxs": self.session_node_idxs,
            "edu_node_idxs": self.edu_node_idxs,
            "argument_node_idxs": self.argument_node_idxs,
        }
        with open(self._pickle_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved graph: {len(self._node_names)} nodes, {len(self._edges)} edges")

    # ------------------------------------------------------------------ node management

    def add_nodes(
        self,
        node_names: list[str],
        node_type: str,
    ) -> None:
        """Add nodes of a given type. Skips nodes that already exist."""
        new_count = 0
        for name in node_names:
            if name not in self._node_name_to_idx:
                idx = len(self._node_names)
                self._node_names.append(name)
                self._node_types.append(node_type)
                self._node_name_to_idx[name] = idx
                new_count += 1

                if node_type == "Session":
                    self.session_node_idxs.append(idx)
                elif node_type == "EDU":
                    self.edu_node_idxs.append(idx)
                elif node_type == "Argument":
                    self.argument_node_idxs.append(idx)

        if new_count:
            logger.debug(f"Added {new_count} {node_type} nodes")

    def has_node(self, name: str) -> bool:
        return name in self._node_name_to_idx

    @property
    def node_count(self) -> int:
        return len(self._node_names)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # ------------------------------------------------------------------ edge management

    def add_edge(self, src: str, tgt: str, weight: float = 1.0) -> None:
        """Add or update an edge between two nodes."""
        if src not in self._node_name_to_idx or tgt not in self._node_name_to_idx:
            return
        src_idx = self._node_name_to_idx[src]
        tgt_idx = self._node_name_to_idx[tgt]
        self._edges.append((src_idx, tgt_idx, weight))

    def add_bidirectional_edge(self, node_a: str, node_b: str, weight: float = 1.0) -> None:
        """Add edges in both directions."""
        self.add_edge(node_a, node_b, weight)
        self.add_edge(node_b, node_a, weight)

    def add_synonymy_edges(
        self,
        arg_embeddings: dict[str, np.ndarray],
        topk: int = 2047,
        sim_threshold: float = 0.8,
    ) -> int:
        """Add synonymy edges between similar argument nodes using KNN.

        Args:
            arg_embeddings: Mapping from argument node name to embedding vector.
            topk: Number of nearest neighbors to consider per argument.
            sim_threshold: Minimum similarity score to create an edge.

        Returns:
            Number of synonymy edges added.
        """
        arg_names = list(arg_embeddings.keys())
        if len(arg_names) < 2:
            return 0

        embeddings = np.array([arg_embeddings[n] for n in arg_names], dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms

        # Compute similarity matrix in batches
        batch_size = 1000
        num_syn_edges = 0

        for i in range(0, len(arg_names), batch_size):
            batch_end = min(i + batch_size, len(arg_names))
            batch_embs = embeddings[i:batch_end]
            sim = np.dot(batch_embs, embeddings.T)  # (batch, total)

            for j in range(len(batch_embs)):
                scores = sim[j]
                # Get top-k indices (excluding self)
                n_scores = len(scores)
                if n_scores <= 1:
                    continue
                k = min(topk + 1, n_scores - 1)
                if k <= 0:
                    continue
                top_indices = np.argpartition(-scores, k)[:k]
                top_indices = top_indices[np.argsort(-scores[top_indices])]

                for tgt_idx in top_indices:
                    if tgt_idx == i + j:
                        continue
                    score = float(scores[tgt_idx])
                    if score < sim_threshold:
                        break
                    self.add_bidirectional_edge(
                        arg_names[i + j], arg_names[tgt_idx], weight=score,
                    )
                    num_syn_edges += 2

        logger.info(f"Added {num_syn_edges} synonymy edges")
        return num_syn_edges

    # ------------------------------------------------------------------ PPR retrieval

    def personalized_pagerank(
        self,
        reset_prob: np.ndarray,
        damping: float = 0.5,
    ) -> np.ndarray:
        """Compute Personalized PageRank scores.

        Args:
            reset_prob: 1-D array of reset probabilities, one per node.
            damping: Damping factor (0–1). Higher = more teleportation.

        Returns:
            1-D array of PPR scores, one per node.
        """
        # Clean reset probabilities
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        total = reset_prob.sum()
        if total == 0:
            return np.zeros(len(self._node_names))
        reset_prob = reset_prob / total

        if _HAS_IGRAPH and self._igraph is not None:
            return self._ppr_igraph(reset_prob, damping)
        return self._ppr_scipy(reset_prob, damping)

    def _build_igraph(self) -> None:
        """Build igraph Graph from stored nodes and edges."""
        if not _HAS_IGRAPH:
            return

        g = ig.Graph(directed=self.directed)
        g.add_vertices(len(self._node_names))
        g.vs["name"] = self._node_names
        g.vs["type"] = self._node_types

        if self._edges:
            edge_list = [(s, t) for s, t, _ in self._edges]
            weights = [w for _, _, w in self._edges]
            g.add_edges(edge_list)
            g.es["weight"] = weights

        self._igraph = g

    def _ppr_igraph(self, reset_prob: np.ndarray, damping: float) -> np.ndarray:
        """PPR using igraph's built-in implementation."""
        if self._igraph is None:
            self._build_igraph()
        assert self._igraph is not None

        scores = self._igraph.personalized_pagerank(
            vertices=range(len(self._node_names)),
            damping=damping,
            directed=False,
            weights="weight",
            reset=reset_prob,
            implementation="prpack",
        )
        return np.array(scores, dtype=np.float64)

    def _ppr_scipy(self, reset_prob: np.ndarray, damping: float) -> np.ndarray:
        """PPR using power iteration on scipy sparse matrix.

        Builds the transition matrix from edges and iterates until convergence.
        """
        try:
            from scipy import sparse
        except ImportError:
            logger.warning("scipy not available; returning reset probabilities as scores")
            return reset_prob

        n = len(self._node_names)
        if n == 0:
            return np.array([])

        # Build adjacency matrix
        row_ind: list[int] = []
        col_ind: list[int] = []
        data: list[float] = []

        # Add edges (undirected — add both directions)
        for src, tgt, weight in self._edges:
            row_ind.append(tgt)
            col_ind.append(src)
            data.append(weight)
            if not self.directed:
                row_ind.append(src)
                col_ind.append(tgt)
                data.append(weight)

        if not data:
            return reset_prob.copy()

        adj = sparse.coo_matrix(
            (data, (row_ind, col_ind)),
            shape=(n, n),
            dtype=np.float64,
        ).tocsr()

        # Normalize columns to get transition matrix
        col_sums = np.array(adj.sum(axis=0)).flatten()
        col_sums[col_sums == 0] = 1.0
        inv_col_sums = sparse.diags(1.0 / col_sums)
        trans = adj @ inv_col_sums  # column-normalized

        # Power iteration
        scores = reset_prob.copy()
        for _ in range(100):
            prev = scores.copy()
            scores = damping * reset_prob + (1 - damping) * (trans @ scores)
            if np.abs(scores - prev).max() < 1e-8:
                break

        return scores

    # ------------------------------------------------------------------ retrieval results

    def ppr_retrieval(
        self,
        reset_prob: np.ndarray,
        damping: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run PPR and return sorted session and EDU results.

        Args:
            reset_prob: Reset probability array (size = node count).
            damping: PPR damping factor.

        Returns:
            Tuple of:
            - sorted_session_ids: Indices in self.session_node_idxs, sorted by score desc.
            - sorted_session_scores: Corresponding PPR scores.
            - sorted_edu_ids: Indices in self.edu_node_idxs, sorted by score desc.
            - sorted_edu_scores: Corresponding PPR scores.
        """
        ppr_scores = self.personalized_pagerank(reset_prob, damping)

        # Session scores
        if self.session_node_idxs:
            session_scores = np.array([ppr_scores[idx] for idx in self.session_node_idxs])
            sorted_session_ids = np.argsort(session_scores)[::-1]
            sorted_session_scores = session_scores[sorted_session_ids]
        else:
            sorted_session_ids = np.array([], dtype=np.intp)
            sorted_session_scores = np.array([])

        # EDU scores
        if self.edu_node_idxs:
            edu_scores = np.array([ppr_scores[idx] for idx in self.edu_node_idxs])
            sorted_edu_ids = np.argsort(edu_scores)[::-1]
            sorted_edu_scores = edu_scores[sorted_edu_ids]
        else:
            sorted_edu_ids = np.array([], dtype=np.intp)
            sorted_edu_scores = np.array([])

        return sorted_session_ids, sorted_session_scores, sorted_edu_ids, sorted_edu_scores

    def get_node_name(self, idx: int) -> str:
        return self._node_names[idx]

    def get_node_type(self, idx: int) -> str:
        return self._node_types[idx]

    def get_node_idx(self, name: str) -> int | None:
        return self._node_name_to_idx.get(name)

    def get_neighbors(self, node_idx: int) -> list[int]:
        """Get neighbors of a node (for argument expansion)."""
        neighbors: set[int] = set()
        for src, tgt, _ in self._edges:
            if src == node_idx:
                neighbors.add(tgt)
            if tgt == node_idx and not self.directed:
                neighbors.add(src)
        return list(neighbors)

    def clear(self) -> None:
        """Reset the graph to empty state."""
        self._node_names = []
        self._node_types = []
        self._node_name_to_idx = {}
        self._edges = []
        self.session_node_idxs = []
        self.edu_node_idxs = []
        self.argument_node_idxs = []
        self._igraph = None
        self._adj_matrix = None
        self._loaded = False
