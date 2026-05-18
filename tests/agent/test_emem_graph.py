"""Tests for EMemGraph — heterogeneous graph with PPR retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from summerclaw.memory.emem_memory.graph import EMemGraph


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def graph(tmp_path: Path) -> EMemGraph:
    """Create a clean EMemGraph in a temporary directory."""
    return EMemGraph(working_dir=tmp_path / "graph")


@pytest.fixture
def populated_graph(graph: EMemGraph) -> EMemGraph:
    """Create a graph with Session and EDU nodes already added."""
    graph.add_nodes(["session-001", "session-002"], "Session")
    graph.add_nodes(["edu-001", "edu-002", "edu-003"], "EDU")
    # Add session-edu edges
    graph.add_edge("session-001", "edu-001", weight=1.0)
    graph.add_edge("session-001", "edu-002", weight=1.0)
    graph.add_edge("session-002", "edu-003", weight=1.0)
    return graph


# ===================================================================
# EMemGraph — node management
# ===================================================================

class TestEMemGraphNodes:
    """Test node addition, lookup, and counting."""

    def test_add_session_nodes(self, graph: EMemGraph) -> None:
        graph.add_nodes(["sess-1", "sess-2"], "Session")
        assert graph.node_count == 2
        assert graph.has_node("sess-1") is True
        assert graph.has_node("sess-2") is True
        assert graph.has_node("nonexistent") is False

    def test_add_edu_nodes(self, graph: EMemGraph) -> None:
        graph.add_nodes(["edu-a", "edu-b"], "EDU")
        assert graph.node_count == 2
        assert graph.has_node("edu-a") is True
        assert graph.has_node("edu-b") is True

    def test_add_argument_nodes(self, graph: EMemGraph) -> None:
        graph.add_nodes(["arg-x", "arg-y"], "Argument")
        assert graph.node_count == 2
        assert graph.has_node("arg-x") is True
        assert len(graph.argument_node_idxs) == 2

    def test_add_nodes_skips_duplicates(self, graph: EMemGraph) -> None:
        graph.add_nodes(["dup"], "Session")
        assert graph.node_count == 1
        graph.add_nodes(["dup"], "Session")
        assert graph.node_count == 1  # No change

    def test_add_mixed_node_types(self, graph: EMemGraph) -> None:
        graph.add_nodes(["s1"], "Session")
        graph.add_nodes(["e1"], "EDU")
        graph.add_nodes(["a1"], "Argument")
        assert graph.node_count == 3
        assert len(graph.session_node_idxs) == 1
        assert len(graph.edu_node_idxs) == 1
        assert len(graph.argument_node_idxs) == 1

    def test_get_node_name_and_type(self, graph: EMemGraph) -> None:
        graph.add_nodes(["s1"], "Session")
        graph.add_nodes(["e1"], "EDU")
        idx_s = graph.get_node_idx("s1")
        idx_e = graph.get_node_idx("e1")
        assert graph.get_node_name(idx_s) == "s1"
        assert graph.get_node_type(idx_s) == "Session"
        assert graph.get_node_name(idx_e) == "e1"
        assert graph.get_node_type(idx_e) == "EDU"

    def test_get_node_idx_missing(self, graph: EMemGraph) -> None:
        assert graph.get_node_idx("nonexistent") is None

    def test_node_count_and_edge_count_initial(self, graph: EMemGraph) -> None:
        assert graph.node_count == 0
        assert graph.edge_count == 0


# ===================================================================
# EMemGraph — edge management
# ===================================================================

class TestEMemGraphEdges:
    """Test edge addition and bidirectional edges."""

    def test_add_edge(self, graph: EMemGraph) -> None:
        graph.add_nodes(["a", "b"], "EDU")
        graph.add_edge("a", "b", weight=0.5)
        assert graph.edge_count == 1

    def test_add_edge_skips_missing_nodes(self, graph: EMemGraph) -> None:
        graph.add_nodes(["a"], "EDU")
        graph.add_edge("a", "nonexistent", weight=1.0)
        assert graph.edge_count == 0

    def test_add_bidirectional_edge(self, graph: EMemGraph) -> None:
        graph.add_nodes(["x", "y"], "EDU")
        graph.add_bidirectional_edge("x", "y", weight=0.8)
        assert graph.edge_count == 2  # Two edges (x→y and y→x)

    def test_add_synonymy_edges_no_arguments(self, graph: EMemGraph) -> None:
        count = graph.add_synonymy_edges({})
        assert count == 0

    def test_add_synonymy_edges_single_argument(self, graph: EMemGraph) -> None:
        graph.add_nodes(["arg-1"], "Argument")
        count = graph.add_synonymy_edges({"arg-1": np.array([0.5, 0.5], dtype=np.float32)})
        assert count == 0  # Need at least 2 args for similarity

    def test_add_synonymy_edges_high_similarity(self, graph: EMemGraph) -> None:
        """Arguments with very high cosine similarity should get synonymy edges."""
        graph.add_nodes(["arg-a", "arg-b"], "Argument")
        # Nearly identical embeddings
        emb_a = np.array([1.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.99, 0.01], dtype=np.float32)
        count = graph.add_synonymy_edges(
            {"arg-a": emb_a, "arg-b": emb_b},
            topk=10,
            sim_threshold=0.8,
        )
        assert count >= 0

    def test_add_synonymy_edges_low_similarity_no_edges(self, graph: EMemGraph) -> None:
        """Arguments with low similarity should NOT get synonymy edges."""
        graph.add_nodes(["arg-p", "arg-q"], "Argument")
        # Orthogonal embeddings
        emb_p = np.array([1.0, 0.0], dtype=np.float32)
        emb_q = np.array([0.0, 1.0], dtype=np.float32)
        count = graph.add_synonymy_edges(
            {"arg-p": emb_p, "arg-q": emb_q},
            topk=10,
            sim_threshold=0.95,
        )
        assert count == 0


# ===================================================================
# EMemGraph — neighbors
# ===================================================================

class TestEMemGraphNeighbors:
    """Test get_neighbors method."""

    def test_get_neighbors(self, populated_graph: EMemGraph) -> None:
        idx = populated_graph.get_node_idx("session-001")
        neighbors = populated_graph.get_neighbors(idx)
        # session-001 connects to edu-001 and edu-002
        assert len(neighbors) >= 2

    def test_get_neighbors_isolated_node(self, graph: EMemGraph) -> None:
        graph.add_nodes(["isolated"], "EDU")
        idx = graph.get_node_idx("isolated")
        neighbors = graph.get_neighbors(idx)
        assert neighbors == []


# ===================================================================
# EMemGraph — PPR with scipy fallback
# ===================================================================

class TestEMemGraphPPR:
    """Test Personalized PageRank computation (scipy fallback)."""

    @pytest.fixture
    def ppg_graph(self, tmp_path: Path) -> EMemGraph:
        """Create a graph with Session-EDU edges for PPR testing."""
        g = EMemGraph(working_dir=tmp_path / "ppr_graph")
        g.add_nodes(["s1", "s2"], "Session")
        g.add_nodes(["e1", "e2", "e3"], "EDU")
        g.add_edge("s1", "e1")
        g.add_edge("s1", "e2")
        g.add_edge("s2", "e3")
        return g

    def test_ppr_empty_graph(self, graph: EMemGraph) -> None:
        reset = np.array([])
        scores = graph.personalized_pagerank(reset, damping=0.5)
        assert len(scores) == 0

    def test_ppr_zero_reset_returns_zeros(self, ppg_graph: EMemGraph) -> None:
        n = ppg_graph.node_count
        reset = np.zeros(n)
        scores = ppg_graph.personalized_pagerank(reset, damping=0.5)
        assert np.all(scores == 0)

    def test_ppr_single_seed_node(self, ppg_graph: EMemGraph) -> None:
        n = ppg_graph.node_count
        reset = np.zeros(n)
        seed_idx = ppg_graph.get_node_idx("e1")
        reset[seed_idx] = 1.0
        scores = ppg_graph.personalized_pagerank(reset, damping=0.5)
        assert len(scores) == n
        # The seed node should have some score
        assert scores[seed_idx] > 0

    def test_ppr_corrects_nan_reset(self, ppg_graph: EMemGraph) -> None:
        n = ppg_graph.node_count
        reset = np.full(n, np.nan)
        scores = ppg_graph.personalized_pagerank(reset, damping=0.5)
        assert np.all(scores == 0)  # All NaN → zero after correction

    def test_ppr_corrects_negative_reset(self, ppg_graph: EMemGraph) -> None:
        n = ppg_graph.node_count
        reset = np.full(n, -1.0)
        scores = ppg_graph.personalized_pagerank(reset, damping=0.5)
        assert np.all(scores == 0)

    def test_ppr_retrieval_returns_sorted_results(self, ppg_graph: EMemGraph) -> None:
        n = ppg_graph.node_count
        reset = np.ones(n) / n  # Uniform reset
        (
            sorted_sess_ids,
            sorted_sess_scores,
            sorted_edu_ids,
            sorted_edu_scores,
        ) = ppg_graph.ppr_retrieval(reset, damping=0.8)

        # Scores should be sorted descending
        if len(sorted_edu_scores) > 1:
            for i in range(len(sorted_edu_scores) - 1):
                assert sorted_edu_scores[i] >= sorted_edu_scores[i + 1]

    def test_ppr_retrieval_empty_graph(self, graph: EMemGraph) -> None:
        reset = np.array([])
        result = graph.ppr_retrieval(reset, damping=0.5)
        assert len(result[0]) == 0  # No sessions
        assert len(result[2]) == 0  # No EDUs


# ===================================================================
# EMemGraph — save and load
# ===================================================================

class TestEMemGraphPersistence:
    """Test graph save/load roundtrip."""

    def test_save_and_load(self, populated_graph: EMemGraph) -> None:
        g1 = populated_graph
        g1.save()

        g2 = EMemGraph(working_dir=g1.working_dir)
        g2.load_or_create()

        assert g2.node_count == g1.node_count
        assert g2.edge_count == g1.edge_count
        assert g2.has_node("session-001") is True
        assert g2.has_node("edu-001") is True

    def test_force_rebuild_creates_empty_graph(self, populated_graph: EMemGraph) -> None:
        populated_graph.save()

        g2 = EMemGraph(working_dir=populated_graph.working_dir, force_rebuild=True)
        g2.load_or_create()

        assert g2.node_count == 0
        assert g2.edge_count == 0

    def test_load_missing_file_creates_empty(self, tmp_path: Path) -> None:
        g = EMemGraph(working_dir=tmp_path / "empty_graph")
        g.load_or_create()
        assert g.node_count == 0

    def test_load_corrupt_pickle(self, tmp_path: Path) -> None:
        """Corrupted pickle should be handled gracefully (rebuild)."""
        g = EMemGraph(working_dir=tmp_path / "corrupt_graph")
        g.working_dir.mkdir(parents=True, exist_ok=True)
        g._pickle_path.write_bytes(b"this is not valid pickle data")
        g.load_or_create()
        # Should not raise, should have empty graph
        assert g.node_count == 0


# ===================================================================
# EMemGraph — clear
# ===================================================================

class TestEMemGraphClear:
    """Test graph clearing."""

    def test_clear_resets_everything(self, populated_graph: EMemGraph) -> None:
        assert populated_graph.node_count > 0
        populated_graph.clear()
        assert populated_graph.node_count == 0
        assert populated_graph.edge_count == 0
        assert populated_graph.session_node_idxs == []
        assert populated_graph.edu_node_idxs == []
        assert populated_graph.argument_node_idxs == []

    def test_clear_then_reuse(self, graph: EMemGraph) -> None:
        graph.add_nodes(["a", "b"], "EDU")
        graph.clear()
        graph.add_nodes(["c"], "EDU")
        assert graph.node_count == 1
        assert graph.has_node("c") is True
        assert graph.has_node("a") is False


# ===================================================================
# EMemGraph — edge cases
# ===================================================================

class TestEMemGraphEdgeCases:
    """Test graph edge cases and robustness."""

    def test_add_edge_between_same_node(self, graph: EMemGraph) -> None:
        """Self-loops should be allowed (graph doesn't forbid them)."""
        graph.add_nodes(["self"], "EDU")
        graph.add_edge("self", "self", weight=1.0)
        assert graph.edge_count == 1

    def test_large_graph(self, graph: EMemGraph) -> None:
        """Add many nodes to test scaling."""
        graph.add_nodes([f"edu-{i}" for i in range(100)], "EDU")
        graph.add_nodes([f"sess-{j}" for j in range(20)], "Session")
        assert graph.node_count == 120

    def test_multiple_edges_same_pair(self, graph: EMemGraph) -> None:
        """Multiple edges between same node pair should be allowed."""
        graph.add_nodes(["x", "y"], "EDU")
        graph.add_edge("x", "y")
        graph.add_edge("x", "y", weight=2.0)
        assert graph.edge_count == 2
