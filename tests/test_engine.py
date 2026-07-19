"""
MetricGuard - Engine Test Suite
================================
Covers the core detection logic so every function has a known-good baseline.
Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engine import (
    MetricDefinition,
    detect_definition_conflicts,
    extract_sql_features,
    find_semantic_groups,
    sql_structural_conflicts,
    trust_risk_score,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal metric dicts for isolation (no file I/O, no embeddings)
# ---------------------------------------------------------------------------

def make_metric(**kwargs) -> dict:
    """Build a valid metric dict with sensible defaults, overriding via kwargs."""
    defaults = dict(
        id="m_test",
        team="Test",
        metric_name="test_metric",
        sql="SELECT SUM(amount) FROM orders",
        description="A test metric.",
        filters=[],
        includes_refunds=None,
        time_grain="month",
    )
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# 1. Pydantic schema validation
# ---------------------------------------------------------------------------

class TestMetricDefinition:
    def test_valid_metric_passes(self):
        m = MetricDefinition(
            id="m01", team="Finance", metric_name="revenue",
            sql="SELECT SUM(amount) FROM orders",
            description="Total revenue.",
        )
        assert m.id == "m01"

    def test_empty_metric_name_raises(self):
        with pytest.raises(Exception):
            MetricDefinition(
                id="m01", team="Finance", metric_name="   ",
                sql="SELECT 1", description="desc",
            )

    def test_missing_filters_defaults_to_empty_list(self):
        m = MetricDefinition(
            id="m01", team="Finance", metric_name="revenue",
            sql="SELECT SUM(amount) FROM orders",
            description="Total revenue.",
        )
        assert m.filters == []

    def test_missing_time_grain_defaults_to_unknown(self):
        m = MetricDefinition(
            id="m01", team="Finance", metric_name="revenue",
            sql="SELECT SUM(amount) FROM orders",
            description="Total revenue.",
        )
        assert m.time_grain == "unknown"

    def test_includes_refunds_optional(self):
        m = MetricDefinition(
            id="m01", team="Finance", metric_name="revenue",
            sql="SELECT SUM(amount) FROM orders",
            description="Total revenue.",
        )
        assert m.includes_refunds is None


# ---------------------------------------------------------------------------
# 2. detect_definition_conflicts
# ---------------------------------------------------------------------------

class TestDetectDefinitionConflicts:
    def test_refund_mismatch_detected(self):
        metrics = [
            make_metric(id="m1", includes_refunds=True),
            make_metric(id="m2", includes_refunds=False),
        ]
        conflicts = detect_definition_conflicts(metrics, [0, 1])
        assert any("Refund" in c for c in conflicts)

    def test_no_conflict_when_refunds_match(self):
        metrics = [
            make_metric(id="m1", includes_refunds=False),
            make_metric(id="m2", includes_refunds=False),
        ]
        conflicts = detect_definition_conflicts(metrics, [0, 1])
        assert not any("Refund" in c for c in conflicts)

    def test_time_grain_mismatch_detected(self):
        metrics = [
            make_metric(id="m1", time_grain="month"),
            make_metric(id="m2", time_grain="30d_rolling"),
        ]
        conflicts = detect_definition_conflicts(metrics, [0, 1])
        assert any("Time window" in c for c in conflicts)

    def test_no_time_conflict_when_grains_match(self):
        metrics = [
            make_metric(id="m1", time_grain="month"),
            make_metric(id="m2", time_grain="month"),
        ]
        conflicts = detect_definition_conflicts(metrics, [0, 1])
        assert not any("Time window" in c for c in conflicts)

    def test_filter_mismatch_detected(self):
        metrics = [
            make_metric(id="m1", filters=["status = 'completed'"]),
            make_metric(id="m2", filters=[]),
        ]
        conflicts = detect_definition_conflicts(metrics, [0, 1])
        assert any("Filter" in c for c in conflicts)

    def test_no_conflict_when_all_match(self):
        metrics = [
            make_metric(id="m1", includes_refunds=False,
                        time_grain="month", filters=["status='completed'"]),
            make_metric(id="m2", includes_refunds=False,
                        time_grain="month", filters=["status='completed'"]),
        ]
        conflicts = detect_definition_conflicts(metrics, [0, 1])
        # SQL features may still differ; we only assert the metadata checks are clean
        metadata_conflicts = [c for c in conflicts
                              if "Refund" in c or "Time" in c or "Filter" in c]
        assert metadata_conflicts == []

    def test_includes_refunds_none_not_a_conflict(self):
        """None means 'not tracked' — two Nones should NOT raise a refund conflict."""
        metrics = [
            make_metric(id="m1", includes_refunds=None),
            make_metric(id="m2", includes_refunds=None),
        ]
        conflicts = detect_definition_conflicts(metrics, [0, 1])
        assert not any("Refund" in c for c in conflicts)

    def test_three_way_conflict(self):
        metrics = [
            make_metric(id="m1", time_grain="month"),
            make_metric(id="m2", time_grain="30d_rolling"),
            make_metric(id="m3", time_grain="calendar_month"),
        ]
        conflicts = detect_definition_conflicts(metrics, [0, 1, 2])
        assert any("Time window" in c for c in conflicts)


# ---------------------------------------------------------------------------
# 3. trust_risk_score
# ---------------------------------------------------------------------------

class TestTrustRiskScore:
    def test_single_team_single_conflict(self):
        # 1 × 15 + 1 × 20 = 35
        assert trust_risk_score([0], ["one conflict"]) == 35

    def test_three_teams_two_conflicts(self):
        # 3 × 15 + 2 × 20 = 45 + 40 = 85
        assert trust_risk_score([0, 1, 2], ["c1", "c2"]) == 85

    def test_caps_at_100(self):
        # 10 teams × 15 + 5 conflicts × 20 = 150 + 100 = 250 → capped at 100
        assert trust_risk_score(list(range(10)), ["c"] * 5) == 100

    def test_zero_conflicts_still_scores_by_teams(self):
        # 2 teams, 0 conflicts → 2 × 15 = 30
        assert trust_risk_score([0, 1], []) == 30

    def test_exact_100_boundary(self):
        # 4 teams × 15 + 2 conflicts × 20 = 60 + 40 = 100 (exactly at cap)
        assert trust_risk_score([0, 1, 2, 3], ["c1", "c2"]) == 100


# ---------------------------------------------------------------------------
# 4. find_semantic_groups (union-find / transitive grouping)
# ---------------------------------------------------------------------------

class TestFindSemanticGroups:
    def _make_sim_matrix(self, n: int, high_pairs: list[tuple[int, int]],
                         high_val: float = 0.9, low_val: float = 0.1) -> np.ndarray:
        """Build a symmetric sim matrix with 1.0 on diagonal."""
        m = np.full((n, n), low_val)
        np.fill_diagonal(m, 1.0)
        for i, j in high_pairs:
            m[i][j] = high_val
            m[j][i] = high_val
        return m

    def test_direct_pair_grouped(self):
        metrics = [make_metric(id=f"m{i}") for i in range(3)]
        sim = self._make_sim_matrix(3, [(0, 1)])
        groups = find_semantic_groups(metrics, sim, threshold=0.7)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_transitive_chain_grouped(self):
        """A~B and B~C but NOT A~C — all three must land in one group."""
        metrics = [make_metric(id=f"m{i}") for i in range(4)]
        sim = self._make_sim_matrix(4, [(0, 1), (1, 2)])   # A-B-C chain
        groups = find_semantic_groups(metrics, sim, threshold=0.7)
        # A, B, C should all be in one group; D is isolated
        group_sets = [set(g) for g in groups]
        assert {0, 1, 2} in group_sets

    def test_isolated_node_not_in_any_group(self):
        """A node with no high-similarity neighbours must NOT appear in any group."""
        metrics = [make_metric(id=f"m{i}") for i in range(3)]
        sim = self._make_sim_matrix(3, [(0, 1)])   # node 2 isolated
        groups = find_semantic_groups(metrics, sim, threshold=0.7)
        all_nodes = [n for g in groups for n in g]
        assert 2 not in all_nodes

    def test_no_groups_when_all_below_threshold(self):
        metrics = [make_metric(id=f"m{i}") for i in range(3)]
        sim = self._make_sim_matrix(3, [], low_val=0.3)
        groups = find_semantic_groups(metrics, sim, threshold=0.7)
        assert groups == []

    def test_all_nodes_connected(self):
        """When every pair is above threshold, one group contains all nodes."""
        n = 4
        metrics = [make_metric(id=f"m{i}") for i in range(n)]
        sim = np.full((n, n), 0.9)
        np.fill_diagonal(sim, 1.0)
        groups = find_semantic_groups(metrics, sim, threshold=0.7)
        assert len(groups) == 1
        assert len(groups[0]) == n


# ---------------------------------------------------------------------------
# 5. extract_sql_features
# ---------------------------------------------------------------------------

class TestExtractSqlFeatures:
    def test_sum_detected(self):
        f = extract_sql_features("SELECT SUM(amount) FROM orders")
        assert "SUM" in f["aggregations"]

    def test_count_distinct_detected(self):
        f = extract_sql_features(
            "SELECT COUNT(DISTINCT user_id) FROM events"
        )
        assert f["has_distinct"] is True

    def test_count_without_distinct(self):
        f = extract_sql_features("SELECT COUNT(*) FROM orders")
        assert f["has_distinct"] is False

    def test_group_by_extracted(self):
        f = extract_sql_features(
            "SELECT DATE_TRUNC('month', order_date), SUM(amount) "
            "FROM orders GROUP BY 1"
        )
        assert len(f["group_by"]) > 0

    def test_empty_sql_does_not_crash(self):
        f = extract_sql_features("")
        assert isinstance(f, dict)

    def test_malformed_sql_does_not_crash(self):
        f = extract_sql_features("THIS IS NOT SQL @@@ !!!")
        assert isinstance(f, dict)


# ---------------------------------------------------------------------------
# 6. sql_structural_conflicts
# ---------------------------------------------------------------------------

class TestSqlStructuralConflicts:
    def test_distinct_mismatch_detected(self):
        members = [
            make_metric(sql="SELECT COUNT(DISTINCT user_id) FROM events"),
            make_metric(sql="SELECT COUNT(*) FROM events"),
        ]
        conflicts = sql_structural_conflicts(members)
        assert any("DISTINCT" in c for c in conflicts)

    def test_no_conflict_when_sql_matches(self):
        sql = "SELECT COUNT(DISTINCT user_id) FROM events"
        members = [make_metric(sql=sql), make_metric(sql=sql)]
        conflicts = sql_structural_conflicts(members)
        assert conflicts == []

    def test_aggregation_mismatch_detected(self):
        members = [
            make_metric(sql="SELECT SUM(amount) FROM orders"),
            make_metric(sql="SELECT AVG(amount) FROM orders"),
        ]
        conflicts = sql_structural_conflicts(members)
        assert any("aggregation" in c.lower() for c in conflicts)


# ---------------------------------------------------------------------------
# 7. avg_similarity guard (empty pair list)
# ---------------------------------------------------------------------------

class TestAvgSimilarityGuard:
    def test_single_element_group_does_not_crash(self):
        """
        A group of 1 produces an empty pair list.
        run_analysis guards this with `if pair_sims else 1.0`.
        We test the guard expression directly.
        """
        group = [0]
        sim = np.array([[1.0]])
        pair_sims = [sim[i][j] for i in group for j in group if i < j]
        avg_sim = float(np.mean(pair_sims)) if pair_sims else 1.0
        assert avg_sim == 1.0

    def test_two_element_group_computes_correctly(self):
        group = [0, 1]
        sim = np.array([[1.0, 0.8], [0.8, 1.0]])
        pair_sims = [sim[i][j] for i in group for j in group if i < j]
        avg_sim = float(np.mean(pair_sims)) if pair_sims else 1.0
        assert abs(avg_sim - 0.8) < 1e-9
