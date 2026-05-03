from __future__ import annotations

import math

import pytest

from hiring_agents.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k

RANKED = ["a", "b", "c", "d", "e"]
POS = {"a", "c"}


def test_recall_all_positives_in_top_k():
    assert recall_at_k(RANKED, POS, k=3) == pytest.approx(1.0)


def test_recall_partial():
    assert recall_at_k(RANKED, POS, k=1) == pytest.approx(0.5)


def test_recall_none_in_top_k():
    assert recall_at_k(["x", "y"], POS, k=2) == pytest.approx(0.0)


def test_recall_empty_positives():
    assert recall_at_k(RANKED, set(), k=5) == pytest.approx(0.0)


def test_precision_perfect():
    assert precision_at_k(["a", "c"], POS, k=2) == pytest.approx(1.0)


def test_precision_partial():
    assert precision_at_k(RANKED, POS, k=5) == pytest.approx(0.4)


def test_precision_zero_k():
    assert precision_at_k(RANKED, POS, k=0) == pytest.approx(0.0)


def test_ndcg_perfect():
    # both positives at rank 1 and 2 → DCG = IDCG
    assert ndcg_at_k(["a", "c", "x"], POS, k=3) == pytest.approx(1.0)


def test_ndcg_empty_positives():
    assert ndcg_at_k(RANKED, set(), k=5) == pytest.approx(0.0)


def test_ndcg_no_hits():
    assert ndcg_at_k(["x", "y", "z"], POS, k=3) == pytest.approx(0.0)


def test_ndcg_single_hit_at_rank2():
    # only "c" at position 2 (index 1); IDCG = 1/log2(2)=1.0; DCG = 1/log2(3)
    result = ndcg_at_k(["x", "c"], {"c"}, k=2)
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert result == pytest.approx(expected)
