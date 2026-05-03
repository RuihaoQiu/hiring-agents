from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from hiring_agents.eval.labels import CRITERIA, generate_labels
from hiring_agents.schemas import GroundTruth

_GT = {
    "c_backend_python": {
        "role_family": "backend",
        "seniority": "senior",
        "location": "Berlin, Germany",
        "total_yoe": 6,
        "tech_stack": ["Python", "Postgres"],
        "domains": ["fintech"],
    },
    "c_ml_pytorch": {
        "role_family": "ml_engineer",
        "seniority": "mid",
        "location": "London, United Kingdom",
        "total_yoe": 4,
        "tech_stack": ["PyTorch", "Python"],
        "domains": ["healthcare"],
    },
    "c_frontend_react": {
        "role_family": "frontend",
        "seniority": "senior",
        "location": "Amsterdam, Netherlands",
        "total_yoe": 7,
        "tech_stack": ["React", "TypeScript"],
        "domains": ["e-commerce"],
    },
    "c_ds_python_sql": {
        "role_family": "data_scientist",
        "seniority": "mid",
        "location": "Paris, France",
        "total_yoe": 3,
        "tech_stack": ["Python", "SQL"],
        "domains": ["adtech"],
    },
    "c_devops_k8s": {
        "role_family": "devops",
        "seniority": "staff",
        "location": "Madrid, Spain",
        "total_yoe": 10,
        "tech_stack": ["Kubernetes", "Terraform"],
        "domains": ["enterprise_saas"],
    },
    "c_unrelated": {
        "role_family": "designer",
        "seniority": "junior",
        "location": "Lisbon, Portugal",
        "total_yoe": 1,
        "tech_stack": ["Figma"],
        "domains": ["consumer"],
    },
}


@pytest.fixture
def gt_file(tmp_path: Path) -> Path:
    p = tmp_path / "ground_truth.json"
    p.write_text(json.dumps(_GT))
    return p


def test_q1_matches_backend_python(gt_file: Path):
    with patch("hiring_agents.eval.labels.GROUND_TRUTH_PATH", gt_file):
        labels = generate_labels()
    assert "c_backend_python" in labels["q1"]
    assert "c_ml_pytorch" not in labels["q1"]


def test_q2_matches_ml_pytorch(gt_file: Path):
    with patch("hiring_agents.eval.labels.GROUND_TRUTH_PATH", gt_file):
        labels = generate_labels()
    assert "c_ml_pytorch" in labels["q2"]
    assert "c_backend_python" not in labels["q2"]


def test_q3_requires_senior_and_react_or_ts(gt_file: Path):
    with patch("hiring_agents.eval.labels.GROUND_TRUTH_PATH", gt_file):
        labels = generate_labels()
    assert "c_frontend_react" in labels["q3"]
    assert "c_unrelated" not in labels["q3"]


def test_q4_matches_ds_python_sql(gt_file: Path):
    with patch("hiring_agents.eval.labels.GROUND_TRUTH_PATH", gt_file):
        labels = generate_labels()
    assert "c_ds_python_sql" in labels["q4"]
    assert "c_devops_k8s" not in labels["q4"]


def test_q5_matches_devops_kubernetes(gt_file: Path):
    with patch("hiring_agents.eval.labels.GROUND_TRUTH_PATH", gt_file):
        labels = generate_labels()
    assert "c_devops_k8s" in labels["q5"]
    assert "c_frontend_react" not in labels["q5"]


def test_unrelated_candidate_excluded_from_all(gt_file: Path):
    with patch("hiring_agents.eval.labels.GROUND_TRUTH_PATH", gt_file):
        labels = generate_labels()
    for qid in CRITERIA:
        assert "c_unrelated" not in labels[qid]


def test_criteria_cover_all_queries():
    assert set(CRITERIA.keys()) == {"q1", "q2", "q3", "q4", "q5"}
