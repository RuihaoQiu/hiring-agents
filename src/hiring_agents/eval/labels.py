from __future__ import annotations

from typing import Callable

from hiring_agents.config import GROUND_TRUTH_PATH, LABELS_PATH
from hiring_agents.io_utils import read_json, write_json
from hiring_agents.schemas import GroundTruth

# One criterion per query: GroundTruth -> bool
CRITERIA: dict[str, Callable[[GroundTruth], bool]] = {
    "q1": lambda gt: gt.role_family == "backend" and "Python" in gt.tech_stack,
    "q2": lambda gt: gt.role_family == "ml_engineer" and "PyTorch" in gt.tech_stack,
    "q3": lambda gt: (
        gt.role_family == "frontend"
        and gt.seniority in ("senior", "staff")
        and ("React" in gt.tech_stack or "TypeScript" in gt.tech_stack)
    ),
    "q4": lambda gt: (
        gt.role_family == "data_scientist"
        and "Python" in gt.tech_stack
        and "SQL" in gt.tech_stack
    ),
    "q5": lambda gt: gt.role_family == "devops" and "Kubernetes" in gt.tech_stack,
}


def generate_labels() -> dict[str, list[str]]:
    raw = read_json(GROUND_TRUTH_PATH)
    ground_truth = {cid: GroundTruth.model_validate(v) for cid, v in raw.items()}
    return {
        qid: [cid for cid, gt in ground_truth.items() if criterion(gt)]
        for qid, criterion in CRITERIA.items()
    }


def save_labels(labels: dict[str, list[str]]) -> None:
    write_json(LABELS_PATH, labels)


def load_or_generate_labels() -> dict[str, list[str]]:
    if LABELS_PATH.exists():
        return read_json(LABELS_PATH)
    labels = generate_labels()
    save_labels(labels)
    return labels
