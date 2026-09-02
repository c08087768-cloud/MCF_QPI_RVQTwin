import pandas as pd
import pytest

from mcfqpi.data.split import build_strict_group_ids, validate_manual_duplicate_review


def test_strict_groups_union_exact_and_approved_near_duplicates() -> None:
    frame = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "phase_sha256": ["same", "same", "c", "d"],
    })
    review = pd.DataFrame([{
        "sample_a": "b", "sample_b": "c", "phash_hamming": 4,
        "ssim": 0.996, "normalized_mae": 0.004, "decision": "duplicate",
    }])
    groups = build_strict_group_ids(frame, review)
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]


def test_manual_review_gate_requires_both_review_strata() -> None:
    review = pd.DataFrame([
        {"decision": "duplicate", "review_stratum": "candidate"},
        {"decision": "distinct", "review_stratum": "threshold_negative"},
    ])
    validate_manual_duplicate_review(review, minimum_per_stratum=1)
    with pytest.raises(ValueError, match="人工核查门禁"):
        validate_manual_duplicate_review(review.iloc[:1], minimum_per_stratum=1)


def test_duplicate_decision_must_satisfy_registered_thresholds() -> None:
    frame = pd.DataFrame({"sample_id": ["a", "b"], "phase_sha256": ["a", "b"]})
    review = pd.DataFrame([{
        "sample_a": "a", "sample_b": "b", "phash_hamming": 8,
        "ssim": 0.99, "normalized_mae": 0.01, "decision": "duplicate",
    }])
    with pytest.raises(ValueError, match="阈值"):
        build_strict_group_ids(frame, review)
