import json
import os

import pytest

# Committed thresholds. Changing these requires a PR and a review -- that is the
# entire point: the bar moves deliberately, not because someone was in a hurry.
MIN_ACCURACY = 0.97
MIN_RECALL_POISONOUS = 0.98
MAX_CV_STD = 0.05

METRICS_PATH = "metrics.json"


@pytest.fixture(scope="module")
def metrics():
    if not os.path.exists(METRICS_PATH):
        pytest.fail(
            f"{METRICS_PATH} not found -- did the 'Train model' workflow step run?"
        )
    with open(METRICS_PATH) as f:
        return json.load(f)


def test_accuracy_above_threshold(metrics):
    assert metrics["accuracy"] >= MIN_ACCURACY, (
        f"QUALITY GATE FAILED: accuracy {metrics['accuracy']:.4f} "
        f"< threshold {MIN_ACCURACY}"
    )


def test_poisonous_recall_above_threshold(metrics):
    assert metrics["recall_poisonous"] >= MIN_RECALL_POISONOUS, (
        f"QUALITY GATE FAILED: recall on poisonous class "
        f"{metrics['recall_poisonous']:.4f} < threshold {MIN_RECALL_POISONOUS}. "
        f"Missing a poisonous mushroom is the costly error."
    )


def test_model_is_stable(metrics):
    assert metrics["cv_accuracy_std"] <= MAX_CV_STD, (
        f"cross-validation std {metrics['cv_accuracy_std']:.4f} exceeds "
        f"{MAX_CV_STD} -- the headline accuracy is not reproducible"
    )


def test_training_set_not_silently_shrunk(metrics):
    assert metrics["n_train"] == 6093, (
        f"expected 6093 training rows, got {metrics['n_train']} -- "
        f"a preprocessing change is dropping data"
    )