import pytest
from metrics import accuracy, compute_auc, compute_auc2, sensitivity, sensitivity_specificity_equivalence_point, specificity


model_outputs = [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
gt_labels = [1, 0, 1, 0]


@pytest.mark.parametrize("threshold, expected", [
    (0.3, 0.75),
    (0.5, 1.0),
    (0.7, 1)
])
def test_accuracy(threshold, expected):
    result = accuracy(model_outputs, gt_labels, threshold)
    assert result == expected


@pytest.mark.parametrize("threshold, expected", [
    (0.3, 1.0),
    (0.5, 1.0),
    (0.7, 1.0)
])
def test_sensitivity(threshold, expected):
    threshold = 0.5
    result = sensitivity(model_outputs, gt_labels, threshold)
    assert result == expected


@pytest.mark.parametrize("threshold, expected", [
    (0.3, 0.5),
    (0.5, 1.0),
    (0.7, 1)
])
def test_specificity(threshold, expected):
    result = specificity(model_outputs, gt_labels, threshold)
    assert result == expected


def test_sensitivity_specificity_equivalence_point():
    result = sensitivity_specificity_equivalence_point(
        model_outputs, gt_labels)
    assert abs(result - 1.0) < 1e-5


def test_compute_auc():
    result = compute_auc(model_outputs, gt_labels)
    assert abs(result - 1.0) < 1e-5
