from typing import List
from pydantic import validate_call

from sklearn.metrics import roc_curve, auc, roc_auc_score


@validate_call
def accuracy(model_outputs: List[List[float]], gt_labels: List[int], threshold: float):
    preds = [1 if mod[1] >= threshold else 0 for mod in model_outputs]
    correct = sum([1 if p == l else 0 for p, l in zip(preds, gt_labels)])
    return correct / len(gt_labels)


@validate_call
def sensitivity(model_outputs: List[List[float]], gt_labels: List[int], threshold: float):
    preds = [1 if mo[1] >= threshold else 0 for mo in model_outputs]
    true_positives = sum(
        [1 for p, l in zip(preds, gt_labels) if p == 1 and l == 1])
    positives = sum(gt_labels)
    if positives == 0:
        return 0
    return true_positives / positives


@validate_call
def specificity(model_outputs: List[List[float]], gt_labels: List[int], threshold: float):
    preds = [1 if mo[1] >= threshold else 0 for mo in model_outputs]
    true_negatives = sum(
        [1 for p, l in zip(preds, gt_labels) if p == 0 and l == 0])
    negatives = len(gt_labels) - sum(gt_labels)
    try:
        result = true_negatives / negatives
    except ZeroDivisionError:
        result = 0
    return result


@validate_call
def sensitivity_specificity_equivalence_point(model_outputs: List[List[float]], gt_labels: List[int]):
    fpr, tpr, thresholds = roc_curve(
        gt_labels, [mo[1] for mo in model_outputs])
    # sen and spc are equal where the difference between the two is zero
    distances = [(abs(tpr[i] - (1 - fpr[i])), tpr[i], 1 - fpr[i])
                 for i in range(len(tpr))]
    my_dict = {"fpr": fpr, "tpr": tpr,
               "thresholds": thresholds, "dis": distances}
    distances.sort(key=lambda x: x[0])
    return sum(distances[0][1:]) / 2


@validate_call
def compute_auc(model_outputs: List[List[float]], gt_labels: List[int]):
    fpr, tpr, _ = roc_curve(gt_labels, [mo[1] for mo in model_outputs])
    return auc(fpr, tpr)


@validate_call
def compute_auc2(model_outputs: List[List[float]], gt_labels: List[int]):
    score = roc_auc_score(gt_labels, [mo[1] for mo in model_outputs])
    return score
