import numpy as np
from sklearn.utils import resample


def bootstrap_confidence_interval(metric_func, model_outputs, gt_labels, threshold, num_bootstraps, alpha):
    bootstrapped_scores = []
    n = len(gt_labels)
    for _ in range(num_bootstraps):
        bootstrap_model_outputs, bootstrap_gt_labels = resample(
            model_outputs, gt_labels)

        if threshold is not None:
            score = metric_func(bootstrap_model_outputs,
                                bootstrap_gt_labels, threshold)
        else:
            score = metric_func(bootstrap_model_outputs, bootstrap_gt_labels)

        bootstrapped_scores.append(score)
    lower = np.percentile(bootstrapped_scores, (1-alpha)*100/2)
    upper = np.percentile(bootstrapped_scores, (1+alpha)*100/2)
    return lower, upper
