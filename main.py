import json
from utils import bootstrap_confidence_interval
from metrics import accuracy, compute_auc, sensitivity, sensitivity_specificity_equivalence_point, specificity


def main(input_filename: str):
    with open(input_filename, "r") as f:
        data = json.load(f)

    metric_funcs = {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "SensitivitySpecificityEquivalencePoint": sensitivity_specificity_equivalence_point,
        "AUC": compute_auc
    }

    metric_func = metric_funcs[data["metric"]]
    threshold = data["threshold"] if "threshold" in data else None
    if threshold is not None:
        value = metric_func(data["model_outputs"],
                            data["gt_labels"], threshold)
    else:
        value = metric_func(data["model_outputs"], data["gt_labels"])

    result = {"value": value}

    if data["ci"]:
        lower, upper = bootstrap_confidence_interval(
            metric_func, data["model_outputs"], data["gt_labels"], threshold,
            data["num_bootstraps"], data["alpha"]
        )
        result["lower_bound"] = lower
        result["upper_bound"] = upper

    with open("data/output.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main("data/input.json")
