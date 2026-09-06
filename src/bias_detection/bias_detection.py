import pandas as pd
import numpy as np
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score


def run_bias_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    demographic_parity_threshold: float = 0.1,
    disparate_impact_threshold: float = 0.8,
) -> dict:
    """
    Run a bias audit on model predictions.

    Returns a dictionary containing:
    - metric_frame: disaggregated metrics by group
    - demographic_parity_diff: difference in selection rates
    - equalized_odds_diff: difference in TPR and FPR
    - disparate_impact_ratio: selection rate ratio
    - violations: list of failed fairness checks
    """

    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
        "recall": lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
        "selection_rate": selection_rate,
    }

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    dp_diff = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    eo_diff = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )

    group_selection_rates = metric_frame.by_group["selection_rate"]
    min_rate = group_selection_rates.min()
    max_rate = group_selection_rates.max()
    disparate_impact = min_rate / max_rate if max_rate > 0 else 0.0

    violations = []

    if dp_diff > demographic_parity_threshold:
        violations.append(
            f"Demographic parity difference ({dp_diff:.4f}) exceeds "
            f"threshold ({demographic_parity_threshold})"
        )

    if disparate_impact < disparate_impact_threshold:
        violations.append(
            f"Disparate impact ratio ({disparate_impact:.4f}) below "
            f"threshold ({disparate_impact_threshold})"
        )

    return {
        "metric_frame": metric_frame,
        "demographic_parity_diff": dp_diff,
        "equalized_odds_diff": eo_diff,
        "disparate_impact_ratio": disparate_impact,
        "violations": violations,
        "passed": len(violations) == 0,
    }


def print_bias_report(audit_result: dict) -> None:
    """Print a formatted bias audit report."""

    print("=" * 60)
    print("BIAS AUDIT REPORT")
    print("=" * 60)

    print("\nMetrics by group:")
    print(audit_result["metric_frame"].by_group.to_string())

    print(f"\nDemographic parity difference: "
          f"{audit_result['demographic_parity_diff']:.4f}")
    print(f"Equalized odds difference: "
          f"{audit_result['equalized_odds_diff']:.4f}")
    print(f"Disparate impact ratio: "
          f"{audit_result['disparate_impact_ratio']:.4f}")

    if audit_result["passed"]:
        print("\nResult: PASSED -- No fairness violations detected.")
    else:
        print(f"\nResult: FAILED -- {len(audit_result['violations'])} "
              f"violation(s) detected:")
        for v in audit_result["violations"]:
            print(f"  - {v}")

    print("=" * 60)