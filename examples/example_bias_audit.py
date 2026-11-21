import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.bias_detection.bias_detection import print_bias_report, run_bias_audit

SEED = 42


def _build_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """Simulate a loan approval dataset with historical gender bias."""
    np.random.seed(SEED)
    data = pd.DataFrame({
        "income": np.random.normal(55000, 15000, n_samples),
        "credit_score": np.random.normal(680, 50, n_samples),
        "debt_ratio": np.random.uniform(0.1, 0.6, n_samples),
        "gender": np.random.choice(["male", "female"], n_samples, p=[0.6, 0.4]),
    })

    approval_prob = (
        0.3
        + 0.3 * (data["income"] > 50000).astype(float)
        + 0.2 * (data["credit_score"] > 700).astype(float)
        - 0.15 * (data["debt_ratio"] > 0.4).astype(float)
        - 0.1 * (data["gender"] == "female").astype(float)
    )
    data["approved"] = (
        approval_prob + np.random.normal(0, 0.15, n_samples) > 0.5
    ).astype(int)
    return data


def _audit(y_test, y_pred, sensitive_features, label: str) -> dict:
    result = run_bias_audit(
        y_true=y_test.values,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        demographic_parity_threshold=0.1,
        disparate_impact_threshold=0.8,
    )
    print(f"\n{label}: passed={result['passed']}")
    return result


def main() -> None:
    data = _build_dataset()

    features = ["income", "credit_score", "debt_ratio"]
    X = data[features]
    y = data["approved"]
    sensitive = data["gender"]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=SEED
    )

    model = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print_bias_report(_audit(y_test, y_pred, sens_test, "Baseline"))

    mitigator = ExponentiatedGradient(
        estimator=GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        constraints=DemographicParity(),
    )
    mitigator.fit(X_train, y_train, sensitive_features=sens_train)
    y_pred_fair = mitigator.predict(X_test, random_state=SEED)
    _audit(y_test, y_pred_fair, sens_test, "ExponentiatedGradient")

    # Fit the postprocessor on training data only. The old example fitted it on
    # X_test, leaking the evaluation set into mitigation.
    postprocessor = ThresholdOptimizer(
        estimator=GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        constraints="demographic_parity",
        prefit=False,
    )
    postprocessor.fit(X_train, y_train, sensitive_features=sens_train)
    y_pred_adjusted = postprocessor.predict(
        X_test, sensitive_features=sens_test, random_state=SEED
    )
    _audit(y_test, y_pred_adjusted, sens_test, "ThresholdOptimizer")


if __name__ == "__main__":
    main()
