import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from bias_detection import run_bias_audit, print_bias_report

np.random.seed(42)
n_samples = 2000

# Simulate a loan approval dataset with a gender feature
data = pd.DataFrame({
    "income": np.random.normal(55000, 15000, n_samples),
    "credit_score": np.random.normal(680, 50, n_samples),
    "debt_ratio": np.random.uniform(0.1, 0.6, n_samples),
    "gender": np.random.choice(["male", "female"], n_samples, p=[0.6, 0.4]),
})

# Introduce historical bias: female applicants have slightly lower
# approval rates in the training data, simulating real-world lending bias
approval_prob = (
    0.3
    + 0.3 * (data["income"] > 50000).astype(float)
    + 0.2 * (data["credit_score"] > 700).astype(float)
    - 0.15 * (data["debt_ratio"] > 0.4).astype(float)
    - 0.1 * (data["gender"] == "female").astype(float)  # historical bias
)
data["approved"] = (approval_prob + np.random.normal(0, 0.15, n_samples) > 0.5).astype(int)

features = ["income", "credit_score", "debt_ratio"]
X = data[features]
y = data["approved"]
sensitive = data["gender"]

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=42
)

# Train a model on biased data (without the gender column as a feature)
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Run the bias audit
result = run_bias_audit(
    y_true=y_test.values,
    y_pred=y_pred,
    sensitive_features=sens_test,
    demographic_parity_threshold=0.1,
    disparate_impact_threshold=0.8,
)

print_bias_report(result)


# Bias Mitigation
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.ensemble import GradientBoostingClassifier

mitigator = ExponentiatedGradient(
    estimator=GradientBoostingClassifier(n_estimators=100, random_state=42),
    constraints=DemographicParity(),
)
mitigator.fit(X_train, y_train, sensitive_features=sens_train)
y_pred_fair = mitigator.predict(X_test)


from fairlearn.postprocessing import ThresholdOptimizer

postprocessor = ThresholdOptimizer(
    estimator=model,
    constraints="demographic_parity",
    prefit=True,
)
postprocessor.fit(X_test, y_test, sensitive_features=sens_test)
y_pred_adjusted = postprocessor.predict(X_test, sensitive_features=sens_test)