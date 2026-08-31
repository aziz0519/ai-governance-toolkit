import json
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from model_card_generator import generate_model_card
from bias_detection import run_bias_audit
from audit_trail import AuditTrail


# ----- Fixtures -----

@pytest.fixture
def trained_model_and_data():
    """Train a model on synthetic loan data for governance testing."""
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        "income": np.random.normal(55000, 15000, n),
        "credit_score": np.random.normal(680, 50, n),
        "debt_ratio": np.random.uniform(0.1, 0.6, n),
        "gender": np.random.choice(["male", "female"], n, p=[0.55, 0.45]),
    })
    approval_prob = (
        0.3
        + 0.3 * (data["income"] > 50000).astype(float)
        + 0.2 * (data["credit_score"] > 700).astype(float)
        - 0.15 * (data["debt_ratio"] > 0.4).astype(float)
    )
    data["approved"] = (
        approval_prob + np.random.normal(0, 0.15, n) > 0.5
    ).astype(int)

    features = ["income", "credit_score", "debt_ratio"]
    X = data[features]
    y = data["approved"]
    sensitive = data["gender"]

    X_train, X_test, y_train, y_test, _, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42
    )

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, sens_test


# ----- Model Card Tests -----

class TestModelCard:
    def test_model_card_contains_required_sections(self, trained_model_and_data):
        model, X_test, y_test, _ = trained_model_and_data
        card = generate_model_card(
            model=model,
            model_name="Test Model",
            model_version="0.1.0",
            X_test=X_test,
            y_test=y_test,

            intended_use="Testing only",
            out_of_scope_use="Production use prohibited",
            training_data_description="Synthetic test data",
            ethical_considerations="None for test",
            limitations="This is a test model",
        )

        required_sections = [
            "## Model Details",
            "## Intended Use",
            "## Out-of-Scope Use",
            "## Training Data",
            "## Evaluation Results",
            "## Ethical Considerations",
            "## Limitations",
        ]
        for section in required_sections:
            assert section in card, f"Missing required section: {section}"

    def test_model_card_includes_metrics(self, trained_model_and_data):
        model, X_test, y_test, _ = trained_model_and_data
        card = generate_model_card(
            model=model,
            model_name="Test Model",
            model_version="0.1.0",
            X_test=X_test,
            y_test=y_test,

            intended_use="Testing",
            out_of_scope_use="N/A",
            training_data_description="Synthetic",
            ethical_considerations="N/A",
            limitations="N/A",
        )
        assert "Accuracy" in card
        assert "Precision" in card
        assert "Recall" in card
        assert "F1 Score" in card


# ----- Bias Detection Tests -----

class TestBiasDetection:
    def test_disparate_impact_above_threshold(self, trained_model_and_data):
        model, X_test, y_test, sens_test = trained_model_and_data
        y_pred = model.predict(X_test)

        result = run_bias_audit(
            y_true=y_test.values,
            y_pred=y_pred,
            sensitive_features=sens_test,
            disparate_impact_threshold=0.8,
        )

        assert result["disparate_impact_ratio"] >= 0.8, (
            f"Disparate impact ratio {result['disparate_impact_ratio']:.4f} "
            f"is below the 0.8 legal threshold"
        )

    def test_demographic_parity_within_tolerance(self, trained_model_and_data):
        model, X_test, y_test, sens_test = trained_model_and_data
        y_pred = model.predict(X_test)

        result = run_bias_audit(
            y_true=y_test.values,
            y_pred=y_pred,
            sensitive_features=sens_test,
            demographic_parity_threshold=0.15,
        )

        assert abs(result["demographic_parity_diff"]) <= 0.15, (
            f"Demographic parity difference "
            f"{result['demographic_parity_diff']:.4f} exceeds tolerance"
        )


# ----- Audit Trail Tests -----

class TestAuditTrail:
    def test_audit_log_captures_prediction(self, tmp_path):
        audit = AuditTrail(log_dir=str(tmp_path))
        request_id = audit.log_prediction(
            model_id="test-model",
            model_version="0.1.0",
            input_data={"feature_a": 1.0},
            output={"class": "positive", "probability": 0.92},
            confidence=0.92,
        )

        assert request_id is not None

        log_files = list(tmp_path.glob("*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            records = [json.loads(line) for line in f]
        assert len(records) == 1
        assert records[0]["model_id"] == "test-model"
        assert records[0]["confidence"] == 0.92

    def test_audit_chain_integrity(self, tmp_path):
        audit = AuditTrail(log_dir=str(tmp_path))

        for i in range(5):
            audit.log_prediction(
                model_id="test-model",
                model_version="0.1.0",
                input_data={"value": i},
                output={"result": i * 2},
                confidence=0.9,
            )

        log_files = list(tmp_path.glob("*.jsonl"))
        with open(log_files[0]) as f:
            lines = f.readlines()

        previous_hash = "genesis"
        for line in lines:
            record = json.loads(line)
            assert record["previous_hash"] == previous_hash
            previous_hash = record["hash"]