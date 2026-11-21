import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from model_card_generator.model_card_generator import generate_model_card
from bias_detection.bias_detection import run_bias_audit
from audit_trail.audit_trail import MAX_RECORD_BYTES, AuditTrail, verify_chain
from human_in_the_loop.human_in_the_loop import HumanInTheLoop


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


@pytest.fixture
def hitl(tmp_path):
    return HumanInTheLoop(audit=AuditTrail(log_dir=str(tmp_path)))


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

    def test_equalized_odds_violation_fails_audit(self, trained_model_and_data):
        model, X_test, y_test, sens_test = trained_model_and_data
        y_pred = model.predict(X_test)

        result = run_bias_audit(
            y_true=y_test.values,
            y_pred=y_pred,
            sensitive_features=sens_test,
            equalized_odds_threshold=0.0,
        )

        assert result["passed"] is False
        assert any("Equalized odds" in violation for violation in result["violations"])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"demographic_parity_threshold": -0.1},
            {"demographic_parity_threshold": 1.1},
            {"disparate_impact_threshold": 0},
            {"disparate_impact_threshold": 1.1},
            {"equalized_odds_threshold": -0.1},
            {"equalized_odds_threshold": 1.1},
        ],
    )
    def test_invalid_thresholds_rejected(self, trained_model_and_data, kwargs):
        model, X_test, y_test, sens_test = trained_model_and_data

        with pytest.raises(ValueError):
            run_bias_audit(
                y_true=y_test.values,
                y_pred=model.predict(X_test),
                sensitive_features=sens_test,
                **kwargs,
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

    def test_chain_resumes_after_restart(self, tmp_path):
        for instance in range(2):
            audit = AuditTrail(log_dir=str(tmp_path))
            for i in range(2):
                audit.log_prediction(
                    model_id="test-model",
                    model_version="0.1.0",
                    input_data={"value": instance * 2 + i},
                    output={"result": instance * 2 + i},
                )

        log_file = next(tmp_path.glob("*.jsonl"))
        assert verify_chain(str(log_file)) is True

        records = [json.loads(line) for line in log_file.read_text().splitlines()]
        assert len(records) == 4
        assert records[0]["previous_hash"] == "genesis"

    def test_chain_restarts_for_new_log_file(self, tmp_path):
        audit = AuditTrail(log_dir=str(tmp_path))
        for i in range(2):
            audit.log_prediction(
                model_id="test-model",
                model_version="0.1.0",
                input_data={"value": i},
                output={"result": i},
            )

        first_log = next(tmp_path.glob("*.jsonl"))
        second_log = tmp_path / "audit_next_day.jsonl"
        audit._get_log_path = lambda: second_log

        for i in range(2):
            audit.log_prediction(
                model_id="test-model",
                model_version="0.1.0",
                input_data={"value": i},
                output={"result": i},
            )

        assert first_log != second_log
        assert verify_chain(str(first_log)) is True
        assert verify_chain(str(second_log)) is True

        second_records = [
            json.loads(line) for line in second_log.read_text().splitlines()
        ]
        assert second_records[0]["previous_hash"] == "genesis"

    def test_verify_chain_detects_tampering(self, tmp_path):
        audit = AuditTrail(log_dir=str(tmp_path))
        for i in range(3):
            audit.log_prediction(
                model_id="test-model",
                model_version="0.1.0",
                input_data={"value": i},
                output={"result": i},
            )

        log_file = next(tmp_path.glob("*.jsonl"))
        lines = log_file.read_text().splitlines(keepends=True)
        tampered = json.loads(lines[1])
        tampered["confidence"] = 0.1
        lines[1] = json.dumps(tampered) + "\n"
        log_file.write_text("".join(lines))

        assert verify_chain(str(log_file)) is False

    def test_concurrent_writers_keep_valid_chain(self, tmp_path):
        audits = [AuditTrail(log_dir=str(tmp_path)) for _ in range(2)]

        def write(audit, offset):
            for i in range(10):
                audit.log_prediction(
                    model_id="test-model",
                    model_version="0.1.0",
                    input_data={"value": offset + i},
                    output={"result": offset + i},
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(write, audit, index * 10)
                for index, audit in enumerate(audits)
            ]
            for future in futures:
                future.result()

        log_file = next(tmp_path.glob("*.jsonl"))
        assert verify_chain(str(log_file)) is True
        records = [json.loads(line) for line in log_file.read_text().splitlines()]
        assert len(records) == 20

    def test_oversized_record_rejected(self, tmp_path):
        audit = AuditTrail(log_dir=str(tmp_path))

        with pytest.raises(ValueError, match="exceeds"):
            audit.log_prediction(
                model_id="test-model",
                model_version="0.1.0",
                input_data={"blob": "x" * (MAX_RECORD_BYTES + 1)},
                output={"result": 1},
            )


# ----- Human-in-the-loop Tests -----

class TestHumanInTheLoop:
    @staticmethod
    def _escalate(hitl, confidence=0.5):
        return hitl.evaluate(
            model_id="test-model",
            model_version="0.1.0",
            input_data={"feature_a": 1.0},
            prediction={"class": "positive"},
            confidence=confidence,
        )

    def test_invalid_decision_rejected(self, hitl):
        review = self._escalate(hitl)

        with pytest.raises(ValueError, match="Invalid decision"):
            hitl.submit_review(review["review_id"], "reviewer", "maybe")

        assert len(hitl.get_pending_reviews()) == 1

    def test_modified_requires_override(self, hitl):
        review = self._escalate(hitl)

        with pytest.raises(ValueError, match="override is required"):
            hitl.submit_review(review["review_id"], "reviewer", "modified")

    def test_duplicate_submission_rejected(self, hitl):
        review = self._escalate(hitl)
        hitl.submit_review(review["review_id"], "reviewer", "approved")

        with pytest.raises(ValueError, match="already been submitted"):
            hitl.submit_review(review["review_id"], "reviewer", "rejected")

        assert len(hitl.reviewed) == 1
        assert hitl.get_override_rate() == 0.0

    def test_authorization_callback_enforced(self, tmp_path):
        hitl = HumanInTheLoop(
            audit=AuditTrail(log_dir=str(tmp_path)),
            authorize_review=lambda reviewer_id: reviewer_id == "allowed",
        )
        review = self._escalate(hitl)

        with pytest.raises(PermissionError, match="not authorized"):
            hitl.submit_review(review["review_id"], "denied", "approved")

        result = hitl.submit_review(review["review_id"], "allowed", "approved")
        assert result["decision"] == "approved"

    def test_escalation_rate_counts_unique_escalations(self, hitl):
        review = self._escalate(hitl, confidence=0.5)
        self._escalate(hitl, confidence=0.9)

        assert hitl.get_escalation_rate() == 0.5

        hitl.submit_review(review["review_id"], "reviewer", "approved")
        with pytest.raises(ValueError):
            hitl.submit_review(review["review_id"], "reviewer", "rejected")

        assert hitl.get_escalation_rate() == 0.5

    def test_invalid_confidence_threshold_rejected(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            HumanInTheLoop(confidence_threshold=1.1)