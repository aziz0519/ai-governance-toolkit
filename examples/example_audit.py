import time

from src.audit_trail.audit_trail import AuditTrail


def main() -> None:
    """Log a prediction and a later human review."""
    audit = AuditTrail(log_dir="./audit_logs")

    start = time.time()
    prediction = {"class": "approved", "probability": 0.87}
    latency = (time.time() - start) * 1000

    request_id = audit.log_prediction(
        model_id="loan-approval-model",
        model_version="2.1.0",
        input_data={"income": 62000, "credit_score": 720, "debt_ratio": 0.35},
        output=prediction,
        confidence=0.87,
        latency_ms=latency,
        escalated=False,
        user_id="applicant-1234",
    )

    audit.log_human_review(
        request_id=request_id,
        reviewer_id="reviewer-jane",
        original_prediction=prediction,
        reviewer_decision="rejected",
        reviewer_override={"class": "denied", "reason": "Incomplete employment history"},
        reason=(
            "Applicant's employment history shows a 2-year gap not captured "
            "in features"
        ),
    )

    print(f"Logged prediction {request_id} and human review.")


if __name__ == "__main__":
    main()
