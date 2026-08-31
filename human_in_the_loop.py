import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
from audit_trail import AuditTrail


@dataclass
class ReviewItem:
    """A prediction awaiting human review."""
    review_id: str
    request_id: str
    model_id: str
    input_data: dict
    prediction: dict
    confidence: float
    reason: str
    created_at: str
    status: str = "pending"  # pending, approved, rejected, modified


class HumanInTheLoop:
    """Confidence-threshold escalation with a review queue."""

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        audit: AuditTrail | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.review_queue: deque[ReviewItem] = deque()
        self.audit = audit or AuditTrail()
        self.reviewed: list[ReviewItem] = []
        self.total_predictions: int = 0

    def evaluate(
        self,
        model_id: str,
        model_version: str,
        input_data: dict,
        prediction: dict,
        confidence: float,
        user_id: str | None = None,
    ) -> dict:
        """
        Route a prediction based on confidence.

        Returns:
        - If confidence >= threshold: the prediction proceeds automatically
        - If confidence < threshold: the prediction is queued for human review
        """

        self.total_predictions += 1
        escalated = confidence < self.confidence_threshold

        request_id = self.audit.log_prediction(
            model_id=model_id,
            model_version=model_version,
            input_data=input_data,
            output=prediction,
            confidence=confidence,
            escalated=escalated,
            user_id=user_id,
        )

        if escalated:
            review_item = ReviewItem(
                review_id=str(uuid.uuid4()),
                request_id=request_id,
                model_id=model_id,
                input_data=input_data,
                prediction=prediction,
                confidence=confidence,
                reason=f"Confidence {confidence:.3f} below threshold "
                       f"{self.confidence_threshold}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self.review_queue.append(review_item)

            return {
                "action": "escalated",
                "request_id": request_id,
                "review_id": review_item.review_id,
                "reason": review_item.reason,
            }

        return {
            "action": "auto_approved",
            "request_id": request_id,
            "prediction": prediction,
        }

    def get_pending_reviews(self) -> list[ReviewItem]:
        """Return all pending review items."""
        return [item for item in self.review_queue if item.status == "pending"]

    def submit_review(
        self,
        review_id: str,
        reviewer_id: str,
        decision: str,
        override: dict | None = None,
        reason: str = "",
    ) -> dict:
        """
        Submit a human review decision.

        decision: 'approved', 'rejected', or 'modified'
        override: if decision is 'modified', the corrected prediction
        """

        target = None
        for item in self.review_queue:
            if item.review_id == review_id:
                target = item
                break

        if target is None:
            raise ValueError(f"Review {review_id} not found in queue")

        target.status = decision
        self.reviewed.append(target)

        self.audit.log_human_review(
            request_id=target.request_id,
            reviewer_id=reviewer_id,
            original_prediction=target.prediction,
            reviewer_decision=decision,
            reviewer_override=override,
            reason=reason,
        )

        return {
            "review_id": review_id,
            "decision": decision,
            "override": override,
        }

    def get_escalation_rate(self) -> float:
        """Calculate the percentage of all predictions that were escalated."""
        if self.total_predictions == 0:
            return 0.0
        escalated_count = len(self.reviewed) + len(self.get_pending_reviews())
        return escalated_count / self.total_predictions

    def get_override_rate(self) -> float:
        """Calculate the percentage of reviewed items where humans disagreed."""
        if not self.reviewed:
            return 0.0
        overridden = sum(
            1 for item in self.reviewed
            if item.status in ("rejected", "modified")
        )
        return overridden / len(self.reviewed)