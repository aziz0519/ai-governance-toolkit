import numpy as np
from human_in_the_loop import HumanInTheLoop

hitl = HumanInTheLoop(confidence_threshold=0.85)

# Simulate 10 predictions with varying confidence
np.random.seed(42)
for i in range(10):
    confidence = np.random.uniform(0.5, 0.99)
    prediction = {
        "class": "approved" if confidence > 0.6 else "denied",
        "probability": round(confidence, 3),
    }

    result = hitl.evaluate(
        model_id="loan-model",
        model_version="2.1.0",
        input_data={"applicant_id": f"APP-{i:04d}", "income": 50000 + i * 5000},
        prediction=prediction,
        confidence=confidence,
        user_id=f"applicant-{i}",
    )

    status = result["action"]
    print(f"Applicant APP-{i:04d}: confidence={confidence:.3f}, "
          f"action={status}")

# Show the review queue
pending = hitl.get_pending_reviews()
print(f"\n{len(pending)} predictions awaiting human review:")
for item in pending:
    print(f"  {item.review_id[:8]}... | confidence={item.confidence:.3f} "
          f"| prediction={item.prediction['class']}")

# Simulate a reviewer processing the first item
if pending:
    first = pending[0]
    hitl.submit_review(
        review_id=first.review_id,
        reviewer_id="reviewer-jane",
        decision="modified",
        override={"class": "denied", "reason": "Insufficient credit history"},
        reason="Model missed that applicant has only 6 months of credit history",
    )
    print(f"\nReviewer overrode prediction for {first.review_id[:8]}...")