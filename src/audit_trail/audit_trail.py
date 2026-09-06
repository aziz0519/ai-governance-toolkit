import json
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path


class AuditTrail:
    """Audit trail for ML model predictions with hash chaining."""

    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.previous_hash = "genesis"

    def _get_log_path(self) -> Path:
        """Return today's log file path."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    def _compute_hash(self, record: dict) -> str:
        """Compute SHA-256 hash chained to the previous record."""
        record_bytes = json.dumps(record, sort_keys=True).encode()
        combined = f"{self.previous_hash}:{record_bytes.decode()}".encode()
        return hashlib.sha256(combined).hexdigest()

    def _write_record(self, record: dict) -> None:
        """Append a JSON record to today's log file."""
        with open(self._get_log_path(), "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def log_prediction(
        self,
        model_id: str,
        model_version: str,
        input_data: dict,
        output: dict,
        confidence: float | None = None,
        latency_ms: float | None = None,
        escalated: bool = False,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Log a single prediction event. Returns the request ID."""

        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "timestamp": timestamp,
            "event": "prediction",
            "request_id": request_id,
            "model_id": model_id,
            "model_version": model_version,
            "input": input_data,
            "output": output,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "escalated": escalated,
            "user_id": user_id,
            "metadata": metadata or {},
        }

        record_hash = self._compute_hash(record)
        record["hash"] = record_hash
        record["previous_hash"] = self.previous_hash
        self.previous_hash = record_hash

        self._write_record(record)
        return request_id

    def log_human_review(
        self,
        request_id: str,
        reviewer_id: str,
        original_prediction: dict,
        reviewer_decision: str,
        reviewer_override: dict | None = None,
        reason: str = "",
    ) -> None:
        """Log a human review decision linked to the original prediction."""

        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "timestamp": timestamp,
            "event": "human_review",
            "request_id": request_id,
            "reviewer_id": reviewer_id,
            "original_prediction": original_prediction,
            "reviewer_decision": reviewer_decision,
            "reviewer_override": reviewer_override,
            "reason": reason,
        }

        record_hash = self._compute_hash(record)
        record["hash"] = record_hash
        record["previous_hash"] = self.previous_hash
        self.previous_hash = record_hash

        self._write_record(record)

    def log_model_update(
        self,
        old_version: str,
        new_version: str,
        change_description: str,
        updated_by: str,
    ) -> None:
        """Log a model version change."""

        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "timestamp": timestamp,
            "event": "model_update",
            "old_version": old_version,
            "new_version": new_version,
            "change_description": change_description,
            "updated_by": updated_by,
        }

        record_hash = self._compute_hash(record)
        record["hash"] = record_hash
        record["previous_hash"] = self.previous_hash
        self.previous_hash = record_hash

        self._write_record(record)


def verify_chain(log_file: str) -> bool:
    """Verify the hash chain integrity of an audit log file."""

    with open(log_file, "r") as f:
        lines = f.readlines()

    previous_hash = "genesis"
    for i, line in enumerate(lines):
        record = json.loads(line)
        stored_hash = record.pop("hash")
        stored_previous = record.pop("previous_hash")

        if stored_previous != previous_hash:
            print(f"Chain broken at line {i + 1}: "
                  f"expected previous_hash {previous_hash}, "
                  f"got {stored_previous}")
            return False

        # Recompute the hash from the record contents
        record_bytes = json.dumps(record, sort_keys=True).encode()
        combined = f"{previous_hash}:{record_bytes.decode()}".encode()
        recomputed = hashlib.sha256(combined).hexdigest()

        if recomputed != stored_hash:
            print(f"Hash mismatch at line {i + 1}: "
                  f"record has been tampered with")
            return False

        previous_hash = stored_hash

    print(f"Chain verified: {len(lines)} records, all hashes valid.")
    return True