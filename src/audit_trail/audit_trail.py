import contextlib
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, TextIO

MAX_RECORD_BYTES = 1_000_000


def _compute_hash(record: dict, previous_hash: str) -> str:
    """Return SHA-256 chained to ``previous_hash``.

    This is tamper-evidence, not authentication: anyone who can rewrite the
    log can recompute the chain. It detects accidental edits and naive tampering.
    """
    record_bytes = json.dumps(record, sort_keys=True).encode("utf-8")
    combined = f"{previous_hash}:{record_bytes.decode('utf-8')}".encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def _validate_chain_lines(lines: Iterable[str]) -> tuple[str, int]:
    """Validate an iterable of JSONL records and return (tail_hash, count)."""
    previous_hash = "genesis"
    count = 0

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        record = json.loads(line)
        stored_hash = record.get("hash")
        stored_previous = record.get("previous_hash")
        payload = {
            key: value
            for key, value in record.items()
            if key not in ("hash", "previous_hash")
        }

        if stored_previous != previous_hash:
            raise ValueError(
                f"Chain broken at line {line_number}: expected previous_hash "
                f"{previous_hash}, got {stored_previous}"
            )

        recomputed = _compute_hash(payload, previous_hash)
        if recomputed != stored_hash:
            raise ValueError(
                f"Hash mismatch at line {line_number}: record has been tampered with"
            )

        previous_hash = stored_hash
        count += 1

    return previous_hash, count


def _tail_hash(handle: TextIO) -> str:
    """Validate a log handle from the start and return its tail hash."""
    handle.seek(0)
    tail_hash, _ = _validate_chain_lines(handle)
    handle.seek(0, os.SEEK_END)
    return tail_hash


@contextlib.contextmanager
def _exclusive_append(path: Path) -> Iterator[TextIO]:
    """Open a JSONL log for atomic, locked append access."""
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.close(fd)

    with open(path, "a+", encoding="utf-8") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

        try:
            handle.seek(0, os.SEEK_END)
            yield handle
        finally:
            handle.flush()
            os.fsync(handle.fileno())
            if os.name == "nt":
                import msvcrt

                # Windows unlocks the region starting at the current position.
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class AuditTrail:
    """Append-only audit trail with a hash chain.

    The chain is per log file and is resumed on construction, after process
    restarts, and after UTC-day rollover. Hashes are unkeyed, so the chain is
    tamper-evident but not cryptographically authenticated.
    """

    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.previous_hash = "genesis"
        self._active_log_path: Path | None = None
        self._activate_log(self._get_log_path())

    def _get_log_path(self) -> Path:
        """Return today's log file path."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    def _activate_log(self, path: Path) -> None:
        """Make ``path`` the active chain, resuming from its tail if present."""
        if path == self._active_log_path:
            return

        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                self.previous_hash = _tail_hash(handle)
        else:
            self.previous_hash = "genesis"

        self._active_log_path = path

    @staticmethod
    def _compute_hash(record: dict, previous_hash: str) -> str:
        """Compute SHA-256 chained to ``previous_hash``."""
        return _compute_hash(record, previous_hash)

    def _append_event(self, event: dict) -> str:
        """Append a fully formed event and return its record hash."""
        log_path = self._get_log_path()
        if log_path != self._active_log_path:
            self._activate_log(log_path)

        with _exclusive_append(log_path) as handle:
            # Other processes may have appended since __init__ or since our
            # last write. Re-read the tail while holding the lock.
            self.previous_hash = _tail_hash(handle)

            record = dict(event)
            record_hash = _compute_hash(record, self.previous_hash)
            record["hash"] = record_hash
            record["previous_hash"] = self.previous_hash

            line = json.dumps(record, sort_keys=True) + "\n"
            if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
                raise ValueError(
                    f"Audit record exceeds {MAX_RECORD_BYTES} bytes"
                )

            handle.write(line)
            self.previous_hash = record_hash
            self._active_log_path = log_path

        return record_hash

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

        self._append_event({
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
        })
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

        self._append_event({
            "timestamp": timestamp,
            "event": "human_review",
            "request_id": request_id,
            "reviewer_id": reviewer_id,
            "original_prediction": original_prediction,
            "reviewer_decision": reviewer_decision,
            "reviewer_override": reviewer_override,
            "reason": reason,
        })

    def log_model_update(
        self,
        old_version: str,
        new_version: str,
        change_description: str,
        updated_by: str,
    ) -> None:
        """Log a model version change."""

        timestamp = datetime.now(timezone.utc).isoformat()

        self._append_event({
            "timestamp": timestamp,
            "event": "model_update",
            "old_version": old_version,
            "new_version": new_version,
            "change_description": change_description,
            "updated_by": updated_by,
        })


def verify_chain(log_file: str) -> bool:
    """Verify the hash chain integrity of an audit log file."""

    with open(log_file, "r", encoding="utf-8") as handle:
        try:
            _, count = _validate_chain_lines(handle)
        except (ValueError, json.JSONDecodeError) as exc:
            print(exc)
            return False

    print(f"Chain verified: {count} records, all hashes valid.")
    return True
