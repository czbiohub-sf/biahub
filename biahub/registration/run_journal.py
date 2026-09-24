"""Run journal for fallback-pass attempts, tagged with an explicit run identity.

So a resumed run never silently treats a stale (different-run) attempt as current, or
clobbers state from a run in progress elsewhere.

Fixes PR #339 review finding 5: existing repair journals (`repair_flagged.json`,
`repair_attempts.jsonl`, `repair_log.json`) have no run identity, so resuming the stock
estimate driver after a restart silently no-ops (resume logic can't tell a
this-run attempt from a stale one), truncates `not_rescued` lists, and scopes the sweep
from stale state. Production hit this in a real restart-clobber incident.
"""

from __future__ import annotations

import json
import uuid

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Attempt:
    run_id: str
    t: int
    pass_name: str
    before_score: float
    after_score: float
    accepted: bool


class RunJournal:
    """Records fallback-pass attempts for one timepoint series, tagged with a run id.

    A fresh `RunJournal()` generates a new run id. Loading from disk
    (`RunJournal.load(path)`) with no `run_id` given ALSO starts a fresh run id while
    preserving prior attempts in history -- so `attempted_this_run(t)` correctly
    returns False for anything recorded under a different (stale) run, forcing
    re-attempt on an ordinary restart. Explicitly continuing the *same* run (e.g.
    resubmitting a job that was killed mid-run, without wanting to redo already-accepted
    work) requires the caller to pass that run's id back in -- resume semantics are
    something the caller decides, never an implicit default.
    """

    def __init__(self, run_id: str | None = None):
        self.current_run_id = run_id or uuid.uuid4().hex
        self.attempts: list[Attempt] = []

    def record(
        self, t: int, pass_name: str, before_score: float, after_score: float, accepted: bool
    ) -> None:
        self.attempts.append(
            Attempt(self.current_run_id, t, pass_name, before_score, after_score, accepted)
        )

    def attempted_this_run(self, t: int, pass_name: str | None = None) -> bool:
        return any(
            a.t == t
            and a.run_id == self.current_run_id
            and (pass_name is None or a.pass_name == pass_name)
            for a in self.attempts
        )

    def accepted_this_run(self, t: int, pass_name: str | None = None) -> bool:
        return any(
            a.t == t
            and a.run_id == self.current_run_id
            and a.accepted
            and (pass_name is None or a.pass_name == pass_name)
            for a in self.attempts
        )

    def to_dict(self) -> dict:
        return {"attempts": [asdict(a) for a in self.attempts]}

    @classmethod
    def from_dict(cls, data: dict, run_id: str | None = None) -> RunJournal:
        journal = cls(run_id=run_id)
        journal.attempts = [Attempt(**a) for a in data.get("attempts", [])]
        return journal

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path, run_id: str | None = None) -> RunJournal:
        if not path.exists():
            return cls(run_id=run_id)
        return cls.from_dict(json.loads(path.read_text()), run_id=run_id)
