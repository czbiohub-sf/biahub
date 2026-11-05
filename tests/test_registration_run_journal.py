"""The run journal: the fallback attempts a finalize writes to run_journal.json."""

import json

from biahub.registration.engine import RunJournal


def test_a_journal_has_a_run_id_and_starts_empty():
    journal = RunJournal()
    assert journal.current_run_id
    assert journal.attempts == []
    assert RunJournal(run_id="abc").current_run_id == "abc"


def test_recorded_attempts_are_saved_with_the_run_id(tmp_path):
    journal = RunJournal(run_id="run1")
    journal.record(t=5, pass_name="repair", before_score=0.6, after_score=0.6, accepted=False)
    journal.record(
        t=6,
        pass_name="repair",
        before_score=0.6,
        after_score=0.78,
        accepted=True,
        failures={"t-1": "EstimationError: too few nodes"},
    )
    path = tmp_path / "run_journal.json"
    journal.save(path)
    attempts = json.loads(path.read_text())["attempts"]
    assert [(a["t"], a["accepted"], a["run_id"]) for a in attempts] == [
        (5, False, "run1"),
        (6, True, "run1"),
    ]
    assert attempts[1]["failures"] == {"t-1": "EstimationError: too few nodes"}
