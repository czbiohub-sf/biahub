from biahub.registration.run_journal import RunJournal


def test_fresh_journal_has_a_run_id_and_no_attempts():
    journal = RunJournal()
    assert journal.current_run_id
    assert journal.attempts == []
    assert journal.attempted_this_run(0) is False
    assert journal.accepted_this_run(0) is False


def test_record_and_query_within_the_same_run():
    journal = RunJournal()
    journal.record(t=5, pass_name="repair", before_score=0.6, after_score=0.6, accepted=False)
    journal.record(t=6, pass_name="repair", before_score=0.6, after_score=0.78, accepted=True)

    assert journal.attempted_this_run(5) is True
    assert journal.accepted_this_run(5) is False  # tried, not rescued

    assert journal.attempted_this_run(6) is True
    assert journal.accepted_this_run(6) is True

    assert journal.attempted_this_run(7) is False  # never touched


def test_pass_name_filter_distinguishes_repair_from_sweep():
    journal = RunJournal()
    journal.record(t=5, pass_name="repair", before_score=0.6, after_score=0.6, accepted=False)

    assert journal.attempted_this_run(5, pass_name="repair") is True
    assert journal.attempted_this_run(5, pass_name="sweep") is False


def test_save_and_load_round_trip_preserves_attempts(tmp_path):
    journal = RunJournal()
    journal.record(t=5, pass_name="repair", before_score=0.6, after_score=0.78, accepted=True)
    path = tmp_path / "journal.json"
    journal.save(path)

    # Explicitly continuing the SAME run (caller passes the id back in).
    reloaded = RunJournal.load(path, run_id=journal.current_run_id)
    assert reloaded.current_run_id == journal.current_run_id
    assert reloaded.attempted_this_run(5) is True
    assert reloaded.accepted_this_run(5) is True


def test_plain_restart_does_not_silently_skip_stale_attempts(tmp_path):
    """Regression test for PR #339 review finding 5: resuming without explicitly
    continuing the prior run must NOT treat that run's attempts as current -- an
    ordinary restart should re-attempt everything, not silently no-op."""
    first_run = RunJournal()
    first_run.record(
        t=5, pass_name="repair", before_score=0.6, after_score=0.6, accepted=False
    )
    path = tmp_path / "journal.json"
    first_run.save(path)

    # A plain restart: load with no run_id, exactly what happens if a resubmitted job
    # doesn't know (or doesn't ask) to continue the exact same run.
    restarted = RunJournal.load(path)
    assert restarted.current_run_id != first_run.current_run_id
    assert restarted.attempted_this_run(5) is False  # not attempted in THIS run
    # The history is still there for inspection, just not treated as "this run's work".
    assert len(restarted.attempts) == 1
