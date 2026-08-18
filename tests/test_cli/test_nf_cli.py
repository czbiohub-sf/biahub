import json

from click.testing import CliRunner

from biahub.cli.main import cli
from biahub.utils import notify as notify_utils


def test_list_positions(example_plate):
    plate_path, plate_dataset = example_plate
    plate_dataset.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["nf", "list-positions", "-i", str(plate_path)])
    assert result.exit_code == 0, result.output

    lines = [line for line in result.output.strip().split("\n") if line]
    assert len(lines) == 3
    assert "A/1/0" in lines
    assert "B/1/0" in lines
    assert "B/2/0" in lines


def test_notify_dry_run_renders_payload_without_posting(monkeypatch):
    monkeypatch.setenv("BIAHUB_SLACK_ID", "U024BE7LH")
    monkeypatch.delenv("BIAHUB_SLACK_WEBHOOK", raising=False)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "notify",
            "--title",
            "2026_07_14_A549 — deskew complete (2/6)",
            "--detail",
            "output: /hpc/x.zarr",
            "--level",
            "good",
            "--ping",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert payload["text"] == "<@U024BE7LH> 2026_07_14_A549 — deskew complete (2/6)"
    assert payload["attachments"][0]["color"] == "#2eb886"


def test_notify_exits_zero_without_a_webhook(monkeypatch):
    # The pipeline calls this on every step; a Slack-less environment must never
    # fail a task.
    monkeypatch.delenv("BIAHUB_SLACK_WEBHOOK", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli, ["nf", "notify", "--title", "hello"])

    assert result.exit_code == 0, result.output
    assert "hello" in result.output


def test_notify_reads_detail_from_a_file(tmp_path, monkeypatch):
    # An error report contains backticks, quotes and newlines, so the pipeline
    # passes it by path rather than on the command line.
    monkeypatch.delenv("BIAHUB_SLACK_WEBHOOK", raising=False)
    report = tmp_path / "report.txt"
    report.write_text('Command error:\n  raise ValueError("bad `config`")\n')

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "notify",
            "--title",
            "failed",
            "--detail-file",
            str(report),
            "--level",
            "error",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "bad `config`" in result.output


def test_notify_min_interval_suppresses_a_repeat(tmp_path, monkeypatch):
    # Debugging a config produces relaunch storms; run-start must not ping twice.
    monkeypatch.delenv("BIAHUB_SLACK_WEBHOOK", raising=False)
    state = tmp_path / "state"
    notify_utils.record_sent(str(state), "run-start")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "nf",
            "notify",
            "--title",
            "launched",
            "--key",
            "run-start",
            "--min-interval",
            "900",
            "--state-dir",
            str(state),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "skipping" in result.output
    assert "launched" not in result.output


def test_notify_operator_flag_names_who_launched_the_run(monkeypatch):
    monkeypatch.delenv("BIAHUB_SLACK_WEBHOOK", raising=False)
    monkeypatch.setattr(notify_utils, "operator_label", lambda: "Ivan Ivanov (ivan.ivanov)")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["nf", "notify", "--title", "started", "--detail", "input: /x", "--operator"],
    )

    assert result.exit_code == 0, result.output
    assert "started by: Ivan Ivanov (ivan.ivanov)" in result.output
    # The operator line goes first, before the rest of the detail.
    assert result.output.index("started by:") < result.output.index("input: /x")


def test_notify_without_operator_flag_omits_it(monkeypatch):
    monkeypatch.delenv("BIAHUB_SLACK_WEBHOOK", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli, ["nf", "notify", "--title", "started"])

    assert result.exit_code == 0, result.output
    assert "started by" not in result.output
