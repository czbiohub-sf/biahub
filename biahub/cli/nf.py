import pathlib
import tempfile

from typing import Annotated, Literal

import typer

from iohub.ngff import open_ome_zarr

from biahub.utils import notify as notify_utils

cli = typer.Typer(
    add_completion=False,
    name="nf",
    help=(
        "Nextflow-oriented utility commands. Generic helpers shared across Nextflow "
        "pipelines. Step-specific init/run logic lives on each step's own CLI command."
    ),
)


@cli.command("list-positions")
def list_positions(
    input_zarr: Annotated[
        pathlib.Path,
        typer.Option("--input-zarr", "-i", exists=True),
    ],
):
    """List position keys in a plate zarr (one per line, for Nextflow fan-out)."""
    with open_ome_zarr(input_zarr, mode="r") as plate:
        for name, _ in plate.positions():
            typer.echo(name)


@cli.command("notify")
def notify(
    title: Annotated[
        str,
        typer.Option("--title", help="One-line summary; should name the dataset."),
    ],
    detail: Annotated[
        str,
        typer.Option("--detail", help="Supporting text, shown in a code fence."),
    ] = "",
    detail_file: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--detail-file",
            exists=True,
            dir_okay=False,
            help=(
                "Read the detail from a file. Use this for anything containing quotes, "
                "backticks, or newlines (e.g. a Nextflow error report) instead of --detail."
            ),
        ),
    ] = None,
    level: Annotated[
        Literal["info", "good", "warn", "error"],
        typer.Option(
            "--level",
            show_default=True,
            help="Severity, which selects the attachment's color bar.",
        ),
    ] = "info",
    ping: Annotated[
        bool,
        typer.Option(
            "--ping/--no-ping",
            help="@-mention $BIAHUB_SLACK_ID. Reserve for messages needing action.",
        ),
    ] = False,
    slack_id: Annotated[
        str | None,
        typer.Option("--slack-id", help="Member ID override, for testing."),
    ] = None,
    key: Annotated[
        str | None,
        typer.Option("--key", help="Rate-limit key, e.g. 'run-start'."),
    ] = None,
    min_interval: Annotated[
        float,
        typer.Option(
            "--min-interval",
            help="Skip if --key was already sent this recently, in seconds.",
        ),
    ] = 0.0,
    state_dir: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--state-dir",
            file_okay=False,
            help="Where --key markers live. Defaults to the temp dir.",
        ),
    ] = None,
    max_detail: Annotated[
        int,
        typer.Option(
            "--max-detail",
            show_default=True,
            help="Character budget for the detail block; the tail is kept.",
        ),
    ] = notify_utils.MAX_DETAIL_CHARS,
    operator: Annotated[
        bool,
        typer.Option(
            "--operator",
            help="Prepend who launched the run, from the account database (not Slack).",
        ),
    ] = False,
    log_file: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--log-file",
            dir_okay=False,
            help=(
                "Append delivery problems here. Use this when the caller cannot capture "
                "stdout, e.g. a Nextflow onComplete handler running during JVM shutdown."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Render the payload without posting."),
    ] = False,
):
    r"""Post a pipeline notification to Slack, falling back to the terminal.

    Reads the webhook from ``$BIAHUB_SLACK_WEBHOOK`` and the operator's member ID
    from ``$BIAHUB_SLACK_ID``. With no webhook set, the message is printed rather
    than posted.

    \b
    This command ALWAYS exits 0. A failed notification must never fail a
    reconstruction that has been running for days, so delivery problems are
    reported on stdout and swallowed.

    \b
    Examples:
      biahub nf notify --level good --title ":white_check_mark: 2026_07_14 — deskewed"
      biahub nf notify --level error --ping --title ":x: 2026_07_14 — failed" --detail-file r
    """
    if detail_file is not None:
        detail = detail_file.read_text(errors="replace")

    if operator:
        # First line of the detail: who to ask about this run. Comes from the
        # account database rather than Slack — see notify_utils.operator_label.
        who = f"operator: {notify_utils.operator_label()}"
        detail = f"{who}\n{detail}" if detail else who

    resolved_state_dir = str(state_dir) if state_dir is not None else tempfile.gettempdir()

    if key and min_interval > 0:
        if not notify_utils.should_send(resolved_state_dir, key, min_interval):
            typer.echo(f"[notify] {key} sent less than {min_interval:g}s ago — skipping")
            return

    ok, status = notify_utils.send(
        title=title,
        detail=detail,
        level=level,
        ping=ping,
        slack_id=slack_id,
        max_detail=max_detail,
        dry_run=dry_run,
        log_file=str(log_file) if log_file is not None else None,
    )

    if ok and key:
        notify_utils.record_sent(resolved_state_dir, key)
    if not ok and not dry_run:
        typer.echo(f"[notify] not delivered: {status}")
