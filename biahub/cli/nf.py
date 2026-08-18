import pathlib
import tempfile

import click

from iohub.ngff import open_ome_zarr

from biahub.utils import notify as notify_utils


@click.group("nf")
def nf_cli():
    """Nextflow-oriented utility commands.

    Generic helpers shared across Nextflow pipelines. Step-specific init/run
    logic lives on each step's own CLI command (e.g. ``biahub deskew``).
    """


@nf_cli.command("list-positions")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
def list_positions(input_zarr: str):
    """List position keys in a plate zarr (one per line, for Nextflow fan-out)."""
    with open_ome_zarr(input_zarr, mode="r") as plate:
        for name, _ in plate.positions():
            click.echo(name)


@nf_cli.command("notify")
@click.option("--title", required=True, help="One-line summary; should name the dataset.")
@click.option("--detail", default="", help="Supporting text, shown in a code fence.")
@click.option(
    "--detail-file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Read the detail from a file. Use this for anything containing quotes, "
    "backticks, or newlines (e.g. a Nextflow error report) instead of --detail.",
)
@click.option(
    "--level",
    type=click.Choice(["info", "good", "warn", "error"]),
    default="info",
    show_default=True,
    help="Severity, which selects the attachment's color bar.",
)
@click.option(
    "--ping/--no-ping",
    default=False,
    help="@-mention $BIAHUB_SLACK_ID. Reserve for messages needing action.",
)
@click.option("--slack-id", default=None, help="Member ID override, for testing.")
@click.option("--key", default=None, help="Rate-limit key, e.g. 'run-start'.")
@click.option(
    "--min-interval",
    type=float,
    default=0.0,
    help="Skip if --key was already sent this recently, in seconds.",
)
@click.option(
    "--state-dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    default=None,
    help="Where --key markers live. Defaults to the temp dir.",
)
@click.option(
    "--max-detail",
    type=int,
    default=notify_utils.MAX_DETAIL_CHARS,
    show_default=True,
    help="Character budget for the detail block; the tail is kept.",
)
@click.option(
    "--operator",
    is_flag=True,
    help="Prepend who launched the run, from the account database (not Slack).",
)
@click.option("--dry-run", is_flag=True, help="Render the payload without posting.")
def notify(
    title: str,
    detail: str,
    detail_file: pathlib.Path | None,
    level: str,
    ping: bool,
    slack_id: str | None,
    key: str | None,
    min_interval: float,
    state_dir: pathlib.Path | None,
    max_detail: int,
    operator: bool,
    dry_run: bool,
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
        started_by = f"started by: {notify_utils.operator_label()}"
        detail = f"{started_by}\n{detail}" if detail else started_by

    resolved_state_dir = str(state_dir) if state_dir is not None else tempfile.gettempdir()

    if key and min_interval > 0:
        if not notify_utils.should_send(resolved_state_dir, key, min_interval):
            click.echo(f"[notify] {key} sent less than {min_interval:g}s ago — skipping")
            return

    ok, status = notify_utils.send(
        title=title,
        detail=detail,
        level=level,
        ping=ping,
        slack_id=slack_id,
        max_detail=max_detail,
        dry_run=dry_run,
    )

    if ok and key:
        notify_utils.record_sent(resolved_state_dir, key)
    if not ok and not dry_run:
        click.echo(f"[notify] not delivered: {status}")
