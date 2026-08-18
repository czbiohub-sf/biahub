"""Posting pipeline progress to Slack.

The Nextflow pipeline calls ``biahub nf notify`` at run start, after each step
completes, and once at run end (see ``nextflow/modules/notify.nf``). Everything
here is deliberately dependency-free (stdlib ``urllib``) and structured as pure
functions so the payload shaping, truncation, and retry rules are unit-testable
without a network.

Two environment variables drive it, both read at call time and never written to
a file, a config default, or a process script:

``BIAHUB_SLACK_WEBHOOK``
    Incoming-webhook URL. **A credential.** When unset, messages are printed
    instead of posted and the exit status is still 0, so a Slack-less user is
    never blocked and a run is never broken by a missing webhook.
``BIAHUB_SLACK_ID``
    The Slack member ID of the person running the reconstruction, used for the
    ``@``-mention on the messages that need action.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

WEBHOOK_ENV = "BIAHUB_SLACK_WEBHOOK"
SLACK_ID_ENV = "BIAHUB_SLACK_ID"

# Slack renders a colored bar down the left edge of an attachment. This is the
# legacy attachments surface rather than Block Kit, chosen deliberately: a Block
# Kit section caps its text at 3000 characters, which is the wrong direction for
# a payload whose worst case is a Python traceback.
LEVEL_COLORS = {
    "good": "#2eb886",
    "warn": "#daa038",
    "error": "#d40e0d",
}

# Slack's hard ceiling on a message is 40000 characters, but the binding limit
# is rendering: clients collapse long text behind a "Show more" link at roughly
# 3-4k. Staying under that is what keeps an alert readable without a click.
MAX_DETAIL_CHARS = 2500
MAX_TITLE_CHARS = 300

# Member IDs start with U (user) or W (Enterprise Grid user).
_SLACK_ID_RE = re.compile(r"^[UW][A-Z0-9]{6,}$")

# CSI escape sequences. cellpose progress bars and tqdm fill the SLURM task
# logs with these, and they render as mojibake inside a Slack code fence.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

_BLANK_RUN_RE = re.compile(r"\n{3,}")

# Statuses where retrying can plausibly succeed: rate limiting, transient
# gateway/backend failures, and request timeouts.
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

RETRY_DELAYS = (2.0, 5.0)
RETRY_AFTER_CAP = 30.0
REQUEST_TIMEOUT = 10.0


def normalize_slack_id(raw: str | None) -> str | None:
    """Normalize a Slack member ID to the mention form Slack actually delivers.

    Accepts the ID as copied from Slack ("Copy member ID" in a profile), and the
    ``<@U…>`` and ``@U…`` spellings people tend to paste instead.

    Parameters
    ----------
    raw : str or None
        The candidate ID.

    Returns
    -------
    str or None
        ``"<@U024BE7LH>"`` when the ID is well-formed, otherwise ``None``.

    Notes
    -----
    A malformed ID is not an error Slack reports: it renders as literal text and
    pings nobody. Callers should warn on ``None`` rather than fail silently. A
    display name (``@ivan``) never pings via the API and is rejected here.
    """
    if not raw:
        return None
    candidate = raw.strip().strip("<>").lstrip("@").strip()
    if not _SLACK_ID_RE.match(candidate):
        return None
    return f"<@{candidate}>"


def clean_and_truncate(detail: str, max_chars: int = MAX_DETAIL_CHARS) -> str:
    """Make arbitrary log output safe and small enough for a Slack code fence.

    Keeps the **tail**, because every source of detail here puts the diagnosis
    last: a Python traceback ends with the exception, Nextflow's ``Command
    error:`` block ends with the raised error, and ``tail -n 40`` of a task log
    is already a tail.

    Parameters
    ----------
    detail : str
        Raw text, typically captured stderr or a Nextflow error report.
    max_chars : int, optional
        Character budget for the result, before the truncation marker.

    Returns
    -------
    str
        Cleaned text, truncated from the front if needed.
    """
    if not detail:
        return ""

    text = _ANSI_RE.sub("", detail)
    # Progress bars rewrite one line with \r; without this the whole log
    # collapses into a single enormous line.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _BLANK_RUN_RE.sub("\n\n", text).strip()
    # A literal fence inside the detail would close our code block early.
    text = text.replace("```", "'''")

    if len(text) <= max_chars:
        return text

    kept = text[-max_chars:]
    dropped = len(text) - max_chars
    return f"… {dropped} characters truncated …\n{kept}"


def build_payload(
    title: str,
    detail: str = "",
    level: str = "info",
    mention: str | None = None,
    max_detail: int = MAX_DETAIL_CHARS,
) -> dict:
    """Build the Slack incoming-webhook JSON body.

    Parameters
    ----------
    title : str
        One-line summary. Should name the dataset.
    detail : str, optional
        Supporting text, rendered in a code fence inside the attachment.
    level : {'info', 'good', 'warn', 'error'}, optional
        Severity, which selects the attachment's color bar. ``'info'`` emits no
        attachment at all.
    mention : str or None, optional
        A normalized ``<@U…>`` mention to prepend to the message text.
    max_detail : int, optional
        Character budget passed to :func:`clean_and_truncate`.

    Returns
    -------
    dict
        A payload ready for ``json.dumps``.

    Notes
    -----
    The mention goes in the top-level ``text`` field, not in the attachment:
    mentions are only reliably delivered as a notification from ``text``.

    With ``level='info'`` and no detail the payload is byte-identical to the
    plain ``{"text": …}`` this module's shell predecessor sent, so existing
    call sites are unaffected by the richer formatting.
    """
    text = " ".join(part for part in (mention, _one_line(title)) if part)
    payload: dict = {"text": text}

    body = clean_and_truncate(detail, max_detail)
    color = LEVEL_COLORS.get(level)
    if body or color:
        attachment: dict = {"mrkdwn_in": ["text"], "fallback": _one_line(title)}
        if color:
            attachment["color"] = color
        if body:
            attachment["text"] = f"```{body}```"
        payload["attachments"] = [attachment]

    return payload


def _one_line(title: str) -> str:
    """Collapse a title to a single line within Slack's practical title budget."""
    collapsed = " ".join(title.split())
    if len(collapsed) <= MAX_TITLE_CHARS:
        return collapsed
    return collapsed[: MAX_TITLE_CHARS - 1] + "…"


def post_with_retry(
    webhook: str,
    payload: dict,
    attempts: int = len(RETRY_DELAYS) + 1,
    sleep=time.sleep,
) -> tuple[bool, str]:
    """POST a payload to a Slack incoming webhook, retrying only what can recover.

    Parameters
    ----------
    webhook : str
        The incoming-webhook URL. Never logged.
    payload : dict
        Body from :func:`build_payload`.
    attempts : int, optional
        Total tries, including the first.
    sleep : callable, optional
        Injection point for tests.

    Returns
    -------
    tuple of (bool, str)
        Whether the post succeeded, and a short human-readable status.

    Notes
    -----
    A permanent rejection (400 invalid payload, 403 action prohibited, 404
    revoked webhook, 410 archived channel) is not retried: repeating the request
    cannot change the outcome and only delays the caller. Slack's plain-text
    response body names the cause and is returned as the status.
    """
    body = json.dumps(payload).encode()
    status = "not attempted"

    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                webhook,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return True, f"HTTP {response.status}"
        except urllib.error.HTTPError as error:
            detail = _read_error_body(error)
            status = f"HTTP {error.code}: {detail}"
            if error.code not in RETRYABLE_STATUSES:
                return False, status
            delay = _retry_delay(attempt, error)
        except (urllib.error.URLError, OSError) as error:
            status = f"transport error: {_mask(str(error), webhook)}"
            delay = _retry_delay(attempt, None)

        if attempt == attempts - 1:
            break
        sleep(delay)

    return False, status


def _retry_delay(attempt: int, error: urllib.error.HTTPError | None) -> float:
    """Pick the backoff before the next attempt, honouring ``Retry-After``."""
    if error is not None:
        raw = error.headers.get("Retry-After") if error.headers else None
        if raw:
            try:
                return min(float(raw), RETRY_AFTER_CAP)
            except ValueError:
                pass
    return RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]


def _read_error_body(error: urllib.error.HTTPError) -> str:
    """Read Slack's plain-text rejection reason, which names the actual problem."""
    try:
        return error.read().decode(errors="replace").strip()[:200] or error.reason
    except Exception:
        return str(error.reason)


def _mask(message: str, webhook: str) -> str:
    """Keep the webhook URL out of any message we print."""
    return message.replace(webhook, "<webhook>") if webhook else message


def render_for_terminal(payload: dict) -> str:
    """Flatten a payload back to plain text, for the no-webhook fallback."""
    lines = [payload.get("text", "")]
    for attachment in payload.get("attachments", []):
        if attachment.get("text"):
            lines.append(attachment["text"])
    return "\n".join(line for line in lines if line)


def should_send(state_dir: str, key: str, min_interval: float) -> bool:
    """Report whether ``key`` was last sent longer ago than ``min_interval``.

    Used only for run start/end, to absorb the relaunch loops that happen while
    debugging a config (five launches in ten minutes is an observed pattern).
    Per-step messages need nothing like this: they are deduplicated by
    Nextflow's own task cache.

    Parameters
    ----------
    state_dir : str
        Directory holding one marker file per key. Created if absent.
    key : str
        Marker name, e.g. ``"run-start"``.
    min_interval : float
        Seconds that must have passed since the last successful send.

    Returns
    -------
    bool
        True when the message should be sent.
    """
    if min_interval <= 0:
        return True
    marker = os.path.join(state_dir, f"{key}.sent")
    try:
        age = time.time() - os.path.getmtime(marker)
    except OSError:
        return True
    return age >= min_interval


def record_sent(state_dir: str, key: str) -> None:
    """Stamp ``key`` as just sent.

    Called only after a successful post, so a Slack outage does not silently
    swallow the next message.
    """
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, f"{key}.sent"), "w") as handle:
            handle.write(f"{time.time()}\n")
    except OSError:
        # State is an optimization, not a correctness requirement.
        pass


def send(
    title: str,
    detail: str = "",
    level: str = "info",
    ping: bool = False,
    slack_id: str | None = None,
    max_detail: int = MAX_DETAIL_CHARS,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Build and post one notification.

    Parameters
    ----------
    title : str
        One-line summary; should name the dataset.
    detail : str, optional
        Supporting text for the code fence.
    level : {'info', 'good', 'warn', 'error'}, optional
        Severity driving the attachment color.
    ping : bool, optional
        Whether to ``@``-mention the operator. Reserve this for messages that
        need action — run end and failures — so the mention keeps meaning
        something.
    slack_id : str or None, optional
        Member ID override; defaults to ``$BIAHUB_SLACK_ID``.
    max_detail : int, optional
        Character budget for the detail block.
    dry_run : bool, optional
        Render and print without posting.

    Returns
    -------
    tuple of (bool, str)
        Whether a post happened, and a human-readable status. Callers should not
        turn a False into a non-zero exit: a failed notification must never fail
        a reconstruction.
    """
    mention = None
    if ping:
        raw = slack_id if slack_id is not None else os.environ.get(SLACK_ID_ENV, "")
        mention = normalize_slack_id(raw)
        if raw and mention is None:
            print(
                f"[notify] ignoring malformed Slack ID {raw!r} — "
                f"expected a member ID like U024BE7LH "
                f"(Slack profile → Copy member ID); display names never ping",
            )

    payload = build_payload(title, detail, level, mention, max_detail)

    if dry_run:
        print(json.dumps(payload, indent=2))
        return False, "dry run"

    webhook = os.environ.get(WEBHOOK_ENV, "")
    if not webhook:
        print(f"[notify] {WEBHOOK_ENV} unset — printing instead of posting")
        print(render_for_terminal(payload))
        return False, f"{WEBHOOK_ENV} unset"

    ok, status = post_with_retry(webhook, payload)
    if not ok:
        # Print the message so it is not lost, and say why on stderr.
        print(render_for_terminal(payload))
        print(f"[notify] Slack post failed ({status})", flush=True)
    return ok, status
