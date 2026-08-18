import json
import urllib.error

import pytest

from biahub.utils import notify


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("U024BE7LH", "<@U024BE7LH>"),
        ("<@U024BE7LH>", "<@U024BE7LH>"),
        ("@U024BE7LH", "<@U024BE7LH>"),
        ("  U024BE7LH  ", "<@U024BE7LH>"),
        ("W012ABC345", "<@W012ABC345>"),
        # A display name never pings via the API, so it must be rejected rather
        # than posted as literal text that silently notifies nobody.
        ("@ivan.ivanov", None),
        ("ivan", None),
        ("U024", None),
        ("u024be7lh", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_slack_id(raw, expected):
    assert notify.normalize_slack_id(raw) == expected


def test_clean_and_truncate_keeps_the_tail():
    text = "\n".join(f"line {i}" for i in range(1000))
    result = notify.clean_and_truncate(text, max_chars=100)

    assert result.endswith("line 999")
    assert "truncated" in result
    assert len(result) <= 100 + 60


def test_clean_and_truncate_short_text_is_untouched():
    assert notify.clean_and_truncate("boom", max_chars=100) == "boom"
    assert notify.clean_and_truncate("") == ""


def test_clean_and_truncate_strips_ansi_and_carriage_returns():
    # tqdm/cellpose progress output rewrites one line with \r and colors it.
    noisy = "\x1b[32m10%\x1b[0m\r\x1b[32m50%\x1b[0m\rdone"
    assert notify.clean_and_truncate(noisy) == "10%\n50%\ndone"


def test_clean_and_truncate_neutralizes_code_fence():
    # A literal fence in the detail would close our code block early.
    assert "```" not in notify.clean_and_truncate("before ``` after")


def test_clean_and_truncate_does_not_split_multibyte_characters():
    result = notify.clean_and_truncate("µ" * 500, max_chars=100)
    assert "�" not in result
    result.encode("utf-8")


def test_build_payload_info_level_is_plain_text():
    # Byte-identical to what the shell predecessor sent, so existing agent-driven
    # call sites are unaffected by the richer formatting.
    assert notify.build_payload("hello") == {"text": "hello"}


def test_build_payload_puts_mention_in_text_not_attachment():
    payload = notify.build_payload(
        "2026_07_14 — done", detail="stats", level="good", mention="<@U024BE7LH>"
    )

    assert payload["text"].startswith("<@U024BE7LH> ")
    assert "U024BE7LH" not in json.dumps(payload["attachments"])


@pytest.mark.parametrize(
    "level, color",
    [("good", "#2eb886"), ("warn", "#daa038"), ("error", "#d40e0d")],
)
def test_build_payload_colors_by_level(level, color):
    payload = notify.build_payload("t", detail="d", level=level)
    assert payload["attachments"][0]["color"] == color
    assert payload["attachments"][0]["text"] == "```d```"


def test_build_payload_collapses_multiline_title():
    payload = notify.build_payload("a\nb   c")
    assert payload["text"] == "a b c"


def test_build_payload_truncates_long_title():
    payload = notify.build_payload("x" * 500)
    assert len(payload["text"]) == notify.MAX_TITLE_CHARS


class _FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, body=b"invalid_payload", headers=None):
    return urllib.error.HTTPError(
        "http://example", code, "err", headers or {}, _BytesBody(body)
    )


class _BytesBody:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def close(self):
        # HTTPError treats its fp as a file object and closes it on teardown.
        pass


def test_post_with_retry_succeeds_first_try(monkeypatch):
    calls = []
    monkeypatch.setattr(
        notify.urllib.request, "urlopen", lambda *a, **k: calls.append(1) or _FakeResponse()
    )

    ok, status = notify.post_with_retry("http://hook", {"text": "x"}, sleep=lambda _: None)

    assert ok
    assert status == "HTTP 200"
    assert len(calls) == 1


@pytest.mark.parametrize("code", sorted(notify.RETRYABLE_STATUSES))
def test_post_with_retry_retries_transient_statuses(monkeypatch, code):
    calls = []

    def fail(*a, **k):
        calls.append(1)
        raise _http_error(code)

    monkeypatch.setattr(notify.urllib.request, "urlopen", fail)

    ok, _ = notify.post_with_retry("http://hook", {"text": "x"}, sleep=lambda _: None)

    assert not ok
    assert len(calls) == 3


@pytest.mark.parametrize("code", [400, 403, 404, 410])
def test_post_with_retry_does_not_retry_permanent_rejections(monkeypatch, code):
    # A revoked webhook or malformed payload cannot be fixed by repeating the
    # request; retrying only delays the caller.
    calls = []

    def fail(*a, **k):
        calls.append(1)
        raise _http_error(code, b"no_service")

    monkeypatch.setattr(notify.urllib.request, "urlopen", fail)

    ok, status = notify.post_with_retry("http://hook", {"text": "x"}, sleep=lambda _: None)

    assert not ok
    assert len(calls) == 1
    assert "no_service" in status


def test_post_with_retry_recovers_after_transient_failure(monkeypatch):
    attempts = []

    def flaky(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise _http_error(503, b"service_unavailable")
        return _FakeResponse()

    monkeypatch.setattr(notify.urllib.request, "urlopen", flaky)

    ok, _ = notify.post_with_retry("http://hook", {"text": "x"}, sleep=lambda _: None)

    assert ok
    assert len(attempts) == 2


def test_post_with_retry_honours_retry_after(monkeypatch):
    slept = []
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(429, b"rate", {"Retry-After": "7"})),
    )

    notify.post_with_retry("http://hook", {"text": "x"}, sleep=slept.append)

    assert slept and slept[0] == 7.0


def test_post_with_retry_caps_retry_after(monkeypatch):
    slept = []
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(
            _http_error(429, b"rate", {"Retry-After": "99999"})
        ),
    )

    notify.post_with_retry("http://hook", {"text": "x"}, sleep=slept.append)

    assert slept[0] == notify.RETRY_AFTER_CAP


def test_post_with_retry_masks_the_webhook_in_transport_errors(monkeypatch):
    secret = "https://hooks.slack.com/services/T0/B0/supersecret"

    def fail(*a, **k):
        raise urllib.error.URLError(f"cannot reach {secret}")

    monkeypatch.setattr(notify.urllib.request, "urlopen", fail)

    _, status = notify.post_with_retry(secret, {"text": "x"}, sleep=lambda _: None)

    assert "supersecret" not in status
    assert "<webhook>" in status


def test_send_without_webhook_prints_and_reports_unset(monkeypatch, capsys):
    monkeypatch.delenv(notify.WEBHOOK_ENV, raising=False)

    ok, status = notify.send("2026_07_14 — done", detail="stats", level="good")

    assert not ok
    assert notify.WEBHOOK_ENV in status
    output = capsys.readouterr().out
    assert "2026_07_14 — done" in output
    assert "stats" in output


def test_send_reads_slack_id_from_environment(monkeypatch, capsys):
    monkeypatch.setenv(notify.SLACK_ID_ENV, "U024BE7LH")
    monkeypatch.delenv(notify.WEBHOOK_ENV, raising=False)

    notify.send("t", ping=True)

    assert "<@U024BE7LH>" in capsys.readouterr().out


def test_send_without_ping_does_not_mention(monkeypatch, capsys):
    monkeypatch.setenv(notify.SLACK_ID_ENV, "U024BE7LH")
    monkeypatch.delenv(notify.WEBHOOK_ENV, raising=False)

    notify.send("t", ping=False)

    assert "U024BE7LH" not in capsys.readouterr().out


def test_send_warns_on_malformed_slack_id(monkeypatch, capsys):
    monkeypatch.setenv(notify.SLACK_ID_ENV, "@ivan")
    monkeypatch.delenv(notify.WEBHOOK_ENV, raising=False)

    notify.send("t", ping=True)

    assert "malformed Slack ID" in capsys.readouterr().out


def test_send_dry_run_does_not_post(monkeypatch, capsys):
    monkeypatch.setenv(notify.WEBHOOK_ENV, "http://hook")
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("dry run must not post"),
    )

    ok, status = notify.send("t", dry_run=True)

    assert not ok
    assert status == "dry run"
    assert json.loads(capsys.readouterr().out)["text"] == "t"


def test_send_prints_the_message_when_delivery_fails(monkeypatch, capsys):
    monkeypatch.setenv(notify.WEBHOOK_ENV, "http://hook")
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(404, b"no_service")),
    )

    ok, _ = notify.send("2026_07_14 — failed", level="error")

    assert not ok
    output = capsys.readouterr().out
    assert "2026_07_14 — failed" in output
    assert "Slack post failed" in output


def test_should_send_and_record_sent(tmp_path):
    state = str(tmp_path / "state")

    assert notify.should_send(state, "run-start", 900)

    notify.record_sent(state, "run-start")
    assert not notify.should_send(state, "run-start", 900)

    # A zero interval disables rate limiting entirely.
    assert notify.should_send(state, "run-start", 0)
    # An unrelated key is unaffected.
    assert notify.should_send(state, "run-end", 900)


def test_record_sent_survives_an_unwritable_state_dir(tmp_path):
    # State is an optimization; losing it must not raise into the pipeline.
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")

    notify.record_sent(str(blocker / "nested"), "run-start")
