import builtins
import json
import sys
import types
import urllib.error

import pytest

from biahub.utils import notify

_real_import = builtins.__import__


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

    # Mention last, so the message opens with its emoji and dataset.
    assert payload["text"] == "2026_07_14 — done <@U024BE7LH>"
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


class _FakeContext:
    """Stand-in for an ssl.SSLContext, recording what got loaded into it."""

    def __init__(self, ca_certs):
        self._ca_certs = list(ca_certs)
        self.loaded = []

    def get_ca_certs(self):
        return self._ca_certs

    def load_verify_locations(self, cafile=None, capath=None, cadata=None):
        self.loaded.append(cafile)


def _patch_default_context(monkeypatch, ca_certs):
    context = _FakeContext(ca_certs)
    monkeypatch.setattr(notify.ssl, "create_default_context", lambda *a, **k: context)
    return context


def test_build_ssl_context_leaves_a_populated_store_alone(monkeypatch):
    # The interpreter already found a trust store, or the user set SSL_CERT_FILE
    # and OpenSSL honoured it. Either way, do not touch it.
    context = _patch_default_context(monkeypatch, [{"subject": ()}])

    assert notify.build_ssl_context() is context
    assert context.loaded == []


def test_build_ssl_context_falls_back_to_certifi_when_the_store_is_empty(
    monkeypatch, tmp_path
):
    # A Python built for another distro's layout loads zero CAs on this cluster.
    context = _patch_default_context(monkeypatch, [])
    bundle = tmp_path / "cacert.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setattr(notify, "_fallback_ca_bundle", lambda: str(bundle))

    assert notify.build_ssl_context() is context
    # Additive: load_verify_locations adds to the store, so a capath configured
    # by the default paths is still consulted.
    assert context.loaded == [str(bundle)]


def test_build_ssl_context_warns_when_no_bundle_exists(monkeypatch, capsys):
    context = _patch_default_context(monkeypatch, [])
    monkeypatch.setattr(notify, "_fallback_ca_bundle", lambda: None)

    assert notify.build_ssl_context() is context
    assert context.loaded == []
    assert "no CA bundle found" in capsys.readouterr().out


def test_build_ssl_context_survives_an_unreadable_bundle(monkeypatch, capsys, tmp_path):
    context = _patch_default_context(monkeypatch, [])
    monkeypatch.setattr(notify, "_fallback_ca_bundle", lambda: str(tmp_path / "gone.pem"))

    def boom(cafile=None, capath=None, cadata=None):
        raise OSError("no such file")

    monkeypatch.setattr(context, "load_verify_locations", boom)

    # A bad bundle must not raise out of the notifier: the POST is allowed to
    # fail and be reported, but never to crash the caller.
    assert notify.build_ssl_context() is context
    assert "could not load CA bundle" in capsys.readouterr().out


def test_fallback_ca_bundle_prefers_certifi(monkeypatch, tmp_path):
    bundle = tmp_path / "certifi.pem"
    bundle.write_text("x")
    monkeypatch.setattr(notify, "SYSTEM_CA_BUNDLES", (str(tmp_path / "system.pem"),))
    fake = types.SimpleNamespace(where=lambda: str(bundle))
    monkeypatch.setitem(sys.modules, "certifi", fake)

    assert notify._fallback_ca_bundle() == str(bundle)


def test_fallback_ca_bundle_uses_a_system_path_without_certifi(monkeypatch, tmp_path):
    system = tmp_path / "ca-bundle.crt"
    system.write_text("x")
    monkeypatch.setattr(
        notify, "SYSTEM_CA_BUNDLES", (str(tmp_path / "absent.pem"), str(system))
    )

    def no_certifi(name, *args, **kwargs):
        if name == "certifi":
            raise ImportError("no certifi")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_certifi)

    assert notify._fallback_ca_bundle() == str(system)


def test_fallback_ca_bundle_returns_none_when_nothing_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(notify, "SYSTEM_CA_BUNDLES", (str(tmp_path / "absent.pem"),))
    monkeypatch.setitem(
        sys.modules, "certifi", types.SimpleNamespace(where=lambda: "/nope/cacert.pem")
    )

    assert notify._fallback_ca_bundle() is None


def test_post_with_retry_passes_the_context_to_urlopen(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(notify, "build_ssl_context", lambda: sentinel)
    seen = {}

    def urlopen(request, timeout=None, context=None):
        seen["context"] = context
        return _FakeResponse()

    monkeypatch.setattr(notify.urllib.request, "urlopen", urlopen)

    ok, _ = notify.post_with_retry("http://hook", {"text": "x"}, sleep=lambda _: None)

    assert ok
    assert seen["context"] is sentinel


def test_post_with_retry_builds_the_context_once_not_per_attempt(monkeypatch):
    # The bundle is a ~220 kB PEM and retries cannot change the outcome of
    # loading it, so it must be read once per call.
    builds = []
    monkeypatch.setattr(notify, "build_ssl_context", lambda: builds.append(1) or object())
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(503, b"slow_down")),
    )

    notify.post_with_retry("http://hook", {"text": "x"}, sleep=lambda _: None)

    assert len(builds) == 1


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


def test_operator_label_uses_the_account_full_name(monkeypatch):
    import pwd

    entry = pwd.struct_passwd(
        ("ivan.ivanov", "*", 5011, 5011, "Ivan Ivanov", "/home/ivan.ivanov", "/bin/bash")
    )
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: entry)

    assert notify.operator_label() == "Ivan Ivanov (ivan.ivanov)"


def test_operator_label_takes_only_the_name_from_gecos(monkeypatch):
    # GECOS is comma-separated: full name, room, work phone, home phone.
    import pwd

    entry = pwd.struct_passwd(
        ("jdoe", "*", 1, 1, "Jane Doe,Bldg 4,x1234,", "/home/jdoe", "/bin/bash")
    )
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: entry)

    assert notify.operator_label() == "Jane Doe (jdoe)"


def test_operator_label_falls_back_to_the_login_name(monkeypatch):
    import pwd

    entry = pwd.struct_passwd(("svc-runner", "*", 1, 1, "", "/home/svc", "/bin/bash"))
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: entry)

    assert notify.operator_label() == "svc-runner"


def test_operator_label_survives_an_unresolvable_account(monkeypatch):
    # A container or a UID with no passwd entry must not break the notification.
    import pwd

    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: (_ for _ in ()).throw(KeyError("uid")))
    monkeypatch.setenv("USER", "fallback.user")

    assert notify.operator_label() == "fallback.user"

    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)
    assert notify.operator_label() == "unknown"


def test_append_log_records_a_delivery_problem(tmp_path):
    log_file = tmp_path / "nested" / "notify.log"

    notify.append_log(str(log_file), "post failed (HTTP 404)")

    assert "post failed (HTTP 404)" in log_file.read_text()


def test_append_log_survives_an_unwritable_path(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")

    notify.append_log(str(blocker / "notify.log"), "anything")


def test_send_records_a_failed_post_to_the_log_file(monkeypatch, tmp_path, capsys):
    # The run-end message is sent during JVM shutdown, where the caller cannot
    # capture our stdout, so the notifier must write its own record.
    monkeypatch.setenv(notify.WEBHOOK_ENV, "http://hook")
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(404, b"no_service")),
    )
    log_file = tmp_path / "notify.log"

    notify.send("2026_07_14 — failed", level="error", log_file=str(log_file))

    recorded = log_file.read_text()
    assert "no_service" in recorded
    assert "2026_07_14 — failed" in recorded


def test_send_records_a_missing_webhook_to_the_log_file(monkeypatch, tmp_path):
    monkeypatch.delenv(notify.WEBHOOK_ENV, raising=False)
    log_file = tmp_path / "notify.log"

    notify.send("2026_07_14 — done", log_file=str(log_file))

    assert notify.WEBHOOK_ENV in log_file.read_text()


def test_send_stays_quiet_in_the_log_on_success(monkeypatch, tmp_path):
    monkeypatch.setenv(notify.WEBHOOK_ENV, "http://hook")
    monkeypatch.setattr(notify.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
    log_file = tmp_path / "notify.log"

    ok, _ = notify.send("fine", log_file=str(log_file))

    assert ok
    assert not log_file.exists()
