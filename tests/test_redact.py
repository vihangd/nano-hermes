"""Unit tests for nano_hermes.redact — pattern coverage + edge cases.

These tests don't touch the rest of nano-hermes — they exercise the redact
function in isolation. Integration tests live in test_propose_skill_redact,
test_memory_redact, and test_reflect_redact.
"""
from __future__ import annotations

import pytest

from nano_hermes.redact import RedactionResult, format_redaction_note, redact


class TestPrefixPatterns:
    @pytest.mark.parametrize("kind,sample", [
        ("openai_or_anthropic", "sk-ant-abc123def456ghi789xyz"),
        ("github_pat_classic",  "ghp_abcdef1234567890abcdef"),
        ("github_pat_fine",     "github_pat_AAA111BBB222CCC333"),
        ("github_oauth",        "gho_abcdef1234567890"),
        ("slack",               "xoxb-1234-5678-abcdefghijkl"),
        ("google_api_key",      "AIzaSy" + "X" * 32),
        ("aws_access_key_id",   "AKIAIOSFODNN7EXAMPLE"),
        ("stripe_live",         "sk_live_abcdef1234567890"),
        ("stripe_test",         "sk_test_abcdef1234567890"),
        ("sendgrid",            "SG.abcdef1234567890.xyzabc1234567"),
        ("huggingface",         "hf_abcdef1234567890"),
        ("groq",                "gsk_abcdef1234567890"),
        ("perplexity",          "pplx-abcdef1234567890"),
        ("replicate",           "r8_abcdef1234567890"),
        ("npm",                 "npm_abcdef1234567890"),
        ("digitalocean",        "dop_v1_abcdef1234567890"),
        ("tavily",              "tvly-abcdef1234567890"),
        ("exa",                 "exa_abcdef1234567890"),
        ("firecrawl",           "fc-abcdef1234567890"),
    ])
    def test_kind_detected_and_masked(self, kind, sample):
        r = redact(f"prefix {sample} suffix")
        assert sample not in r.text, f"raw {kind} secret survived: {r.text!r}"
        assert r.count >= 1
        assert kind in r.kinds

    def test_short_token_fully_masked(self):
        # 14 chars — under the 18-char cutoff → fully masked.
        r = redact("sk-shortish123")
        assert "sk-" not in r.text
        assert "█" in r.text

    def test_long_token_keeps_head_and_tail(self):
        sample = "sk-ant-1234567890abcdefghijklmnopqrstuv"
        r = redact(sample)
        assert r.count == 1
        # First 6 chars preserved.
        assert "sk-ant" in r.text
        # Last 4 chars preserved.
        assert r.text.endswith("rstuv"[-4:])
        # Middle is masked.
        assert "█" in r.text


class TestStructuralPatterns:
    def test_env_assignment_quoted(self):
        r = redact('OPENAI_API_KEY="sk-thisisalongsecret123"')
        assert "sk-thisisalongsecret123" not in r.text
        # Either env_assignment or openai_or_anthropic fires; both is fine.
        assert r.count >= 1

    def test_env_assignment_unquoted(self):
        r = redact("export FOO_TOKEN=abcd1234efgh5678ijkl")
        assert "abcd1234efgh5678ijkl" not in r.text
        assert "env_assignment" in r.kinds

    def test_env_assignment_arbitrary_secret_name(self):
        # Name contains TOKEN/SECRET/etc — generic test.
        r = redact("MY_AUTH=somelongsecretvalue123")
        assert "somelongsecretvalue123" not in r.text
        assert "env_assignment" in r.kinds

    def test_json_field_api_key(self):
        r = redact('{"api_key": "verysecretvaluehere1234"}')
        assert "verysecretvaluehere1234" not in r.text
        assert "json_field" in r.kinds

    def test_json_field_token(self):
        r = redact('{"token": "anothersecretvaluehere"}')
        assert "anothersecretvaluehere" not in r.text
        assert "json_field" in r.kinds

    def test_auth_header(self):
        r = redact("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5c")
        assert "eyJhbGciOiJIUzI1NiIsInR5c" not in r.text
        assert "auth_header" in r.kinds

    def test_telegram_bot_token(self):
        r = redact("Token: 1234567890:AAA-BBB_CCCdddEEEfffGGGhhh-IIIjjj")
        assert "1234567890:AAA-BBB_CCCdddEEEfffGGGhhh-IIIjjj" not in r.text
        assert "telegram_bot_token" in r.kinds

    def test_env_assignment_with_provider_value_no_double_count(self):
        """OPENAI_API_KEY=sk-... is ONE secret. Before the structural-first
        reorder it counted as two (env_assignment + openai_or_anthropic),
        which inflated the agent-visible 'redacted N secrets' note.
        """
        r = redact("OPENAI_API_KEY=sk-abcdefghij1234567890xyz")
        assert r.count == 1, f"expected 1, got {r.count} (kinds={r.kinds})"
        assert r.kinds == ("env_assignment",)
        assert "sk-abcdefghij1234567890xyz" not in r.text

    def test_json_field_with_provider_value_no_double_count(self):
        """Same bug on the JSON side: {"api_key": "sk-..."} should be one
        secret, classified as json_field, not (json_field + provider).
        """
        r = redact('{"api_key": "sk-abcdefghij1234567890xyz"}')
        assert r.count == 1, f"expected 1, got {r.count} (kinds={r.kinds})"
        assert r.kinds == ("json_field",)
        assert "sk-abcdefghij1234567890xyz" not in r.text

    def test_auth_header_with_provider_value_no_double_count(self):
        """Authorization: Bearer sk-... should be one secret (auth_header)."""
        r = redact("Authorization: Bearer sk-abcdefghij1234567890xyz")
        assert r.count == 1, f"expected 1, got {r.count} (kinds={r.kinds})"
        assert r.kinds == ("auth_header",)
        assert "sk-abcdefghij1234567890xyz" not in r.text


class TestNoFalsePositives:
    @pytest.mark.parametrize("safe", [
        "the api was returning 500s today",
        "Use POST /api/v1/users to create a user",
        "kubernetes secret rotation policy",
        "passwords should be at least 12 chars",
        "the token bucket algorithm uses tokens",
        "auth flow: user → server → db",
        "see Authorization section in docs",
        "JWT tokens have three parts: header.payload.signature",
        "set the API_KEY env var on the host",  # name only, no value
        "we need a credential vault",
    ])
    def test_prose_unchanged(self, safe):
        r = redact(safe)
        assert r.text == safe, f"prose mutated: {r.text!r}"
        assert r.count == 0
        assert r.kinds == ()


class TestEdges:
    def test_empty_string(self):
        r = redact("")
        assert r.text == ""
        assert r.count == 0
        assert r.kinds == ()

    def test_multiple_secrets_aggregated(self):
        r = redact("sk-aaaaa1111122222 and ghp_bbbbb2222233333")
        assert r.count == 2
        assert "openai_or_anthropic" in r.kinds
        assert "github_pat_classic" in r.kinds

    def test_kinds_sorted_and_unique(self):
        # Two openai-style keys → kinds dedupes to one entry; count is 2.
        r = redact("sk-aaaa1111aaa and sk-bbbb2222bbb")
        assert r.kinds == ("openai_or_anthropic",)
        assert r.count == 2

    def test_kinds_returned_sorted(self):
        # Provider patterns are processed in list order, but kinds is sorted.
        r = redact("ghp_aaaa1111aaaaaa and sk-bbbb2222bbbbbb")
        assert r.kinds == ("github_pat_classic", "openai_or_anthropic")

    def test_idempotent(self):
        # Masked output must not itself match any pattern (would be a bug
        # because the agent would see "(redacted N)" repeatedly grow).
        once = redact("sk-abcdefghijklmnopqrst").text
        twice = redact(once)
        assert twice.text == once
        assert twice.count == 0

    def test_unicode_preserved_around_match(self):
        r = redact("héllo 世界 sk-abcdefghij1234567 🚀")
        assert "héllo 世界" in r.text
        assert "🚀" in r.text
        assert "sk-abcdefghij1234567" not in r.text


class TestFormatRedactionNote:
    def test_no_redactions_empty_note(self):
        r = RedactionResult(text="", count=0, kinds=())
        assert format_redaction_note(r) == ""

    def test_singular(self):
        r = RedactionResult(text="", count=1, kinds=("openai_or_anthropic",))
        note = format_redaction_note(r)
        assert "1 secret-shaped string" in note
        assert "openai_or_anthropic" in note
        # Note: no trailing period; format leaves room for caller to add.
        assert note.startswith(" (")

    def test_plural(self):
        r = RedactionResult(text="", count=3, kinds=("github_pat_classic", "slack"))
        note = format_redaction_note(r)
        assert "3 secret-shaped strings" in note
        assert "github_pat_classic" in note
        assert "slack" in note


class TestRedactionResult:
    def test_dataclass_immutable(self):
        r = RedactionResult(text="x", count=1, kinds=("openai_or_anthropic",))
        with pytest.raises(Exception):
            r.text = "y"  # type: ignore[misc]

    def test_dataclass_equality(self):
        a = RedactionResult(text="x", count=0, kinds=())
        b = RedactionResult(text="x", count=0, kinds=())
        assert a == b


# --- structural secrets (hermes dc6149b22d, f2a510b259, 2d54c978f7, a9d55a3603)

@pytest.mark.parametrize("text, secret, kind", [
    ("git remote add o https://bob:hunter2pass@github.com/x.git", "hunter2pass", "url_userinfo"),
    ("DSN postgresql://app:s3cretPw9@db:5432/prod", "s3cretPw9", "url_userinfo"),
    ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpwYXNzd29yZA==", "auth_header"),
    ("Proxy-Authorization: Digest abcdef0123456789", "abcdef0123456789", "auth_header"),
    ("Cookie: session=9f8e7d6c5b4a; theme=dark", "9f8e7d6c5b4a", "cookie"),
    ("Server=db;Database=x;Password=Tr0ub4dor&3;", "Tr0ub4dor&3", "connection_string"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----",
     "MIIEow", "private_key"),
    ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnop", "auth_header"),
    ("head -c 600 id_rsa\n-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAx",
     "MIIEpAIBAAKCAQEAx", "private_key"),
    ("postgres://user:p@ss@host/db", "ss@host", "url_userinfo"),
    ("Authorization: Bearer abc123", "abc123", "auth_header"),
    ("authorization: bearer 9f8e7d6c5b4a3210", "9f8e7d6c5b4a3210", "auth_header"),
    ("Authorization: token abc123def456", "abc123def456", "auth_header"),
    # hermes 4c7af743b1, fb6f8bc91c
    ("mcp https://mcp.example.com/sse?page=2&userToken=Zq9xLm2Pt", "Zq9xLm2Pt", "url_query"),
    ("GET /v1/x?apikey=abc123def&y=1", "abc123def", "url_query"),
    ("GET /v1/x?token=ABCDEF123&y=1", "ABCDEF123", "url_query"),  # caps with digits
    ("https://hooks.slack.com/services/T0001/B0001/XyZ123abcDEF456ghi", "XyZ123abcDEF456ghi",
     "webhook_url"),
    ("https://discord.com/api/webhooks/123456789/AbCdEfGh-IjKlMn", "AbCdEfGh-IjKlMn",
     "webhook_url"),
    ("https://acme.webhook.office.com/webhookb2/a1b2c3d4@e5f6/IncomingWebhook/9z8y",
     "a1b2c3d4@e5f6", "webhook_url"),
    # hermes 5b3168ea01, 68e81e4f8d, 03593a3538, fd0655ab84
    ("postgresql://db/app?user=app&pwd=Zq7xS3cr3t", "Zq7xS3cr3t", "url_query"),
    ("redis://h/0?sslpassword=Zq7xS3cr3t", "Zq7xS3cr3t", "url_query"),
    ("auth ya29.a0AfH6SMBc1d2e3f4g5h6i7j8k9l0", "SMBc1d2e3f4g5h6", "google_oauth"),
    ("client GOCSPX-abcdefghijk1", "GOCSPX-abcdefghijk1", "google_client_secret"),
    ("refresh 1//0gAbCdEfGhIjKlMnOpQrStUv", "AbCdEfGhIjKlMnOp", "google_refresh"),
    ("machine api.x.io login bob password hunter2Zq", "hunter2Zq", "netrc"),
    ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG", "XUtnFEMI/K7M", "aws_credentials"),
    ("token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dozjgNryP4J3jVmNHl0w5N_X", "eyJzdWIi",
     "jwt"),
])
def test_structural_secret_is_masked(text, secret, kind):
    r = redact(text)
    assert secret not in r.text and kind in r.kinds and r.count == 1
    if "p@ss" in text:  # the password runs to the last "@" before the host
        assert r.text.endswith("@host/db") and "@ss" not in r.text


@pytest.mark.parametrize("text", [
    "Send the Authorization: Basic auth header.",
    "Authorization: token required for this endpoint",
    'curl -H "Authorization: Bearer $API_TOKEN"',
    "See https://example.com/a:b and user@example.com",
    "the Cookie: header carries state",
    "password = get_password()",
    "https://example.com/search?q=token&sort=asc&keyword=api",
    "https://discord.com/channels/123/456",
    # Template placeholders in skills are not secrets.
    'curl "https://api.x.io/v1?api_key=$API_KEY&q=1"',
    "https://api.x.io/v1?token=${TOKEN}&key={key}&sig=<signature>&auth=%s",
    "https://api.x.io/v1?api_key=YOUR_API_KEY",
    "REDACT_SECRETS=false GITHUB_TOKEN=none AUTH_ENABLED='on' API_KEY=null",
    "enter login and password then click",
    "credential_process = /usr/local/bin/aws-vault exec prod",
    "token = os.environ['GITHUB_TOKEN']",
    "postgresql://db/app?pwd=${PG_PASSWORD}",
    "http://localhost:3000?email=foo@bar.com",
    "http://localhost:3000#user@example.com",
])
def test_structural_prose_is_untouched(text):
    assert redact(text).text == text


@pytest.mark.parametrize("text", [
    "Authorization: required for access",
    "a-" * 50_000,  # an unbounded URL-scheme regex took 14 s here
    "a." * 50_000,
    "?token" * 10_000,
    "https://hooks.slack.com/services/" * 2_000,
    "eyJ." * 20_000,
    "machine x " * 10_000,
    "aws_secret_access_key " * 5_000,
])
def test_no_false_positive_or_slow_scan(text):
    import time

    start = time.perf_counter()
    assert redact(text).text == text
    assert time.perf_counter() - start < 0.5


@pytest.mark.parametrize("text", ["Cookie: TOKEN=abc123", "Server=x;PASSWORD=hunter2;"])
def test_one_secret_counts_once(text):
    assert redact(text).count == 1


def test_scoped_npm_url_is_untouched():
    text = "https://registry.npmjs.org:443/@scope/pkg"
    assert redact(text).text == text


def test_a_boolean_lookalike_secret_is_still_masked():
    assert "falsehood9" not in redact("API_KEY=falsehood9").text


def test_truncated_private_key_keeps_the_prose_after_it():
    r = redact("-----BEGIN RSA PRIVATE KEY-----\nMIIEabc\n\nThe deploy key lives in vault")
    assert "MIIEabc" not in r.text and r.text.endswith("\n\nThe deploy key lives in vault")


@pytest.mark.parametrize("text, kept", [
    # head of a key file: the base64 lines go, the command output after stays
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAxyz12abcd\nAAAAB3NzaC1yc2EAAAabcd\n"
     "done\nnext step", "\ndone\nnext step"),
    # tail of a key file: walks back from the lone END, short "=" line included
    ("tail id\nMIIEpAIBAAKCAQEAxyz12abcd\nAAAAB3NzaC1yc2EAAAabcd\nQFBgc=\n"
     "-----END OPENSSH PRIVATE KEY-----\nok", "tail id\n"),
    # truncated last line with no digit/+/ after a full 64-char line
    ("-----BEGIN PRIVATE KEY-----\n"
     "AAAAB3NzaC1yc2EAAAADAQABAAABAQC7xyz0123456789abcdefghijklmnopqrstu\n"
     "QFBgcWxyzAbcdEfgh\nDone", "\nDone"),
    # a hex digest is not key body
    ("sha deadbeefdeadbeefdeadbeef0123\n-----END RSA PRIVATE KEY-----",
     "sha deadbeefdeadbeefdeadbeef0123\n"),
])
def test_partial_private_key_prints(text, kept):
    r = redact(text)
    assert kept in r.text and "AAAAB3Nz" not in r.text and "QFBgc" not in r.text
    assert r.kinds == ("private_key",) and r.count == 1


def test_many_lone_begins_are_linear():
    import time

    start = time.perf_counter()
    redact("-----BEGIN RSA PRIVATE KEY-----\n" * 3000)
    assert time.perf_counter() - start < 0.5


@pytest.mark.parametrize("text", [
    'if "-----BEGIN PRIVATE KEY-----" in text:',
    "the -----END PRIVATE KEY----- marker",
    "-----BEGIN PRIVATE KEY-----\nExample",
    "ImplementationDetails\n-----END PRIVATE KEY-----",
])
def test_lone_armor_keeps_the_text_around_it(text):
    r = redact(text)
    kept = [w for w in ("if", "in text:", "marker", "Example", "ImplementationDetails") if w in text]
    assert all(w in r.text for w in kept) and "[REDACTED PRIVATE KEY]" in r.text


def test_crlf_partial_key_body_is_masked():
    r = redact("-----BEGIN PRIVATE KEY-----\r\nMIIEvQIBADANBgkqhkiG9w0B\r\nQWxwaGFCZXRhR2FtbWEx\r\nok\r\n")
    assert "MIIEvQ" not in r.text and "QWxwaGFC" not in r.text and r.text.endswith("\r\nok\r\n")


@pytest.mark.parametrize("text", [
    "default password is changeme",
    "aws_secret_access_key = <your-key>",
    "aws_secret_access_key = YOUR_SECRET_KEY",
])
def test_credential_file_lookalikes_are_untouched(text):
    assert redact(text).text == text


_B64 = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7"


@pytest.mark.parametrize("text", [
    # JSON string: the newlines are literal "\n" escapes
    '{"private_key": "-----BEGIN PRIVATE KEY-----\\n%s\\n%s\\n"}' % (_B64, _B64),
    '{"k": "%s\\n%s\\n-----END PRIVATE KEY-----\\n"}' % (_B64, _B64),
    # encrypted PEM: headers and a blank line before the body
    "-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\n"
    "DEK-Info: AES-128-CBC,ABCDEF0123456789\n\n%s\n%s\ndone" % (_B64, _B64),
])
def test_partial_key_body_never_leaks(text):
    assert "MIIEvQ" not in redact(text).text


def test_multi_line_netrc_password_is_masked():
    r = redact("machine example.com\n  login bob\n  password s3cretZq\n")
    assert "s3cretZq" not in r.text and "netrc" in r.kinds


def test_netrc_lookalike_prose_is_untouched():
    assert redact("machine learning password reset").text == "machine learning password reset"
