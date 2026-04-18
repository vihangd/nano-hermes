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
