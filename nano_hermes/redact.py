"""Regex-based secret redaction for content the agent persists.

Applied at write-time to skill bodies, companion files, memory entries,
and reflections. Unlike hermes-agent which redacts log lines, we redact
the data that lands on disk — once a secret enters MEMORY.md or a
SKILL.md it re-enters context every session, so redaction at the boundary
is the only place that prevents a persistent leak.

Masking: short tokens (≤18 chars) → fully masked. Longer tokens → keep
the first 6 and last 4 characters for debuggability ('sk-ant…████…XYZ').
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Provider key prefixes — adapted from hermes-agent/agent/redact.py.
_PREFIX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_or_anthropic", re.compile(r"sk-[A-Za-z0-9_-]{10,}")),
    ("github_pat_classic",  re.compile(r"ghp_[A-Za-z0-9]{10,}")),
    ("github_pat_fine",     re.compile(r"github_pat_[A-Za-z0-9_]{10,}")),
    ("github_oauth",        re.compile(r"gho_[A-Za-z0-9]{10,}")),
    ("github_user",         re.compile(r"ghu_[A-Za-z0-9]{10,}")),
    ("github_server",       re.compile(r"ghs_[A-Za-z0-9]{10,}")),
    ("github_refresh",      re.compile(r"ghr_[A-Za-z0-9]{10,}")),
    ("slack",               re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("google_api_key",      re.compile(r"AIza[A-Za-z0-9_-]{30,}")),
    ("perplexity",          re.compile(r"pplx-[A-Za-z0-9]{10,}")),
    ("aws_access_key_id",   re.compile(r"AKIA[A-Z0-9]{16}")),
    ("stripe_live",         re.compile(r"sk_live_[A-Za-z0-9]{10,}")),
    ("stripe_test",         re.compile(r"sk_test_[A-Za-z0-9]{10,}")),
    ("stripe_restricted",   re.compile(r"rk_live_[A-Za-z0-9]{10,}")),
    ("sendgrid",            re.compile(r"SG\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("huggingface",         re.compile(r"hf_[A-Za-z0-9]{10,}")),
    ("replicate",           re.compile(r"r8_[A-Za-z0-9]{10,}")),
    ("npm",                 re.compile(r"npm_[A-Za-z0-9]{10,}")),
    ("pypi",                re.compile(r"pypi-[A-Za-z0-9_-]{10,}")),
    ("digitalocean",        re.compile(r"dop_v1_[A-Za-z0-9]{10,}")),
    ("groq",                re.compile(r"gsk_[A-Za-z0-9]{10,}")),
    ("tavily",              re.compile(r"tvly-[A-Za-z0-9]{10,}")),
    ("exa",                 re.compile(r"exa_[A-Za-z0-9]{10,}")),
    ("firecrawl",           re.compile(r"fc-[A-Za-z0-9]{10,}")),
    ("browserbase",         re.compile(r"bb_live_[A-Za-z0-9_-]{10,}")),
    ("fal_ai",              re.compile(r"fal_[A-Za-z0-9_-]{10,}")),
    ("matrix",              re.compile(r"syt_[A-Za-z0-9]{10,}")),
    ("jwt",                 re.compile(r"eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_=-]{4,}){0,2}")),
]

# KEY=value with secret-like name, optionally quoted.
_SECRET_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
# A boolean/null value is config, not a secret (hermes 4f8bb8eb00): masking
# REDACT_SECRETS=false taught skills a corrupted line.
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z0-9_]{{0,50}}{_SECRET_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)"
    r"(?!(?i:true|false|yes|no|on|off|none|null)\2(?![^\s'\"]))([^\s'\"]+)\2",
)

# JSON {"api_key": "value"} (and variants).
_JSON_KEY_NAMES = (
    r"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|"
    r"auth_token|bearer|secret_value)"
)
_JSON_FIELD_RE = re.compile(
    rf'("{_JSON_KEY_NAMES}")(\s*:\s*)"([^"]+)"',
    re.IGNORECASE,
)

# Authorization / Proxy-Authorization (hermes f2a510b259). Schemes match
# case-sensitively so prose ("Authorization: token required") isn't a header:
# after Bearer any credential; after another scheme 8+ chars; bare, 8+ with a
# digit or one of = _ - . / + ("Authorization: required", "... Basic auth").
_AUTH_HEADER_RE = re.compile(
    r"((?i:(?:Proxy-)?Authorization):\s*"
    r"(?:Bearer\s+|(?:Basic|Digest|Token|Bot|Negotiate|NTLM|ApiKey)\s+(?=[^\s\"']{8})"
    r"|(?=[^\s\"']{8})(?=[^\s\"']*[\d=_\-./+])))"
    r"([^\s\"']+)",
)

# Structural secrets masked in group 2 (hermes dc6149b22d, 2d54c978f7,
# a9d55a3603): URL userinfo password (any scheme, never across whitespace),
# a Cookie header's value (needs a name=value), an ADO-style "; Password=".
_STRUCTURAL = [
    # Scheme bounded and not preceded by a scheme char: an unbounded scheme
    # restarted at every "-" of a long run — "a-" * 50_000 took 14 s.
    ("url_userinfo", re.compile(
        # The password may hold "@": it runs to the last "@" before the host.
        r"((?<![a-z0-9+.-])[a-z][a-z0-9+.-]{0,30}://[^:\s/@]{0,256}:)"
        r"([^\s/]{1,256}?)(?=@[^@\s/]+(?:[/\s:?#]|$))", re.I
    )),
    ("cookie", re.compile(r"((?:Set-)?Cookie:[ \t]*)([^\s=;]+=[^\r\n]+)", re.I)),
    ("connection_string", re.compile(r"(;\s*(?:Password|Pwd)\s*=\s*)([^;\s\"']+)", re.I)),
    # Credential-named query params (hermes 4c7af743b1): ?token=, &sig=, …
    ("url_query", re.compile(
        r"([?&](?:access_token|refresh_token|id_token|token|api_?key|apikey|client_secret|"
        r"password|auth|jwt|secret|key|signature|sig|x-amz-signature|x-goog-signature|"
        r"usertoken)=)"
        # A template placeholder ($API_KEY, {key}, %s, <token>, YOUR_KEY) is
        # not a secret; masking it breaks the skill that carries it.
        r"(?![$%{<])(?!(?-i:[A-Z][A-Z_]*)(?:[&#\s\"'<>]|$))([^&#\s\"'<>]{1,512})", re.I
    )),
    # Webhook URLs whose path is the credential (hermes fb6f8bc91c).
    ("webhook_url", re.compile(
        r"(https?://(?:hooks\.slack(?:-gov)?\.com/(?:services|workflows|triggers)/"
        r"|(?:canary\.|ptb\.|www\.)?discord(?:app)?\.com/api/(?:v\d+/)?webhooks/"
        r"|[A-Za-z0-9-]{1,63}\.webhook\.office\.com/webhookb2/))"
        r"([\w@/%+=-]{8,512})"
    )),
]
# A truncated block (``head -c 600 id_rsa``) has no END: mask its base64
# lines only, so prose after it survives.
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----"
    r"(?:[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----|(?:\r?\n[A-Za-z0-9+/=]+(?=\r?\n|\Z))*)"
)

# Telegram bot token: optional 'bot' prefix + numeric id + ':' + 30+ char token.
_TELEGRAM_RE = re.compile(r"\b(?:bot)?\d{6,12}:[A-Za-z0-9_-]{30,}\b")


@dataclass(frozen=True)
class RedactionResult:
    text: str               # the redacted text (== input if count == 0)
    count: int              # number of distinct secret-shaped strings masked
    kinds: tuple[str, ...]  # unique pattern ids that fired (sorted, deduped)


def _mask(token: str) -> str:
    """Mask *token*. Short tokens get fully masked; long ones preserve
    the first 6 and last 4 chars for debuggability.
    """
    if len(token) <= 18:
        return "█" * max(8, len(token))
    return f"{token[:6]}…{'█' * 8}…{token[-4:]}"


def redact(text: str) -> RedactionResult:
    """Mask provider keys, env-assigned secrets, JSON secret fields, bearer
    tokens, and Telegram bot tokens in *text*. Returns the redacted text
    plus a count and a sorted tuple of pattern ids that matched.

    Pattern order matters: structural patterns (env_assignment, json_field,
    auth_header, telegram) run BEFORE provider-prefix patterns. Once a
    structural pattern masks a value the secret is replaced by `█` chars,
    which can't match any provider-prefix regex — so each secret counts
    exactly once. If the prefix pass ran first, `OPENAI_API_KEY=sk-...`
    would double-count (once as the provider key, again as the env value).
    """
    if not text:
        return RedactionResult(text=text, count=0, kinds=())

    count = 0
    kinds: set[str] = set()

    # 1. ENV assignment: KEY=value (or KEY="value").
    def _env_sub(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        kinds.add("env_assignment")
        return f"{m.group(1)}={m.group(2)}{_mask(m.group(3))}{m.group(2)}"
    text = _ENV_ASSIGN_RE.sub(_env_sub, text)

    # 2. JSON field: "api_key": "value".
    def _json_sub(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        kinds.add("json_field")
        return f'{m.group(1)}{m.group(2)}"{_mask(m.group(3))}"'
    text = _JSON_FIELD_RE.sub(_json_sub, text)

    # 3. Authorization: Bearer <token>.
    def _auth_sub(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        kinds.add("auth_header")
        return f"{m.group(1)}{_mask(m.group(2))}"
    text = _AUTH_HEADER_RE.sub(_auth_sub, text)

    # 3b. Private-key blocks, URL userinfo, cookies, connection strings.
    def _pem_sub(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        kinds.add("private_key")
        return "[REDACTED PRIVATE KEY]"
    text = _PRIVATE_KEY_RE.sub(_pem_sub, text)
    for kind, pattern in _STRUCTURAL:
        def _struct_sub(m: re.Match[str], _kind: str = kind) -> str:
            nonlocal count
            if "█" in m.group(2):  # already masked by an earlier pass
                return m.group(0)
            count += 1
            kinds.add(_kind)
            return f"{m.group(1)}{_mask(m.group(2))}"
        text = pattern.sub(_struct_sub, text)

    # 4. Telegram bot tokens.
    def _tg_sub(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        kinds.add("telegram_bot_token")
        return _mask(m.group(0))
    text = _TELEGRAM_RE.sub(_tg_sub, text)

    # 5. Provider prefixes. Runs LAST so it only matches secrets that weren't
    # already redacted by a structural pattern (the mask chars don't match
    # any prefix regex).
    for kind, pattern in _PREFIX_PATTERNS:
        def _sub(m: re.Match[str], _kind: str = kind) -> str:
            nonlocal count
            count += 1
            kinds.add(_kind)
            return _mask(m.group(0))
        text = pattern.sub(_sub, text)

    return RedactionResult(text=text, count=count, kinds=tuple(sorted(kinds)))


def format_redaction_note(result: RedactionResult) -> str:
    """Human-friendly suffix for tool success messages, or '' if nothing
    was redacted. Keep this format stable — agent output learns from it.
    """
    if result.count == 0:
        return ""
    plural = "s" if result.count != 1 else ""
    kinds_str = ", ".join(result.kinds) if result.kinds else "unclassified"
    return f" (redacted {result.count} secret-shaped string{plural}: {kinds_str})"
