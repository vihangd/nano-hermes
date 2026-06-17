"""Tests for nano_hermes/utils/error_classifier.py."""
from __future__ import annotations

import pytest

from nano_hermes.utils.error_classifier import (
    ClassifiedError,
    EvolutionAbortError,
    FailoverReason,
    classify_http_status,
    classify_llm_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Resp:
    """Minimal LLMResponse stand-in."""

    def __init__(
        self,
        finish_reason="stop",
        error_status_code=None,
        error_type="",
        error_code="",
        content="",
        error_kind="",
    ):
        self.finish_reason = finish_reason
        self.error_status_code = error_status_code
        self.error_type = error_type
        self.error_code = error_code
        self.content = content
        self.error_kind = error_kind


def _err_resp(**kwargs) -> _Resp:
    return _Resp(finish_reason="error", **kwargs)


# ---------------------------------------------------------------------------
# classify_llm_response — success cases
# ---------------------------------------------------------------------------


class TestClassifyLlmResponseSuccess:
    def test_none_returns_unknown(self):
        result = classify_llm_response(None)
        assert result is not None
        assert result.reason == FailoverReason.unknown

    def test_finish_stop_returns_none(self):
        assert classify_llm_response(_Resp(finish_reason="stop")) is None

    def test_finish_length_returns_none(self):
        assert classify_llm_response(_Resp(finish_reason="length")) is None

    def test_finish_none_treated_as_stop(self):
        assert classify_llm_response(_Resp(finish_reason=None)) is None


# ---------------------------------------------------------------------------
# classify_llm_response — error status codes
# ---------------------------------------------------------------------------


class TestClassifyLlmResponseStatusCodes:
    def test_401_maps_to_auth(self):
        r = classify_llm_response(_err_resp(error_status_code=401))
        assert r.reason == FailoverReason.auth
        assert r.status_code == 401

    def test_403_maps_to_auth(self):
        r = classify_llm_response(_err_resp(error_status_code=403))
        assert r.reason == FailoverReason.auth

    def test_402_maps_to_billing(self):
        r = classify_llm_response(_err_resp(error_status_code=402))
        assert r.reason == FailoverReason.billing

    def test_429_maps_to_rate_limit(self):
        r = classify_llm_response(_err_resp(error_status_code=429))
        assert r.reason == FailoverReason.rate_limit

    def test_503_maps_to_overloaded(self):
        r = classify_llm_response(_err_resp(error_status_code=503))
        assert r.reason == FailoverReason.overloaded

    def test_529_maps_to_overloaded(self):
        r = classify_llm_response(_err_resp(error_status_code=529))
        assert r.reason == FailoverReason.overloaded

    def test_500_maps_to_server_error(self):
        r = classify_llm_response(_err_resp(error_status_code=500))
        assert r.reason == FailoverReason.server_error

    def test_502_maps_to_server_error(self):
        r = classify_llm_response(_err_resp(error_status_code=502))
        assert r.reason == FailoverReason.server_error


# ---------------------------------------------------------------------------
# classify_llm_response — text-based signals
# ---------------------------------------------------------------------------


class TestClassifyLlmResponseTextSignals:
    def test_billing_token_in_error_code(self):
        r = classify_llm_response(_err_resp(error_code="insufficient_quota"))
        assert r.reason == FailoverReason.billing

    def test_billing_token_in_content(self):
        r = classify_llm_response(_err_resp(content="billing_hard_limit exceeded"))
        assert r.reason == FailoverReason.billing

    def test_rate_limit_in_error_type(self):
        r = classify_llm_response(_err_resp(error_type="rate_limit"))
        assert r.reason == FailoverReason.rate_limit

    def test_too_many_requests_in_content(self):
        r = classify_llm_response(_err_resp(content="too many requests from your IP"))
        assert r.reason == FailoverReason.rate_limit

    def test_overload_in_content(self):
        r = classify_llm_response(_err_resp(content="service overloaded, try later"))
        assert r.reason == FailoverReason.overloaded

    def test_context_length_exceeded_in_error_code(self):
        r = classify_llm_response(_err_resp(error_code="context_length_exceeded"))
        assert r.reason == FailoverReason.context_overflow

    def test_context_window_in_content(self):
        r = classify_llm_response(_err_resp(content="context window exceeded"))
        assert r.reason == FailoverReason.context_overflow

    def test_timeout_in_error_kind(self):
        r = classify_llm_response(_err_resp(error_kind="timeout"))
        assert r.reason == FailoverReason.timeout

    def test_timeout_in_content(self):
        r = classify_llm_response(_err_resp(content="request timeout"))
        assert r.reason == FailoverReason.timeout

    def test_unknown_fallback(self):
        r = classify_llm_response(_err_resp(error_status_code=418))
        assert r.reason == FailoverReason.unknown
        assert r.status_code == 418

    def test_content_truncated_to_200(self):
        long_msg = "x" * 500
        r = classify_llm_response(_err_resp(error_status_code=401, content=long_msg))
        assert len(r.message) == 200


# ---------------------------------------------------------------------------
# ClassifiedError properties
# ---------------------------------------------------------------------------


class TestClassifiedErrorProperties:
    def test_should_abort_billing(self):
        c = ClassifiedError(reason=FailoverReason.billing)
        assert c.should_abort is True

    def test_should_abort_auth(self):
        c = ClassifiedError(reason=FailoverReason.auth)
        assert c.should_abort is True

    def test_should_abort_rate_limit_false(self):
        c = ClassifiedError(reason=FailoverReason.rate_limit)
        assert c.should_abort is False

    def test_should_backoff_rate_limit(self):
        c = ClassifiedError(reason=FailoverReason.rate_limit)
        assert c.should_backoff is True

    def test_should_backoff_overloaded(self):
        c = ClassifiedError(reason=FailoverReason.overloaded)
        assert c.should_backoff is True

    def test_should_backoff_billing_false(self):
        c = ClassifiedError(reason=FailoverReason.billing)
        assert c.should_backoff is False

    def test_should_skip_skill_context_overflow(self):
        c = ClassifiedError(reason=FailoverReason.context_overflow)
        assert c.should_skip_skill is True

    def test_should_skip_skill_other_false(self):
        c = ClassifiedError(reason=FailoverReason.unknown)
        assert c.should_skip_skill is False


# ---------------------------------------------------------------------------
# EvolutionAbortError
# ---------------------------------------------------------------------------


class TestEvolutionAbortError:
    def test_carries_classified(self):
        c = ClassifiedError(reason=FailoverReason.billing, message="out of credits")
        exc = EvolutionAbortError(c)
        assert exc.classified is c

    def test_message_contains_reason_and_message(self):
        c = ClassifiedError(reason=FailoverReason.auth, message="bad key")
        exc = EvolutionAbortError(c)
        assert "auth" in str(exc)
        assert "bad key" in str(exc)

    def test_is_exception(self):
        c = ClassifiedError(reason=FailoverReason.billing)
        exc = EvolutionAbortError(c)
        assert isinstance(exc, Exception)

    def test_propagates_through_generic_except(self):
        c = ClassifiedError(reason=FailoverReason.billing)

        def inner():
            try:
                raise EvolutionAbortError(c)
            except EvolutionAbortError:
                raise
            except Exception:
                pass  # should not reach here

        with pytest.raises(EvolutionAbortError):
            inner()


# ---------------------------------------------------------------------------
# classify_http_status
# ---------------------------------------------------------------------------


class TestClassifyHttpStatus:
    def test_401_auth(self):
        r = classify_http_status(401)
        assert r.reason == FailoverReason.auth
        assert r.status_code == 401

    def test_403_auth(self):
        assert classify_http_status(403).reason == FailoverReason.auth

    def test_402_billing(self):
        assert classify_http_status(402).reason == FailoverReason.billing

    def test_429_rate_limit(self):
        assert classify_http_status(429).reason == FailoverReason.rate_limit

    def test_503_overloaded(self):
        assert classify_http_status(503).reason == FailoverReason.overloaded

    def test_529_overloaded(self):
        assert classify_http_status(529).reason == FailoverReason.overloaded

    def test_500_server_error(self):
        assert classify_http_status(500).reason == FailoverReason.server_error

    def test_502_server_error(self):
        assert classify_http_status(502).reason == FailoverReason.server_error

    def test_418_unknown(self):
        r = classify_http_status(418, "I'm a teapot")
        assert r.reason == FailoverReason.unknown
        assert r.message == "I'm a teapot"

    def test_message_passed_through(self):
        r = classify_http_status(401, "token expired")
        assert r.message == "token expired"
