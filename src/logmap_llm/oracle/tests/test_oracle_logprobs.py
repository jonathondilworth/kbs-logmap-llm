"""A provider that refuses logprobs must not kill the run.

OpenAI's gpt-5 reasoning family returns HTTP 400 "logprobs are not supported with reasoning
models"; on such a refusal the manager latches logprobs off and retries instead of aborting.
Dropping logprobs is safe: Oracle_confidence (derived from logprobs) does not gate retention in
the Java refiner — measured retention is ~99% on both sides of the min_conf_pro_map=0.80
threshold — and cloud oracles already return no logprobs while local vLLM models do.
"""
import pytest

from logmap_llm.oracle.manager import OracleConsultationManager as OracleManager


class _Refuses:
    """Stand-in for a provider that rejects the logprobs parameter."""
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("logprobs"):
            raise RuntimeError(
                "Error code: 400 - logprobs are not supported with reasoning models."
            )
        return "OK"


def _mgr() -> OracleManager:
    import threading
    m = OracleManager.__new__(OracleManager)   # bypass __init__ / network
    m.logprobs = True
    m.top_logprobs = 3
    m._logprobs_lock = threading.Lock()
    return m


def test_logprobs_are_sent_when_the_provider_accepts_them():
    """Providers that accept logprobs are still sent them (no run-id aliasing)."""
    m = _mgr()
    kwargs = {}
    if m.logprobs:
        kwargs["logprobs"] = m.logprobs
        kwargs["top_logprobs"] = m.top_logprobs
    assert kwargs == {"logprobs": True, "top_logprobs": 3}


def test_logprobs_are_omitted_once_disabled():
    m = _mgr()
    m.logprobs = False
    kwargs = {}
    if m.logprobs:
        kwargs["logprobs"] = m.logprobs
        kwargs["top_logprobs"] = m.top_logprobs
    assert "logprobs" not in kwargs, "a refusing provider must not be sent logprobs again"


def test_refusal_latches_off_and_retries_instead_of_aborting_the_run():
    """A logprobs 400 latches logprobs off and retries instead of burning the failure budget."""
    m = _mgr()
    client = _Refuses()

    def dispatch(msg, dev=None, fs=None):
        kw = {"model": "openai/gpt-5-nano", "messages": msg}
        if m.logprobs:
            kw["logprobs"] = m.logprobs
            kw["top_logprobs"] = m.top_logprobs
        return client(**kw)

    m._dispatch_consult = dispatch
    out = OracleManager.consult_oracle(m, [{"role": "user", "content": "x"}])

    assert out == "OK", "the retry must succeed, not propagate the 400"
    assert m.logprobs is False, "the refusal must LATCH — otherwise every call pays a failed request"
    assert len(client.calls) == 2, "exactly one retry"
    assert client.calls[0].get("logprobs") is True
    assert "logprobs" not in client.calls[1]


def test_the_LATCH_RACE_that_broke_the_first_fix():
    """Consultations run on max_workers threads against one shared manager: a request already in
    flight with logprobs=True can hit the 400 after a peer thread has cleared the latch, so the
    retry must not depend on the latch state.
    """
    m = _mgr()
    m.logprobs = False          # a peer thread already latched it off
    m.top_logprobs = None
    calls = []

    def dispatch(msg, dev=None, fs=None):
        calls.append(dict(logprobs=m.logprobs))
        if len(calls) == 1:
            # this thread's request was already in flight with logprobs, so it still gets the 400
            raise RuntimeError("Error code: 400 - logprobs are not supported with reasoning models.")
        return "OK"

    m._dispatch_consult = dispatch
    out = OracleManager.consult_oracle(m, [{"role": "user", "content": "x"}])

    assert out == "OK", (
        "a thread that arrives after the latch is already cleared MUST still retry — "
        "otherwise it re-raises and burns the OC-1 consecutive-failure budget"
    )
    assert len(calls) == 2


def test_a_non_logprobs_error_still_propagates():
    """The fallback must not swallow unrelated failures — that would hide real outages."""
    m = _mgr()

    def dispatch(msg, dev=None, fs=None):
        raise RuntimeError("Error code: 429 - rate limited")

    m._dispatch_consult = dispatch
    with pytest.raises(RuntimeError, match="429"):
        OracleManager.consult_oracle(m, [{"role": "user", "content": "x"}])
    assert m.logprobs is True, "an unrelated error must not disable logprobs"
