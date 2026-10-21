"""The brain's contract is "never improvise on an escalation topic, and when anything is
off, fail toward escalating". These tests drive it with a scripted model."""

from __future__ import annotations

import pytest
from google.genai import errors as genai_errors

from tests.conftest import call_response, text_response
from voice_proxy import brain as brain_module

ESCALATE_ARGS = {
    "category": "money_or_pricing",
    "reason": "They asked for a price.",
    "urgency": "high",
    "holding_reply": "Lemme check and get back to you",
}


def test_plain_reply_is_returned_unescalated(make_brain):
    brain, _ = make_brain(text_response("Okay thanks"))
    turn = brain.respond("stranger_or_business", [], "ok whatever you decide")
    assert turn.text == "Okay thanks" and turn.escalated is False


def test_escalation_returns_the_holding_reply_and_metadata(make_brain):
    brain, _ = make_brain(call_response("escalate_to_human", ESCALATE_ARGS))
    turn = brain.respond("stranger_or_business", [], "what do you charge?")
    assert turn.escalated is True
    assert turn.text == "Lemme check and get back to you"
    assert (turn.category, turn.urgency) == ("money_or_pricing", "high")


def test_calendar_result_is_fed_back_before_the_final_answer(make_brain):
    brain, client = make_brain(
        call_response("check_calendar", {"date": "2030-01-02", "start_time": "15:30", "end_time": "16:30"}),
        text_response("Can't do that slot, I have a team sync"),
    )
    turn = brain.respond("friend_close", [], "free at 15:30 on the 2nd?")
    assert turn.escalated is False and "team sync" in turn.text
    function_response = client.requests[1][-1].parts[0].function_response
    assert function_response.name == "check_calendar"
    assert function_response.response["free"] is False


def test_history_is_translated_to_gemini_roles(make_brain):
    brain, client = make_brain(text_response("hey"))
    brain.respond(
        "friend_close",
        [{"role": "user", "content": "yo"}, {"role": "assistant", "content": "hey"}],
        "sup",
    )
    assert [c.role for c in client.requests[0]] == ["user", "model", "user"]


def test_unknown_tool_fails_safe_to_escalation(make_brain):
    brain, _ = make_brain(call_response("transfer_money", {"amount": 100}))
    turn = brain.respond("stranger_or_business", [], "hi")
    assert turn.escalated is True and turn.category == "low_confidence"
    assert "transfer_money" in turn.reason


def test_tool_loop_is_bounded(make_brain):
    looping = [call_response("check_calendar", {"date": "2030-01-02"})] * brain_module.MAX_TOOL_ITERATIONS
    brain, client = make_brain(*looping)
    turn = brain.respond("friend_close", [], "when are you free?")
    assert turn.escalated is True and turn.category == "low_confidence"
    assert len(client.requests) == brain_module.MAX_TOOL_ITERATIONS


# --- fail-safe behaviour when the model's output is malformed ---------------------------

def test_escalation_missing_holding_reply_still_escalates(make_brain):
    args = {k: v for k, v in ESCALATE_ARGS.items() if k != "holding_reply"}
    brain, _ = make_brain(call_response("escalate_to_human", args))
    turn = brain.respond("stranger_or_business", [], "what do you charge?")
    assert turn.escalated is True
    assert turn.text.strip()  # the other person is never left with an empty reply
    assert turn.category == "money_or_pricing"


def test_escalation_with_no_args_at_all_still_escalates(make_brain):
    brain, _ = make_brain(call_response("escalate_to_human", None))
    turn = brain.respond("stranger_or_business", [], "hello?")
    assert turn.escalated is True and turn.text.strip()


def test_calendar_call_with_unexpected_argument_does_not_crash(make_brain):
    brain, client = make_brain(
        call_response("check_calendar", {"date": "2030-01-02", "timezone": "UTC"}),
        text_response("Looks open"),
    )
    turn = brain.respond("friend_close", [], "free on the 2nd?")
    assert turn.text == "Looks open"  # the extra argument is ignored; the lookup still ran
    assert "busy_blocks" in client.requests[1][-1].parts[0].function_response.response


def test_calendar_call_with_unparseable_time_returns_an_error_to_the_model(make_brain):
    brain, client = make_brain(
        call_response("check_calendar", {"date": "2030-01-02", "start_time": "7pm", "end_time": "8pm"}),
        text_response("Let me confirm the time"),
    )
    brain.respond("friend_close", [], "free at 7pm?")
    response = client.requests[1][-1].parts[0].function_response.response
    assert "error" in response


def test_calendar_call_without_date_returns_an_error_to_the_model(make_brain):
    brain, client = make_brain(call_response("check_calendar", {}), text_response("When exactly?"))
    brain.respond("friend_close", [], "are you free?")
    assert "error" in client.requests[1][-1].parts[0].function_response.response


@pytest.mark.parametrize("blocked", ["no_candidates", "no_content", "no_parts"])
def test_blocked_or_empty_model_response_escalates_instead_of_crashing(make_brain, blocked):
    from types import SimpleNamespace

    if blocked == "no_candidates":
        response = SimpleNamespace(candidates=[])
    elif blocked == "no_content":
        response = SimpleNamespace(candidates=[SimpleNamespace(content=None)])
    else:
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=None))])
    brain, _ = make_brain(response)
    turn = brain.respond("stranger_or_business", [], "hello")
    assert turn.escalated is True and turn.category == "low_confidence" and turn.text.strip()


# --- rate limiting ----------------------------------------------------------------------

@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(brain_module.time, "sleep", slept.append)
    return slept


def _rate_limited(message="Quota exceeded. Please retry in 7.5s"):
    return genai_errors.ClientError(429, {"error": {"message": message}}, None)


def test_rate_limit_retries_using_the_server_suggested_delay(make_brain, no_sleep):
    brain, client = make_brain(_rate_limited(), text_response("ok"))
    assert brain.respond("friend_close", [], "hi").text == "ok"
    assert no_sleep == [8.5]  # suggested 7.5s + 1s margin
    assert len(client.requests) == 2


def test_rate_limit_without_a_hint_uses_the_default_delay(make_brain, no_sleep):
    brain, _ = make_brain(_rate_limited("Too many requests"), text_response("ok"))
    brain.respond("friend_close", [], "hi")
    assert no_sleep == [brain_module.DEFAULT_RETRY_DELAY_SECONDS]


def test_rate_limit_gives_up_after_the_retry_budget(make_brain, no_sleep):
    brain, client = make_brain(*[_rate_limited() for _ in range(brain_module.MAX_RATE_LIMIT_RETRIES + 1)])
    with pytest.raises(genai_errors.ClientError):
        brain.respond("friend_close", [], "hi")
    assert len(client.requests) == brain_module.MAX_RATE_LIMIT_RETRIES + 1


def test_other_client_errors_are_not_retried(make_brain, no_sleep):
    bad_request = genai_errors.ClientError(400, {"error": {"message": "bad"}}, None)
    brain, client = make_brain(bad_request, text_response("never reached"))
    with pytest.raises(genai_errors.ClientError):
        brain.respond("friend_close", [], "hi")
    assert no_sleep == [] and len(client.requests) == 1
