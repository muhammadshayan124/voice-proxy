"""The "brain": turns (contact, conversation history, their message) into either
a natural in-character reply, or an escalation (with a holding reply sent instead).

Phase 1 scope: text in, text out. No STT/TTS/telephony yet.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .calendar_tool import Calendar
from .persona import Persona

# Free-tier Gemini quota is a handful of requests/minute — retry with the
# server-suggested backoff instead of failing a whole conversation on a 429.
MAX_RATE_LIMIT_RETRIES = 4
DEFAULT_RETRY_DELAY_SECONDS = 20

# Safety cap on the check_calendar <-> model back-and-forth within one reply,
# so a confused model can't loop forever.
MAX_TOOL_ITERATIONS = 5

# NOTE: verify this against the current model list in Google AI Studio —
# naming/availability shifts over time and may have moved on since this was written.
MODEL = os.environ.get("VOICE_PROXY_MODEL", "gemini-3.5-flash-lite")

ESCALATE_FUNCTION = types.FunctionDeclaration(
    name="escalate_to_human",
    description=(
        "Flag this conversation to Alex for a decision instead of answering it "
        "yourself. Use this whenever the topic falls under an always-escalate "
        "category, or you are not confident how Alex would want to respond. "
        "Do not guess on Alex's behalf on anything you'd call here."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "category": types.Schema(
                type="STRING",
                description="Which policy category this falls under (e.g. money_or_pricing, "
                "deadlines_or_promises, legal_or_contractual, medical_or_health, "
                "personal_or_relationship, sensitive_personal_info, low_confidence).",
            ),
            "reason": types.Schema(
                type="STRING",
                description="One or two sentences: why this needs Alex's input.",
            ),
            "urgency": types.Schema(type="STRING", enum=["low", "normal", "high"]),
            "holding_reply": types.Schema(
                type="STRING",
                description="A short, in-character reply to send to the other person right "
                "now, in the same register/tone as the conversation, so they aren't left hanging.",
            ),
        },
        required=["category", "reason", "urgency", "holding_reply"],
    ),
)

CHECK_CALENDAR_FUNCTION = types.FunctionDeclaration(
    name="check_calendar",
    description=(
        "The only source of truth for your schedule. Never guess or assume your "
        "availability — always call this before confirming, declining, or suggesting "
        "a time for any call/meeting/hangout. Resolve relative dates ('today', "
        "'tonight', 'tomorrow', 'kal') against the current date given in your "
        "instructions before calling this."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "date": types.Schema(type="STRING", description="ISO date, YYYY-MM-DD."),
            "start_time": types.Schema(
                type="STRING", description="Optional 24h HH:MM start time to check for a conflict."
            ),
            "end_time": types.Schema(
                type="STRING", description="Optional 24h HH:MM end time to check for a conflict."
            ),
        },
        required=["date"],
    ),
)

TOOLS = types.Tool(function_declarations=[ESCALATE_FUNCTION, CHECK_CALENDAR_FUNCTION])


def _to_gemini_contents(history: list[dict]) -> list[types.Content]:
    # Our stored history uses "user"/"assistant" (provider-neutral); Gemini wants "user"/"model".
    role_map = {"user": "user", "assistant": "model"}
    return [
        types.Content(role=role_map[turn["role"]], parts=[types.Part(text=turn["content"])])
        for turn in history
    ]


@dataclass
class Turn:
    text: str
    escalated: bool = False
    category: Optional[str] = None
    reason: Optional[str] = None
    urgency: Optional[str] = None


_CALENDAR_ARGS = ("date", "start_time", "end_time")
_FALLBACK_HOLDING_REPLY = "Let me check on that and get back to you."


def _fail_safe(reason: str) -> "Turn":
    """Anything unexpected from the model becomes an escalation with a neutral holding
    reply, so the other person is never left with silence or an exception."""
    return Turn(
        text=_FALLBACK_HOLDING_REPLY,
        escalated=True,
        category="low_confidence",
        reason=reason,
        urgency="normal",
    )


def _default_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


@dataclass
class Brain:
    persona: Persona = field(default_factory=Persona.load)
    calendar: Calendar = field(default_factory=Calendar.load)
    client: genai.Client = field(default_factory=_default_client)

    def opening_disclosure(self) -> str:
        """Deterministic — never generated by the model, so it can't be skipped."""
        return self.persona.disclosure

    def _system_prompt(self, register_key: str) -> str:
        return (
            self.persona.build_system_prompt(register_key)
            + f"\n\n{self.calendar.today_context()} Use the check_calendar tool — described "
            "in your tools, not repeated here — for anything involving your schedule; once "
            "you have a real answer from it, you have the authority to confirm an open slot "
            "or decline a conflicting one yourself, since you're relaying a fact rather than "
            "making a judgment call. Still escalate separately if the conversation also "
            "touches an always-escalate topic (e.g. a new commitment, a price)."
        )

    def _generate_with_retry(self, system_prompt: str, contents: list[types.Content]):
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[TOOLS],
            max_output_tokens=1024,
        )
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return self.client.models.generate_content(
                    model=MODEL, contents=contents, config=config
                )
            except genai_errors.ClientError as e:
                if e.code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                match = re.search(r"retry in ([\d.]+)s", str(e))
                delay = float(match.group(1)) + 1 if match else DEFAULT_RETRY_DELAY_SECONDS
                print(f"  (rate limited, waiting {delay:.0f}s before retrying...)")
                time.sleep(delay)

    def _run_calendar_tool(self, args: dict) -> dict:
        """Model-supplied arguments are untrusted: keep only the declared ones and report
        problems back to the model as data, so it can correct itself, instead of raising."""
        kwargs = {k: v for k, v in args.items() if k in _CALENDAR_ARGS}
        if not kwargs.get("date"):
            return {"error": "check_calendar requires a 'date' (YYYY-MM-DD)."}
        try:
            return self.calendar.check(**kwargs)
        except (ValueError, TypeError, AttributeError):
            return {"error": "Could not read the date or times; use 24h HH:MM and an ISO date."}

    def respond(self, register_key: str, history: list[dict], their_message: str) -> Turn:
        system_prompt = self._system_prompt(register_key)
        contents = _to_gemini_contents(history) + [
            types.Content(role="user", parts=[types.Part(text=their_message)])
        ]

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self._generate_with_retry(system_prompt, contents)
            candidates = getattr(response, "candidates", None) or []
            model_content = candidates[0].content if candidates else None
            parts = model_content.parts if model_content is not None else None
            if not parts:
                return _fail_safe("Model returned no usable content (empty or blocked response).")
            function_call = next((p.function_call for p in parts if p.function_call), None)

            if function_call is None:
                text = "".join(p.text for p in parts if p.text)
                return Turn(text=text)

            if function_call.name == "escalate_to_human":
                args = dict(function_call.args or {})
                return Turn(
                    text=(args.get("holding_reply") or "").strip() or _FALLBACK_HOLDING_REPLY,
                    escalated=True,
                    category=args.get("category") or "low_confidence",
                    reason=args.get("reason") or "Model escalated without giving a reason.",
                    urgency=args.get("urgency") or "normal",
                )

            if function_call.name == "check_calendar":
                result = self._run_calendar_tool(dict(function_call.args or {}))
                contents.append(model_content)
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name="check_calendar", response=result
                                )
                            )
                        ],
                    )
                )
                continue

            # Unknown tool call — fail safe rather than guess.
            return _fail_safe(f"Model called unrecognized tool '{function_call.name}'.")

        return _fail_safe("Exceeded tool-call iteration limit without reaching a final reply.")
