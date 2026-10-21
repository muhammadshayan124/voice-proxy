"""Shared fixtures: a scripted fake Gemini client, so the brain's control flow can be
tested deterministically with no network and no API key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from voice_proxy.brain import Brain
from voice_proxy.calendar_tool import Calendar
from voice_proxy.persona import Persona


def text_response(text: str):
    part = SimpleNamespace(text=text, function_call=None)
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[part]))])


def call_response(name: str, args):
    call = SimpleNamespace(name=name, args=args)
    part = SimpleNamespace(text=None, function_call=call)
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[part]))])


class ScriptedClient:
    """Returns queued responses in order; records the contents of each request.

    An item that is an Exception is raised instead of returned, to simulate API errors.
    """

    def __init__(self, *scripted):
        self.queue = list(scripted)
        self.requests: list[list] = []
        self.models = self  # client.models.generate_content(...)

    def generate_content(self, *, model, contents, config):
        self.requests.append(list(contents))
        if not self.queue:
            raise AssertionError("brain made more model calls than the test scripted")
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def calendar():
    return Calendar(
        timezone="UTC",
        events=[
            {"date": "2030-01-02", "start": "15:00", "end": "16:00", "title": "Team sync"},
            {"date": "2030-01-02", "start": "19:00", "end": "22:00", "title": "Family dinner"},
        ],
    )


@pytest.fixture
def make_brain(calendar):
    def _make(*scripted):
        client = ScriptedClient(*scripted)
        return Brain(persona=Persona.load(), calendar=calendar, client=client), client

    return _make
