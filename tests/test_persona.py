from pathlib import Path

import pytest
import yaml

from voice_proxy import brain as brain_module
from voice_proxy.persona import Persona

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def persona():
    return Persona.load()


def test_disclosure_says_it_is_an_ai_and_not_the_person(persona):
    text = persona.disclosure.lower()
    assert "ai assistant" in text
    assert "not talking" in text and persona.name.lower() in text


def test_disclosure_is_fixed_text_independent_of_the_model(make_brain, persona):
    brain, client = make_brain()  # no scripted responses: any model call would fail
    assert brain.opening_disclosure() == persona.disclosure
    assert client.requests == []


@pytest.mark.parametrize(
    "key, expected",
    [
        ("friend_close", "friend_close"),
        ("stranger_or_business", "stranger_or_business"),
        ("friend_of_a_friend", "friend_acquaintance"),  # unknown friend-ish -> middle ground
        ("random_unknown", "stranger_or_business"),  # unknown -> the formal default
    ],
)
def test_unknown_registers_fall_back_conservatively(persona, key, expected):
    assert persona.register(key) is persona.raw["registers"][expected]


def test_every_always_escalate_category_reaches_the_system_prompt(persona):
    prompt = persona.build_system_prompt("friend_close")
    for category in persona.escalation_policy["always_escalate"]:
        assert category in prompt


def test_system_prompt_tells_the_model_not_to_redisclose_or_hedge(persona):
    prompt = persona.build_system_prompt("stranger_or_business")
    assert "Do not repeat it" in prompt
    assert "escalate_to_human" in prompt


def test_escalation_categories_in_config_match_the_tool_description():
    """The tool schema lists example categories; they should all exist in the policy, or
    the model is being told to use categories the policy does not define."""
    policy = yaml.safe_load((ROOT / "config" / "persona.yaml").read_text(encoding="utf-8"))["escalation_policy"]
    described = brain_module.ESCALATE_FUNCTION.parameters.properties["category"].description
    for category in policy["always_escalate"]:
        assert category in described


def test_conservative_defaults(persona):
    policy = persona.escalation_policy
    assert policy["posture"] == "conservative"
    assert "low_confidence" in policy["always_escalate"]


def test_every_mock_conversation_is_well_formed_and_uses_a_known_register(persona):
    files = sorted((ROOT / "mock_conversations").glob("*.yaml"))
    assert files
    for path in files:
        scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert scenario["contact"]["register"] in persona.raw["registers"], path.name
        assert scenario["turns"] and all(isinstance(t, str) for t in scenario["turns"]), path.name


def test_local_persona_override_is_gitignored_and_preferred():
    from voice_proxy import persona as persona_module

    assert "config/persona.local.yaml" in (ROOT / ".gitignore").read_text(encoding="utf-8")
    expected = ROOT / "config" / ("persona.local.yaml" if (ROOT / "config" / "persona.local.yaml").exists() else "persona.yaml")
    assert persona_module.DEFAULT_PERSONA_PATH == expected


def test_tracked_persona_contains_no_real_names():
    text = (ROOT / "config" / "persona.yaml").read_text(encoding="utf-8").lower()
    assert "shayan" not in text
