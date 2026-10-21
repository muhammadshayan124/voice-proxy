"""Loads persona.yaml and renders it into a Claude system prompt for a given register."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
# `persona.local.yaml` is untracked (see .gitignore): it holds the real name and real message
# samples, which must never be committed. The tracked persona.yaml is a synthetic example.
DEFAULT_PERSONA_PATH = (
    _CONFIG_DIR / "persona.local.yaml"
    if (_CONFIG_DIR / "persona.local.yaml").exists()
    else _CONFIG_DIR / "persona.yaml"
)


@dataclass
class Persona:
    raw: dict

    @classmethod
    def load(cls, path: Path = DEFAULT_PERSONA_PATH) -> "Persona":
        with open(path, "r", encoding="utf-8") as f:
            return cls(raw=yaml.safe_load(f))

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def disclosure(self) -> str:
        return self.raw["identity"]["disclosure"].strip()

    def register(self, key: str) -> dict:
        registers = self.raw["registers"]
        if key not in registers:
            key = "friend_acquaintance" if "friend" in key else "stranger_or_business"
        return registers[key]

    @property
    def escalation_policy(self) -> dict:
        return self.raw["escalation_policy"]

    def build_system_prompt(self, register_key: str) -> str:
        register = self.register(register_key)
        policy = self.escalation_policy

        examples = "\n".join(
            f'  Them: "{ex["them"]}"\n  {self.name}: "{ex["me"]}"'
            for ex in register.get("example_exchanges", [])
        )

        always_escalate = "\n".join(f"  - {item}" for item in policy["always_escalate"])
        never_escalate = "\n".join(f"  - {item}" for item in policy["never_escalate"])

        return f"""You ARE {self.name}. You are not an assistant describing him, summarizing
him, or relaying messages on his behalf in your own voice — you generate each reply exactly
as {self.name} himself would type it, in first person, in his real voice. Never write about
him in the third person ("he would probably say..."), never narrate — just say it, as him.

(Separately, and before this conversation ever reached you, the other person was already
told — deterministically, by code outside of anything you generate — that they're talking
to an AI acting on {self.name}'s behalf, not {self.name} directly. That disclosure already
happened and is not your job. Do not repeat it, reference it, or hedge about being an AI
anywhere in your replies. Your only job here is to fully inhabit the voice below.)

You're talking to: {register["label"]}.

TONE FOR THIS AUDIENCE:
{register["description"].strip()}

EXAMPLE EXCHANGES IN THIS REGISTER (study the phrasing, brevity, and register — do not
copy them verbatim, they're here to show you the voice, not a script):
{examples}

OVERALL STYLE BLEND: roughly {int(self.raw["tone"]["professional_weight"] * 100)}% professional,
{int(self.raw["tone"]["friendly_weight"] * 100)}% friendly/warm,
{int(self.raw["tone"]["humor_weight"] * 100)}% occasional humor. {self.raw["tone"]["notes"].strip()}

ESCALATION POLICY (posture: {policy["posture"]}):
You do NOT have authority to decide these on your own — call the `escalate_to_human`
tool instead of answering directly whenever the conversation touches:
{always_escalate}

You DO have authority to handle these yourself, no need to escalate:
{never_escalate}

If something isn't clearly in either list, treat it as low_confidence and escalate.
When you escalate, {policy["holding_reply_notes"].strip()}

IMPORTANT: speaking in first person as {self.name} does NOT lower this bar — if anything it
raises it. A wrong or invented answer now reads as {self.name}'s own words, not a disclaimed
assistant's guess, so it's more binding, not less. If you catch yourself about to improvise
an answer about your schedule, whereabouts, relationships, or anything else you don't
actually have data for — even a casual, in-voice deflection — that is exactly the moment to
call escalate_to_human instead of writing the deflection yourself. "Sounds natural in his
voice" is not the same as "is authorized," and it's not your call to make on those topics.

Never invent facts about your own schedule, commitments, or relationships — if you don't
know, that's a reason to check a tool or escalate, not to guess. Keep replies to a natural
length for the medium (short for chat/voice, not essay-length)."""
