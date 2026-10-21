#!/usr/bin/env python
"""Run a mock conversation file through the brain and print/log the transcript.

Usage:
    python scripts/run_mock.py mock_conversations/friend_reschedule.yaml
    python scripts/run_mock.py mock_conversations/*.yaml   # (shell-expanded) run several
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Windows consoles default to a legacy codepage (cp1252) that can't encode
# emoji/em-dashes used in transcript output — force UTF-8 so this doesn't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Must run before importing voice_proxy.brain — it reads VOICE_PROXY_MODEL
# from the environment at import time, so .env has to be loaded first.
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_proxy.brain import Brain  # noqa: E402
from voice_proxy.storage import Storage  # noqa: E402


def run_one(path: Path, brain: Brain, storage: Storage) -> None:
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    contact = scenario["contact"]
    register = contact["register"]
    channel = scenario.get("channel", "chat")

    print(f"\n{'=' * 70}\nSCENARIO: {path.name}  |  contact: {contact['name']}  |  register: {register}\n{'=' * 70}")

    conv_id = storage.create_conversation(contact["name"], register, channel)

    turn_index = 0
    history: list[dict] = []

    disclosure = brain.opening_disclosure()
    storage.add_message(conv_id, turn_index, "assistant", disclosure)
    print(f"[{contact['name']}'s side sees]  Assistant: {disclosure}\n")
    turn_index += 1

    for their_message in scenario["turns"]:
        storage.add_message(conv_id, turn_index, "them", their_message)
        print(f"Them: {their_message}")
        turn_index += 1

        turn = brain.respond(register, history, their_message)
        history.append({"role": "user", "content": their_message})
        history.append({"role": "assistant", "content": turn.text})

        msg_id = storage.add_message(conv_id, turn_index, "assistant", turn.text)
        turn_index += 1

        if turn.escalated:
            storage.add_escalation(
                conv_id, msg_id, turn.category, turn.urgency, turn.reason, turn.text
            )
            print(f"Assistant: {turn.text}")
            print(f"  🔔 ESCALATED  [{turn.category} / {turn.urgency}] — {turn.reason}\n")
        else:
            print(f"Assistant: {turn.text}\n")

    print(f"(conversation_id={conv_id} logged to data/voice_proxy.db)")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    brain = Brain()
    storage = Storage()

    for arg in sys.argv[1:]:
        run_one(Path(arg), brain, storage)


if __name__ == "__main__":
    main()
