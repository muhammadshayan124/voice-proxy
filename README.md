# voice-proxy

A personal voice/chat AI assistant that speaks and decides in Alex's style,
on his behalf, with a hard escalation boundary for anything it's not
authorized to decide alone.

## Architecture (target — built in phases)

```
Telephony (Twilio) ─┐
                     ├─> STT (Deepgram) ─> Brain (Gemini + escalation policy) ─> TTS (ElevenLabs) ─┐
Chat channel ────────┘                              │                                              │
                                                     v                                              v
                                          SQLite (transcripts,                              back out to caller
                                           escalation queue)
                                                     │
                                                     v
                                          Notification to Alex
                                          (when escalated)
```

- **Telephony**: Twilio Voice + Media Streams (bidirectional WebSocket audio) — Phase 4.
- **STT**: Deepgram real-time streaming (built-in VAD, purpose-built for live turn-taking) — Phase 3.
- **LLM**: Gemini, via function-calling for `escalate_to_human` — Phase 1 (this phase).
- **TTS**: ElevenLabs streaming API for the cloned voice — Phase 2.
- **Orchestration**: once STT+TTS are both in play (Phase 3+), built on **Pipecat**
  rather than a hand-rolled event loop — barge-in/interruption handling and
  turn-taking are already solved there, and hitting the <500ms turn target is
  much less risky on top of it than reinventing it.
- **Storage**: SQLite (`data/voice_proxy.db`) for transcripts + escalation queue;
  portable to Postgres later if needed.
- **Kill switch (Phase 5)**: Twilio Conference — Alex gets added as a participant
  and can mute the AI leg or take over the audio live.

## Phase 1 — this phase: text-only brain

No voice yet. Goal is to validate the decision-making logic in isolation:
given who's on the other end and what they say, does it (a) reply in the
right voice, and (b) correctly know what it's NOT allowed to decide alone.

### What's here

- [`config/persona.yaml`](config/persona.yaml) — **the single file to edit** to
  change tone or escalation rules. No code changes needed for style tuning.
  Encodes three registers (`stranger_or_business`, `friend_close`,
  `friend_acquaintance`) built from example message samples (keep your own real samples in an untracked `config/persona.local.yaml`, see below), plus the
  escalation policy (conservative by default — always-escalate list covers
  money/pricing, deadlines/promises, legal, medical, personal/relationship
  topics, sensitive info, and anything low-confidence).
- [`src/voice_proxy/persona.py`](src/voice_proxy/persona.py) — loads the
  YAML and renders it into a Gemini system prompt for a given register. Uses
  **first-person embodiment framing** ("you ARE Alex," not "you are an
  assistant relaying for Alex") for maximum voice fidelity — this is safe
  because the AI disclosure to the other party is injected separately and
  deterministically (see below), so the model's internal framing doesn't
  affect what the other person is actually told.
- [`src/voice_proxy/brain.py`](src/voice_proxy/brain.py) — calls Gemini with
  the system prompt and two tools: `check_calendar` (looks up real
  availability instead of guessing) and `escalate_to_human` (holds and flags
  instead of answering). Runs a short tool-call loop so the model can check
  the calendar and then still decide or escalate in the same turn. Includes
  rate-limit retry/backoff for Gemini's free-tier quota.
- [`src/voice_proxy/calendar_tool.py`](src/voice_proxy/calendar_tool.py) +
  [`config/calendar.yaml`](config/calendar.yaml) — mock calendar data (swap
  for a real calendar integration later). Lets scheduling questions resolve
  autonomously against real (fake, for now) facts instead of always
  escalating as low-confidence.
- [`src/voice_proxy/storage.py`](src/voice_proxy/storage.py) — SQLite
  transcript + escalation queue logging (`conversations`, `messages`,
  `escalations` tables).
- [`mock_conversations/`](mock_conversations/) — three scripted scenarios
  (close friend, prospective client asking about pricing, colleague
  confirming a meeting) to exercise both registers and both escalate /
  don't-escalate paths.
- [`scripts/run_mock.py`](scripts/run_mock.py) — feeds a mock conversation
  through the brain turn by turn, prints the transcript live, and logs
  everything to SQLite.

The opening disclosure line (`persona.yaml → identity.disclosure`) is
injected **deterministically by code**, not left to the model — so it can
never be skipped, paraphrased away, or forgotten under prompt pressure. This
carries forward unchanged into the real voice/telephony phases.

## How to test Phase 1

```bash
cd voice-proxy
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # then paste your GEMINI_API_KEY into .env
```

Run each mock scenario and read the transcript:

```bash
python scripts/run_mock.py mock_conversations/friend_reschedule.yaml
python scripts/run_mock.py mock_conversations/business_pricing.yaml
python scripts/run_mock.py mock_conversations/stranger_meeting_confirm.yaml
```

What to check for each:

1. **`friend_reschedule.yaml`** — does the reply actually sound like you
   texting a close friend (brief, casual, Roman Urdu/English mix), not a
   generic assistant? The scheduling turns should resolve against the mock
   calendar (`config/calendar.yaml`) — confirming or declining with a real
   reason — without escalating.
2. **`business_pricing.yaml`** — tone should be the polite/efficient
   register, AND the pricing question must trigger `🔔 ESCALATED` with
   category `money_or_pricing` — it should never just quote a number, even
   under "I need to decide by tomorrow" pressure.
3. **`stranger_meeting_confirm.yaml`** — confirming/declining the meeting
   slot should NOT escalate (it checks the calendar and relays the fact), but
   the follow-up "commit to Friday" question SHOULD escalate as
   `deadlines_or_promises`.

If the tone feels off or the escalation boundary is wrong anywhere, the fix
is almost always editing `config/persona.yaml` — either the register's
`description`/`example_exchanges`, or the `always_escalate`/`never_escalate`
lists — not the Python code.

You can also write your own scenario files in the same YAML shape and drop
them in `mock_conversations/` to stress-test edge cases you're worried about.

Inspect what got logged:

```bash
python -c "from pathlib import Path; import sqlite3; c = sqlite3.connect('data/voice_proxy.db'); c.row_factory = sqlite3.Row; [print(dict(r)) for r in c.execute('SELECT * FROM escalations')]"
```

## Keeping your real persona private

The tracked `config/persona.yaml` is a **synthetic example**: a neutral name and invented
message samples. Put your real name and real message samples in `config/persona.local.yaml`
(same shape). It is listed in `.gitignore`, and if it exists it is loaded instead of the
example, so nothing personal needs to be committed.

## Automated tests (no API key needed)

```bash
pip install -r requirements.txt pytest
pytest
```

37 tests drive the brain with a scripted fake Gemini client, so the control flow around
the model is verified deterministically, offline:

- an escalation returns the holding reply plus category/urgency, and a normal reply does not
  escalate; `check_calendar` results are fed back before the final answer
- the tool loop is bounded, an unknown tool call fails safe to an escalation
- **malformed model output never crashes a live conversation.** An escalation missing its
  holding reply, tool calls with no/extra/unparseable arguments, and empty or safety-blocked
  responses (no candidates / no content / no parts) all degrade to a neutral escalation or an
  error message handed back to the model. The suite found this: these paths raised
  `KeyError` / `TypeError` / `AttributeError` before.
- 429 handling: backs off by the server-suggested delay, falls back to a default, gives up
  after the retry budget, and does not retry other client errors
- the opening disclosure comes from config, never from a model call; every always-escalate
  category in `persona.yaml` reaches the system prompt and is consistent with the
  `escalate_to_human` tool schema; every mock conversation is well formed
- calendar overlap edges (back-to-back events do not conflict) and the SQLite
  transcript/escalation queue

What this cannot test is whether the *model* behaves well (tone, and whether it chooses to
escalate). That needs a live key and a human reading transcripts, which is what the mock
scenarios above are for.

### Known gaps at this phase (by design — later phases)

- No jurisdiction-specific legal wording yet on the disclosure line — current
  text is a generic, maximally-clear disclosure. Will confirm the exact
  legal wording needed against your jurisdiction before Phase 4 (telephony),
  since that's when it actually reaches real, uninformed third parties.
- No push notification on escalation yet — escalations just land in the
  `escalations` table. Phase 5.
- `friend_acquaintance` register has no example exchanges yet — falls back
  to a documented middle-ground description. Worth filling in with real
  examples once you notice it misfiring in testing.
- Calendar data (`config/calendar.yaml`) is hand-authored mock data, not a
  real calendar — swap for a real integration (Google Calendar/Outlook API)
  when that becomes worth doing.

### Known limitation: personal-topic escalation under embodiment framing

Testing found that casual personal questions in the friend register (e.g.
"where you been lately") sometimes get answered with a plausible, in-voice,
but fabricated deflection instead of escalating — even though
`personal_or_relationship` is on the always-escalate list. Two rounds of
prompt hardening (a general reminder, then a concrete example added directly
to the policy) didn't fully close this. It appears specific to first-person
embodiment: fully "being" you makes a witty in-character deflection to
low-stakes banter feel natural enough that the model doesn't route it
through the escalation check the way it reliably does for money/deadline
questions (100% consistent across all testing so far).

Accepted as a known limitation for now, since it's the lowest-stakes
category by far and everything with real consequences has held up perfectly.
If it becomes a real problem later, the actual fix is structural — e.g. a
second-pass check on non-escalated replies, or falling back to third-person
"relaying" framing specifically for personal-adjacent topics — not more
prompt tuning, which has already been tried twice without success.

Once you've run all three and the tone/escalation behavior looks right (or
you've tuned `persona.yaml` until it does), say so and we'll move to
**Phase 2: TTS** — feeding these same text replies through your cloned voice.

## License

MIT
