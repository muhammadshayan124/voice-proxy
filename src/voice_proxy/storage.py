"""SQLite-backed transcript and escalation-queue storage.

Kept as plain sqlite3 (no ORM) since the schema is small and stable.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "voice_proxy.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_name TEXT,
    register TEXT NOT NULL,
    channel TEXT NOT NULL,
    started_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,               -- 'them' | 'assistant' | 'system'
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    message_id INTEGER REFERENCES messages(id),
    category TEXT NOT NULL,
    urgency TEXT NOT NULL,
    reason TEXT NOT NULL,
    holding_reply TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'resolved'
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_conversation(self, contact_name: str, register: str, channel: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (contact_name, register, channel, started_at) "
                "VALUES (?, ?, ?, ?)",
                (contact_name, register, channel, _now()),
            )
            return cur.lastrowid

    def add_message(self, conversation_id: int, turn_index: int, role: str, content: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (conversation_id, turn_index, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, turn_index, role, content, _now()),
            )
            return cur.lastrowid

    def add_escalation(
        self,
        conversation_id: int,
        message_id: Optional[int],
        category: str,
        urgency: str,
        reason: str,
        holding_reply: str,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO escalations "
                "(conversation_id, message_id, category, urgency, reason, holding_reply, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                (conversation_id, message_id, category, urgency, reason, holding_reply, _now()),
            )
            return cur.lastrowid

    def get_transcript(self, conversation_id: int) -> list[sqlite3.Row]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY turn_index",
                (conversation_id,),
            ).fetchall()

    def pending_escalations(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                "SELECT * FROM escalations WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
