# -*- coding: utf-8 -*-
"""Conversation history that outlives the process.

Only the REPL uses this. eval and the tests drive Orchestrator directly and
must keep writing nothing, so the calls live in main() rather than in answer().

WHAT COMES BACK IS NOT WHAT WENT IN. The file is the archive; the slice handed
back is user turns and final answers only. Observations are left behind on
purpose - yesterday's search result costs its tokens again on every turn of
today's conversation (~6 s per 1000 at the measured 175 tok/s) and invites
answering from stale sources. Dropping them means the assistant turns that were
tool calls have to go too, or the model would see a call with no response, a
shape it never met in training.

The window is deliberately short. Defect C is a stale answer bleeding into an
unrelated turn, and restoring history is exactly the condition that feeds it.
"""
import json
import os
from pathlib import Path

DIR = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "myagent"
FILE = DIR / "session.jsonl"
PROFILE_FILE = DIR / "profile.txt"
RESTORE_MESSAGES = 8            # four exchanges
MAX_FACTS = 20                  # every fact is re-sent on every turn, so it is capped
KEEP = ("role", "content", "tool")


class Session:
    def __init__(self, path: Path = FILE, enabled: bool = True):
        self.path = path
        self.enabled = enabled

    def load(self, limit: int = RESTORE_MESSAGES) -> list[dict]:
        if not self.enabled or not self.path.exists():
            return []
        messages: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue        # a half-written line must not cost the session
            role, content = m.get("role"), m.get("content", "")
            if role == "_reset":
                messages.clear()
            elif role == "user" or (role == "assistant" and "<tool_call>" not in content):
                messages.append({"role": role, "content": content})
        window = messages[-limit:]
        # A window that opens on an answer reads as the model talking to itself.
        while window and window[0]["role"] != "user":
            window.pop(0)
        return window

    def append(self, messages: list[dict]) -> None:
        if not self.enabled or not messages:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps({k: v for k, v in m.items() if k in KEEP},
                                ensure_ascii=False) + "\n")

    def mark_reset(self) -> None:
        """/reset has to survive a restart, or it only cleared the screen."""
        self.append([{"role": "_reset", "content": ""}])


class Profile:
    """Durable facts about the user: one per line, plain text, hand-editable.

    Written by /remember, never by the model. The model over-triggers tools
    (6865 calls against 97 non-calls in training, item 7), so a `remember` tool
    would fire constantly; and it still fabricates on empty results (defect H),
    which here would mean a false fact poisoning EVERY later turn rather than
    one answer. Whoever decides what is durable has to be reliable, so it is
    the user.
    """

    def __init__(self, path: Path = PROFILE_FILE):
        self.path = path

    def load(self) -> list[str]:
        if not self.path.exists():
            return []
        return [line.strip() for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()][:MAX_FACTS]

    def add(self, fact: str) -> list[str]:
        facts = self.load()
        fact = " ".join(fact.split())
        if fact and fact not in facts:
            facts.append(fact)
        return self._save(facts[-MAX_FACTS:])

    def remove(self, index: int) -> str | None:
        """1-based, matching what /memory prints."""
        facts = self.load()
        if not 1 <= index <= len(facts):
            return None
        dropped = facts.pop(index - 1)
        self._save(facts)
        return dropped

    def _save(self, facts: list[str]) -> list[str]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(facts) + "\n" if facts else "", encoding="utf-8")
        return facts
