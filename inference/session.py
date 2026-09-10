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
# A PROMPT budget, not a storage limit - every fact is re-sent on every turn.
# It is applied by for_prompt(), never by the code that writes the file. It used
# to be applied in both places and with opposite ends of the list: load() kept
# the first 20 and add() saved the last 20, so on a file with 25 hand-written
# lines a single /remember rewrote the file with 20 and destroyed six facts -
# the oldest one and everything past the cap - with nothing printed. /forget did
# the same. This file is documented as hand-editable, which makes silently
# rewriting it the one thing it must not do.
MAX_FACTS = 20
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
        """Everything on disk. The cap belongs to the prompt, not to the file."""
        if not self.path.exists():
            return []
        return [line.strip() for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def for_prompt(self) -> list[str]:
        """The newest MAX_FACTS - what the turn can afford to carry."""
        return self.load()[-MAX_FACTS:]

    def add(self, fact: str) -> tuple[list[str], str | None]:
        """Returns the facts and why nothing was added, if nothing was.

        The REPL printed "remembered (4 facts)" for an empty or duplicate
        /remember too, because it only ever saw the count. Reporting a write
        that did not happen is the same class of untruth as item 22.
        """
        facts = self.load()
        fact = " ".join(fact.split())
        if not fact:
            return facts, "nothing to remember"
        if fact in facts:
            return facts, "already remembered"
        facts.append(fact)
        return self._save(facts), None

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
