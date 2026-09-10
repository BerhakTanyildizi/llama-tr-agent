# -*- coding: utf-8 -*-
"""Keyword search over the user's own notes.

The cheapest retrieval that could work, deliberately. A few hundred personal
notes do not need embeddings, and an index that can go stale is a liability
before it is a feature. If keyword matching loses to semantic search here, it
should lose in a measurement rather than in advance.

ONE SHOT, NO FOLLOW-UP. Measured over the training data: of the 8144 assistant
turns that follow an observation, none is a tool call (CLAUDE.md item 24). The
model cannot search, look at the hits and then read the promising one, so a
separate `read_note` tool would be dead on arrival. This one returns a passage
wide enough to answer from directly.

There is no path parameter, so there is nothing to traverse: the model supplies
words, never a location.
"""
import re
from pathlib import Path

from .. import config

SCHEMA = {"type": "function", "function": {"name": "search_notes", "description": "Search the notes the user keeps on this machine and return the matching passages. Use it for anything the user wrote down themselves: their decisions, plans, meeting notes, drafts. For public information use google_search instead.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Words to look for, in the language the notes are written in."}}, "required": ["query"]}}}

# Like google_search's query: the model composes it from the question rather
# than quoting the user, so grounding it would reject valid searches.
GROUNDED_PARAMS: frozenset[str] = frozenset()

NOTES_DIR = Path(config.get("AGENT_NOTES_DIR") or "~/Notes").expanduser()
SUFFIXES = {".md", ".markdown", ".txt"}
MAX_FILES = 500
MAX_FILE_BYTES = 400_000
MAX_RESULTS = 3
WINDOW = 600             # characters returned around the match
MAX_TOTAL_CHARS = 3000   # the context costs ~6 s per 1000 tokens on every later turn

# Same folding as the orchestrator's grounding check, and for the same reason:
# a note saying "Elazığ" must be found by someone typing "elazigda".
_FOLD = str.maketrans("çğıİöşüÇĞÖŞÜ", "cgiiosucgosu")


def _fold(text: str) -> str:
    return text.translate(_FOLD).lower().replace("̇", "")


def _passage(text: str, folded: str, terms: list[str]) -> str:
    """A window around the first match, cut back to whitespace at both ends."""
    hits = [p for p in (folded.find(t) for t in terms) if p >= 0]
    start = max(0, min(hits) - WINDOW // 3) if hits else 0
    chunk = text[start:start + WINDOW]
    if start:
        chunk = chunk.partition(" ")[2]
    return " ".join(chunk.split())


def run(query: str) -> dict:
    terms = sorted(set(re.findall(r"\w{2,}", _fold(str(query)))))
    if not terms:
        return {"error": "empty_query",
                "message": f"'{query}' has no searchable words in it."}
    if not NOTES_DIR.is_dir():
        # Named explicitly: a missing folder is a setup problem, and the model
        # reporting it as "no notes found" would send the user looking for the
        # wrong thing.
        return {"error": "no_notes_directory",
                "message": (f"There is no notes folder at {NOTES_DIR}, so nothing could be "
                            "searched. Tell the user the folder is missing and that "
                            "AGENT_NOTES_DIR in .env points the tool elsewhere. Do NOT "
                            "invent notes and do NOT answer from prior knowledge.")}

    scored = []
    files = [p for p in sorted(NOTES_DIR.rglob("*"))
            if p.suffix.lower() in SUFFIXES and p.is_file()][:MAX_FILES]
    for path in files:
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue          # unreadable file must not take the search down
        folded = _fold(text)
        present = [t for t in terms if t in folded]
        if not present:
            continue
        # Distinct terms first, then how often they occur: a note mentioning
        # every word once beats one that repeats a single word.
        scored.append((len(present), sum(folded.count(t) for t in present), path, text, folded))

    if not scored:
        # Defect H is still open, so the emptiness is stated rather than left
        # for the model to fill in. Same wording that held up for google_search.
        return {"status": "no_results", "query": query, "provider": "notes",
                "results": [], "source_count": 0,
                "message": (f"Searched {len(files)} notes and none of them mention this. "
                            "Tell the user nothing was found in their notes. Do NOT answer "
                            "from prior knowledge and do NOT invent notes.")}

    scored.sort(key=lambda s: (-s[0], -s[1], str(s[2])))
    results, total = [], 0
    for _, _, path, text, folded in scored[:MAX_RESULTS]:
        passage = _passage(text, folded, terms)
        if total + len(passage) > MAX_TOTAL_CHARS:
            continue
        results.append({"note": str(path.relative_to(NOTES_DIR)), "passage": passage})
        total += len(passage)

    return {"status": "ok", "query": query, "provider": "notes",
            "results": results, "source_count": len(results),
            "searched": len(files)}
