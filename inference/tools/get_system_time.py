# -*- coding: utf-8 -*-
"""Current date/time tool. Simplest of the three; first smoke test for the loop.

SCHEMA is copied verbatim from the training data. A single changed character
shifts the system prompt away from the training distribution.
"""
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA = {"type": "function", "function": {"name": "get_system_time", "description": "Get the current date and time, optionally for a specific timezone.", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "IANA timezone name, e.g. 'Europe/Istanbul'."}}, "required": []}}}

# Arguments the orchestrator must find grounded in the conversation before
# dispatching. Empty here: 'Seul' -> 'Asia/Seoul' is a legitimate derivation,
# not a fabrication.
GROUNDED_PARAMS: frozenset[str] = frozenset()

DEFAULT_TZ = "Europe/Istanbul"

# Written out by hand for the same reason prompts.MONTHS is (CLAUDE.md item 6):
# strftime("%A") follows the OS locale, so under a Turkish locale this field
# came back as "Pazartesi". Tool output is English by contract - the hybrid
# strategy is that the tool speaks English and the model localizes the answer -
# and a field that changes language with the machine's LC_TIME breaks it
# silently, on someone else's machine rather than this one.
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]


def run(timezone: str | None = None) -> dict:
    tz_name = timezone or DEFAULT_TZ
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        # Explicit error observation; eval I05 showed the model relays this shape honestly.
        return {"error": "invalid_timezone",
                "message": f"Unknown timezone: '{tz_name}'. Use an IANA name like 'Europe/Istanbul'."}
    now = datetime.now(tz)
    return {"datetime": now.isoformat(timespec="seconds"),
            "timezone": tz_name,
            "weekday": WEEKDAYS[now.weekday()]}
