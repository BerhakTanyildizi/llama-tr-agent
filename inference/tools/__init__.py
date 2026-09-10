# -*- coding: utf-8 -*-
"""Tool registry.

Single source of truth. Three consumers read from this same list, so one can
never be updated while another is forgotten:
  1. orchestrator.py  -> writes the <tools> block into the system prompt
  2. generate_gbnf.py -> builds the grammar
  3. orchestrator.py  -> dispatches the produced tool_call here

Adding a tool: write the module (SCHEMA + GROUNDED_PARAMS + run), append it to
MODULES. Nothing else changes.
"""
from . import calculate, get_system_time, get_weather, google_search

MODULES = [google_search, get_weather, get_system_time, calculate]

REGISTRY = {m.SCHEMA["function"]["name"]: m for m in MODULES}
SCHEMAS = [m.SCHEMA for m in MODULES]

# JSON schema type -> accepted Python types. bool must be handled before int:
# isinstance(True, int) is True in Python (this trap was hit once in eval).
TYPES = {"string": str, "integer": int, "number": (int, float),
         "boolean": bool, "array": list, "object": dict}


def grounded_params(name: str) -> frozenset[str]:
    """Arguments that must be traceable to the conversation before dispatch.

    Declared per tool because the distinction is not derivable: 'location' is
    extracted from the user's words, while google_search's 'query' is composed
    by the model. Grounding the latter would reject valid searches.
    """
    mod = REGISTRY.get(name)
    return getattr(mod, "GROUNDED_PARAMS", frozenset()) if mod else frozenset()


def required_params(name: str) -> frozenset[str]:
    """Parameters the schema marks required.

    The orchestrator needs this to tell two placeholders apart: one sitting in a
    required slot is a fabrication and must block the call, while one in an
    optional slot just means the model had nothing to put there.
    """
    mod = REGISTRY.get(name)
    if not mod:
        return frozenset()
    return frozenset(mod.SCHEMA["function"]["parameters"].get("required", []))


def validate(name: str, args: dict) -> str | None:
    """Schema validation. Returns an explanation on failure, None on success.

    This layer only looks at the schema. The 'is this argument fabricated'
    check needs conversation context and belongs to the orchestrator
    (eval E01/E04 finding).
    """
    if name not in REGISTRY:
        return f"Unknown tool '{name}'. Available: {', '.join(REGISTRY)}."
    params = REGISTRY[name].SCHEMA["function"]["parameters"]
    props, required = params.get("properties", {}), params.get("required", [])

    for k in required:
        if k not in args:
            return f"Missing required parameter '{k}' for {name}."
    for k, v in args.items():
        if k not in props:
            return f"Unknown parameter '{k}' for {name}."
        t = props[k].get("type")
        if t == "boolean" and not isinstance(v, bool):
            return f"Parameter '{k}' must be a boolean."
        if t in TYPES and t != "boolean":
            if isinstance(v, bool) or not isinstance(v, TYPES[t]):
                return f"Parameter '{k}' must be of type {t}."
        enum = props[k].get("enum")
        if enum and v not in enum:
            return f"Parameter '{k}' must be one of {enum}."
    return None


def dispatch(name: str, args: dict) -> dict:
    """tool_call -> observation. NEVER raises.

    Every failure becomes an observation the model can read, so the ReAct loop
    cannot crash; the model then retries or reports it honestly to the user
    (eval I05 showed it does the latter correctly).
    """
    problem = validate(name, args or {})
    if problem:
        return {"error": "invalid_tool_call", "message": problem}
    try:
        return REGISTRY[name].run(**(args or {}))
    except Exception as e:
        return {"error": "tool_failed", "message": f"{type(e).__name__}: {str(e)[:120]}"}
