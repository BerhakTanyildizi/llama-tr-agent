# -*- coding: utf-8 -*-
"""Builds a GBNF grammar dynamically from the active tool schemas.

Why a static .gbnf file is not enough: schemas come from the registry, so the
grammar has to change when a tool is added or removed. A hand-written file
inevitably drifts away from the registry.

GUARANTEE BOUNDARY (CLAUDE.md item 3 — the split is deliberate):
The grammar guarantees SYNTAX — valid JSON, key names DEFINED in the schema,
correct value TYPES, enum values.
The grammar does NOT guarantee SEMANTICS — picking the right tool, supplying
required parameters, key order. Those are fine-tuning's job.

Requiredness could have been encoded here, but every combination of optional
parameters would need its own branch (2^n for n optional params), making the
output unreadable. That check already runs in tools.validate() before dispatch.

NOTE: in eval the model produced 100% valid JSON with the grammar OFF. So this
layer is INSURANCE, not a necessity: it earns its keep once temperature rises
above zero and contexts get longer.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.tools import SCHEMAS  # noqa: E402

BASE = r'''string ::= "\"" ([^"\\] | "\\" .)* "\""
integer ::= "-"? [0-9]+
number ::= "-"? [0-9]+ ("." [0-9]+)?
boolean ::= "true" | "false"
ws ::= " "?'''


def _rule(*parts: str) -> str:
    """GBNF rule name: letters/digits/dashes. Underscores become dashes."""
    return "-".join(parts).replace("_", "-")


def _value(prop: dict, rule_name: str) -> tuple[str, list[str]]:
    """Value rule for one parameter. Returns (reference, extra rules)."""
    if "enum" in prop:
        options = " | ".join('"\\"%s\\""' % e for e in prop["enum"])
        return rule_name, [f"{rule_name} ::= {options}"]
    t = prop.get("type", "string")
    if t == "array":
        inner = "string" if prop.get("items", {}).get("type", "string") == "string" else "number"
        return rule_name, [f'{rule_name} ::= "[" ws ({inner} ("," ws {inner})*)? ws "]"']
    return {"integer": "integer", "number": "number",
            "boolean": "boolean"}.get(t, "string"), []


def build(schemas: list[dict] | None = None) -> str:
    schemas = schemas or SCHEMAS
    rules, call_rules = [], []

    for s in schemas:
        fn = s["function"]
        tool = fn["name"]
        call = _rule("call", tool)
        call_rules.append(call)
        props = fn["parameters"].get("properties", {})

        if not props:
            rules.append(f'{call} ::= "{{\\"name\\": \\"{tool}\\", \\"arguments\\": {{}}}}"')
            continue

        key_rules = []
        for k, prop in props.items():
            value_rule = _rule("val", tool, k)
            reference, extra = _value(prop, value_rule)
            rules += extra
            key_rule = _rule("key", tool, k)
            rules.append(f'{key_rule} ::= "\\"{k}\\":" ws {reference}')
            key_rules.append(key_rule)

        kv, pair = _rule("kv", tool), _rule("pair", tool)
        rules.append(f"{pair} ::= " + " | ".join(key_rules))
        rules.append(f'{kv} ::= {pair} ("," ws {pair})*')
        rules.append(
            f'{call} ::= "{{\\"name\\": \\"{tool}\\", \\"arguments\\": {{" ws {kv}? ws "}}}}"')

    root = ('root ::= "<tool_call>" ws-nl call ws-nl "</tool_call>"\n'
            'ws-nl ::= [ \\t\\n]*\n'
            "call ::= " + " | ".join(call_rules))
    return "\n".join([root, "", *rules, "", BASE]) + "\n"


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "tool_call.gbnf"
    text = build()
    target.write_text(text, encoding="utf-8")
    print(text)
    print(f"# written -> {target}")
