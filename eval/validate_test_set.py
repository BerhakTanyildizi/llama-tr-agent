# -*- coding: utf-8 -*-
"""Structural checks for eval/test_set.jsonl. Run it after every edit.

It lives in the repo rather than a scratch directory because the previous copy
did not, and was lost with the session. Every check here fired on a real defect
at least once while the set was being built.

    python eval/validate_test_set.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.tools import REGISTRY                                   # noqa: E402

TEST_SET = ROOT / "eval" / "test_set.jsonl"
TRAIN = [ROOT / "training/data/train.jsonl", ROOT / "training/data/eval.jsonl"]

# Schemas allowed to differ from the training data, with the reason. Anything
# else that differs is treated as accidental drift and reported as an error.
INTENTIONAL_DRIFT = {
    "get_weather": "days_ahead added for 'tomorrow' / 'in N days' questions",
    "calculate": "tool added after training; the model cannot do arithmetic reliably",
    "search_notes": "tool added after training; local notes retrieval",
}

TYPES = {"string": str, "integer": int, "number": (int, float),
        "boolean": bool, "array": list, "object": dict}
SPECIAL = (None, "either", "prefer_none", "any_of")

errors, warnings = [], []


def training_inventory():
    """Tool names and user messages seen in training, for drift and leak checks."""
    names, messages, schemas = set(), set(), {}
    for path in TRAIN:
        if not path.exists():
            warnings.append(f"{path.name} missing - drift and leak checks skipped")
            continue
        for line in path.open(encoding="utf-8"):
            text = json.loads(line)["text"]
            block = re.search(r"<tools>\s*\n(.*?)\n\s*</tools>", text, re.S)
            if block:
                try:
                    for fn in json.loads(block.group(1)):
                        names.add(fn["function"]["name"])
                        schemas.setdefault(fn["function"]["name"], fn)
                except Exception:
                    pass
            for m in re.finditer(
                    r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", text, re.S):
                messages.add(m.group(1).strip())
    return names, messages, schemas


def check_registry_drift(train_schemas):
    for name, module in REGISTRY.items():
        if name not in train_schemas or module.SCHEMA == train_schemas[name]:
            continue
        if name in INTENTIONAL_DRIFT:
            print(f"  intentional drift  {name}: {INTENTIONAL_DRIFT[name]}")
        else:
            errors.append(f"[registry] '{name}' differs from training and is not "
                        f"listed in INTENTIONAL_DRIFT - accidental?")


def check_record(rec, seen_ids, train_names, train_messages):
    rid = rec.get("id", "?")
    for field in ("id", "category", "tools", "messages", "expect", "scoring", "tools_seen"):
        if field not in rec:
            errors.append(f"[{rid}] missing field: {field}")
    if rid in seen_ids:
        errors.append(f"[{rid}] duplicate id")
    seen_ids.add(rid)

    schemas = {t["function"]["name"]: t["function"] for t in rec.get("tools", [])}
    if not schemas:
        errors.append(f"[{rid}] no tools")

    # The eval must score the schema production actually uses.
    for tool in rec.get("tools", []):
        name = tool["function"]["name"]
        if name in REGISTRY and tool != REGISTRY[name].SCHEMA:
            errors.append(f"[{rid}] '{name}' schema differs from the registry "
                        f"(eval/production drift)")
    # "unseen" must really be unseen.
    unseen = [n for n in schemas if n not in train_names]
    if rec.get("tools_seen") != (not unseen):
        errors.append(f"[{rid}] tools_seen={rec.get('tools_seen')} but unseen tools are {unseen}")

    messages = rec.get("messages", [])
    if messages and messages[0]["role"] != "user":
        errors.append(f"[{rid}] first message is not from the user")
    if messages and messages[-1]["role"] not in ("user", "ipython"):
        errors.append(f"[{rid}] last role '{messages[-1]['role']}' leaves nothing to generate")
    for a, b in zip(messages, messages[1:]):
        if a["role"] == b["role"]:
            errors.append(f"[{rid}] two '{a['role']}' turns in a row")
    for m in messages:
        if m["role"] != "ipython":
            continue
        try:                                   # the double-encoded training shape
            inner = json.loads(m["content"])
            assert inner.startswith("<tool_response>\n") and inner.endswith("\n</tool_response>")
            json.loads(inner[len("<tool_response>\n"):-len("\n</tool_response>")])
        except Exception as e:
            errors.append(f"[{rid}] observation is not a valid <tool_response> block: {e}")
    for m in messages:
        if m["role"] == "user" and m["content"].strip() in train_messages:
            errors.append(f"[{rid}] LEAK - this user message is verbatim in the training data")

    exp, mode = rec.get("expect", {}), rec.get("scoring", {})
    want = exp.get("tool_call")
    if want not in SPECIAL and want not in schemas:
        errors.append(f"[{rid}] expected tool '{want}' is not in this record's tools")
    for field in exp:
        if field not in mode and field not in ("args", "any_of"):
            warnings.append(f"[{rid}] '{field}' has no scoring mode - it silently counts as strict")
    if want in SPECIAL:
        return

    props = schemas[want]["parameters"].get("properties", {})
    required = set(schemas[want]["parameters"].get("required", []))
    args = exp.get("args") or {}
    must, may = args.get("must") or {}, set(args.get("may") or [])
    if_present = args.get("if_present") or {}
    for key in list(must) + list(may) + list(if_present):
        if key not in props:
            errors.append(f"[{rid}] '{key}' is not a parameter of {want}")
    if required - set(must):
        errors.append(f"[{rid}] required parameter(s) missing from the reference: "
                    f"{sorted(required - set(must))}")
    for key, accepted in list(must.items()) + list(if_present.items()):
        if accepted == "*" or key not in props:
            continue
        kind, enum = props[key].get("type"), props[key].get("enum")
        for value in (accepted if isinstance(accepted, list) else [accepted]):
            if kind in TYPES and kind != "array" and not isinstance(value, TYPES[kind]):
                errors.append(f"[{rid}] '{key}' value {value!r} is not of schema type '{kind}'")
            if enum and isinstance(value, str) and value not in enum:
                errors.append(f"[{rid}] '{key}' value {value!r} is not in enum {enum}")


def main():
    records = [json.loads(line) for line in TEST_SET.open(encoding="utf-8") if line.strip()]
    train_names, train_messages, train_schemas = training_inventory()
    print(f"{len(records)} records | training inventory: {len(train_names)} tool names, "
        f"{len(train_messages)} user messages")

    check_registry_drift(train_schemas)
    seen_ids = set()
    for rec in records:
        check_record(rec, seen_ids, train_names, train_messages)

    from collections import Counter
    print("\nBY CATEGORY")
    for name, n in sorted(Counter(r["category"] for r in records).items()):
        print(f"  {name:<22}{n:>3}")
    unseen = sum(1 for r in records if not r["tools_seen"])
    print(f"\n  unseen schemas: {unseen}   seen schemas: {len(records) - unseen}")

    print()
    if errors:
        print(f"ERRORS ({len(errors)})")
        for e in errors:
            print(f"  {e}")
    else:
        print("NO ERRORS")
    if warnings:
        print(f"\nWARNINGS ({len(warnings)})")
        for w in warnings:
            print(f"  {w}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
