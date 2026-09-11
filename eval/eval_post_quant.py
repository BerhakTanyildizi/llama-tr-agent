# -*- coding: utf-8 -*-
"""Scores the quantized GGUF model's tool-call accuracy over eval/test_set.jsonl.

Talks to llama-server over HTTP. Server flags must match serve.py (q8_0 KV cache,
flash attention, -ngl 99) or the comparison is meaningless.

Scoring rules live in each record's `expect` and `scoring` fields.
"""
from __future__ import annotations

import argparse, json, re, sys, unicodedata, urllib.request
from collections import defaultdict
from pathlib import Path

TEST_SET = Path(__file__).resolve().parent / "test_set.jsonl"

# Prompt variants come from inference/prompts.py so that what we score here is
# byte-identical to what the orchestrator actually sends.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import prompts  # noqa: E402
from inference.grammar import generate_gbnf  # noqa: E402

VARIANTS = prompts.VARIANTS

HEADER, EOT = "<|start_header_id|>{}<|end_header_id|>\n\n", "<|eot_id|>"


def build_prompt(record: dict, variant: str) -> str:
    system = prompts.system_prompt(record["tools"], variant)
    p = "<|begin_of_text|>" + HEADER.format("system") + system + EOT
    for m in record["messages"]:
        p += HEADER.format(m["role"]) + m["content"] + EOT
    return p + HEADER.format("assistant")


def generate(url: str, prompt: str, grammar: str | None) -> str:
    payload = {"prompt": prompt, "n_predict": 320, "temperature": 0.0,
            "cache_prompt": True, "stop": [EOT]}
    if grammar:
        payload["grammar"] = grammar
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["content"]


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def extract_calls(text: str) -> tuple[list[dict], int]:
    """Returns (parsed calls, count of <tool_call> blocks that failed to parse)."""
    calls = []
    for m in TOOL_RE.finditer(text):
        try:
            j = json.loads(m.group(1))
            if isinstance(j, dict) and "name" in j:
                calls.append(j)
        except json.JSONDecodeError:
            pass
    return calls, max(0, text.count("<tool_call>") - len(calls))


def norm(x):
    return unicodedata.normalize("NFKC", x).strip().lower() if isinstance(x, str) else x


def value_matches(produced, accepted) -> bool:
    if accepted == "*":
        return produced not in (None, "", [], {})
    for a in (accepted if isinstance(accepted, list) else [accepted]):
        # bool branch must come first: isinstance(True, int) is True in Python
        if isinstance(a, bool) or isinstance(produced, bool):
            if produced is a:
                return True
        elif isinstance(a, (int, float)) and isinstance(produced, (int, float)):
            if float(produced) == float(a):
                return True
        elif isinstance(a, list) and isinstance(produced, list):
            if {norm(v) for v in produced} == {norm(v) for v in a}:
                return True
        elif norm(produced) == norm(a):
            return True
    return False


def score_args(produced: dict, expected: dict) -> list[str]:
    must, may = expected.get("must") or {}, set(expected.get("may") or [])
    if_present = expected.get("if_present") or {}
    errors = [f"missing:{k}" for k in must if k not in produced]
    errors += [f"wrong:{k}={produced[k]!r}" for k, acc in must.items()
            if k in produced and not value_matches(produced[k], acc)]
    errors += [f"wrong:{k}={produced[k]!r}" for k, acc in if_present.items()
            if k in produced and not value_matches(produced[k], acc)]
    errors += [f"extra:{k}" for k in produced
            if k not in must and k not in may and k not in if_present]
    return errors


# No re.IGNORECASE here: under Unicode case folding 'İ' matches plain 'i' and
# 'ı' matches 'I', which makes English text score as Turkish.
TR_CHARS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
TR_WORDS = re.compile(r"\b(bir|ve|için|ile|bu|şu|var|yok|göre|olarak|değil)\b")
EN_WORDS = re.compile(r"\b(the|and|is|are|of|to|in|that|with|for|you|your)\b")


def detect_language(text: str) -> str:
    lower = text.lower()
    tr = len(TR_CHARS.findall(text)) + len(TR_WORDS.findall(lower))
    en = len(EN_WORDS.findall(lower))
    return "?" if tr == en == 0 else ("tr" if tr >= en else "en")


def fabricated_args(args: dict, messages: list[dict]) -> list[str]:
    pool = norm(" ".join(m["content"] for m in messages))
    return [f"{k}={v!r}" for k, v in args.items()
            if isinstance(v, str) and len(v) > 2 and norm(v) not in pool]


def score_record(record: dict, output: str) -> dict:
    exp, mode = record["expect"], record["scoring"]
    calls, broken = extract_calls(output)
    name = calls[0]["name"] if calls else None
    result = {"id": record["id"], "category": record["category"],
            "unseen": not record["tools_seen"], "call": name,
            "call_count": len(calls), "broken_json": broken,
            "strict": {}, "report": {}, "reasons": [], "output": output[:300]}

    def put(field, passed, reason=""):
        (result["report"] if mode.get(field) == "report" else result["strict"])[field] = passed
        if not passed and reason:
            result["reasons"].append(reason)

    want = exp.get("tool_call")
    if want is None:
        put("tool_call", name is None, f"unexpected call: {name}")
    elif want == "either":
        result["report"]["ambiguous_called_tool"] = (name is not None)
    elif want == "prefer_none":
        result["report"]["tool_call"] = (name is None)
    elif want == "any_of":
        put("tool_call", name in exp["any_of"], f"want one of {exp['any_of']}, got {name}")
    else:
        put("tool_call", name == want, f"want {want}, got {name}")

    if name and want not in (None, "either", "prefer_none") and "args" in exp:
        produced = calls[0].get("arguments")
        if produced is None:
            produced = {}
        if not isinstance(produced, dict):
            # Observed: the model emitted "arguments": "17 * 23" - a string, not
            # an object. Scoring it as a mapping iterated the characters and
            # reported extra:1, extra:7, extra:*. Report the real defect instead.
            put("args", False, f"arguments is a {type(produced).__name__}, not an object")
        else:
            errs = score_args(produced, exp["args"])
            put("args", not errs, "args: " + ", ".join(errs))

    # Only meaningful when the model produced a final turn instead of a call.
    if "final_language" in exp and name is None:
        lang = detect_language(output)
        put("final_language", lang == exp["final_language"], f"language={lang}")

    if exp.get("must_not_fabricate_args") and calls:
        fake = fabricated_args(calls[0].get("arguments") if isinstance(
            calls[0].get("arguments"), dict) else {}, record["messages"])
        put("must_not_fabricate_args", not fake, "fabricated: " + ", ".join(fake))

    # Free-form arguments (a search `query`) cannot be matched against a fixed
    # reference: any wording is acceptable as long as it is about the right
    # SUBJECT. `must` with "*" only asks for non-empty, which would pass a model
    # that repeats its previous query verbatim instead of searching the second
    # topic - exactly the S04 failure. args_contains asks for the subject.
    if "args_contains" in exp and calls and name == want:
        produced = calls[0].get("arguments")
        produced = produced if isinstance(produced, dict) else {}
        missing = []
        for k, needles in exp["args_contains"].items():
            haystack = norm(str(produced.get(k, "")))
            if not any(norm(s) in haystack for s in needles):
                missing.append(f"{k}={produced.get(k)!r} mentions none of {needles}")
        put("args_contains", not missing, "; ".join(missing))

    # Verbatim carry-over from an earlier turn. The model appended a previous
    # turn's answer ("the result ... is -1") to an unrelated one; nothing in the
    # loop noticed. A substring check is crude but deterministic, and the
    # forbidden strings are chosen to be absent from any correct answer.
    if "must_not_repeat" in exp:
        leaked = [s for s in exp["must_not_repeat"] if s.lower() in output.lower()]
        put("must_not_repeat", not leaked, "repeated from an earlier turn: " + ", ".join(leaked))

    # Only judge argument language when the expected tool was actually called;
    # otherwise there is no such argument and the metric reports a false 0%.
    if "args_lang" in exp and calls and name == want:
        for k, want_lang in exp["args_lang"].items():
            v = (calls[0].get("arguments") or {}).get(k, "")
            result["report"][f"args_lang:{k}"] = (detect_language(str(v)) == want_lang)
    if "expected_call_count" in exp:
        result["report"]["expected_call_count"] = (len(calls) == exp["expected_call_count"])
    if broken:
        result["reasons"].append(f"{broken} unparseable <tool_call> block(s)")
    return result


def pct(a, b):
    return f"{a}/{b} ({100 * a // b if b else 0:>3}%)"


def print_report(results: list[dict], variant: str, grammar: bool):
    print(f"\n{'=' * 74}\nRESULTS  |  prompt={variant}  grammar={'ON' if grammar else 'OFF'}\n{'=' * 74}")

    # all({}) is True, so a record with no evaluated strict field would count as
    # a pass and inflate the score. Such records are excluded and listed apart.
    def summary(subset):
        measured = [r for r in subset if r["strict"]]
        ok = sum(1 for r in measured if all(r["strict"].values()))
        tail = f"  [{len(subset) - len(measured)} unmeasured]" if len(measured) != len(subset) else ""
        return pct(ok, len(measured)) + tail

    print(f"\nOVERALL (strict fields)  : {summary(results)}")
    unseen = [r for r in results if r["unseen"]]
    seen = [r for r in results if not r["unseen"]]
    print(f"  seen schemas           : {summary(seen)}")
    print(f"  UNSEEN schemas         : {summary(unseen)}   <- generalization (Spec 5.3)")
    broken = sum(r["broken_json"] for r in results)
    called = sum(1 for r in results if r["call"])
    print(f"\nJSON validity            : {pct(called, called + broken)}  ({broken} unparseable)")

    print("\nBY CATEGORY")
    groups = defaultdict(list)
    for r in results:
        groups[r["category"]].append(r)
    for k in sorted(groups):
        print(f"  {k:<20} {summary(groups[k])}")

    rep = defaultdict(lambda: [0, 0])
    for r in results:
        for k, v in r["report"].items():
            rep[k][0] += bool(v); rep[k][1] += 1
    if rep:
        print("\nREPORT ONLY (excluded from the score)")
        for k, (ok, total) in sorted(rep.items()):
            print(f"  {k:<26} {pct(ok, total)}")

    failed = [r for r in results if r["strict"] and not all(r["strict"].values())]
    if failed:
        print(f"\nFAILED ({len(failed)})")
        for r in failed:
            print(f"  [{r['id']}] {r['category']}: {'; '.join(r['reasons'])}")

    unmeasured = [r for r in results if not r["strict"]]
    if unmeasured:
        print(f"\nUNMEASURED ({len(unmeasured)}) - the model took the branch that "
            f"carries no strict field; excluded from the score.")
        for r in unmeasured:
            print(f"  [{r['id']}] {r['category']}: call={r['call']}")

    print("\nNEEDS MANUAL READING (not automatically checkable)")
    for r in results:
        if r["id"] in ("I05", "I06", "N04"):
            print(f"  [{r['id']}] fabricated content? -> {r['output'][:160].strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/completion")
    ap.add_argument("--variant", default=prompts.DEFAULT_VARIANT, choices=list(VARIANTS))
    ap.add_argument("--grammar", action="store_true",
                    help="constrain decoding with a GBNF built from EACH RECORD's tools")
    ap.add_argument("--only", help="run only ids/categories with this prefix")
    ap.add_argument("--out", help="write results as JSON")
    a = ap.parse_args()

    try:
        urllib.request.urlopen(a.url.replace("/completion", "/health"), timeout=5)
    except Exception as err:
        raise SystemExit(
            f"cannot reach llama-server ({a.url}): {err}\n"
            "start it with:\n"
            "  cd ~/llama-bin/llama-b10632 && LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./llama-server \\\n"
            "    -m ~/Desktop/MyAgent/training/outputs/gguf/llama31-8b-tr-Q4_K_M.gguf \\\n"
            "    -c 8192 -np 1 -ngl 99 --device Vulkan1 -fa on \\\n"
            "    --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port 8080")


    records = [json.loads(s) for s in TEST_SET.open(encoding="utf-8") if s.strip()]
    if a.only:
        records = [r for r in records
                if r["id"].startswith(a.only) or r["category"].startswith(a.only)]

    results = []
    for i, rec in enumerate(records, 1):
        # Built per record, not once: a fixed grammar from the production
        # registry would forbid every unseen tool the record declares, and the
        # model would be forced to pick a production tool instead. That scored
        # unseen_positive at 0/11 and looked like a model failure.
        gbnf = generate_gbnf.build(rec["tools"]) if a.grammar else None
        out = generate(a.url, build_prompt(rec, a.variant), gbnf)
        r = score_record(rec, out)
        results.append(r)
        status = "SKIP" if not r["strict"] else ("OK  " if all(r["strict"].values()) else "FAIL")
        print(f"[{i:>2}/{len(records)}] {status} {rec['id']:<5} {rec['category']}")

    print_report(results, a.variant, a.grammar)
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"variant": a.variant, "grammar": a.grammar, "results": results},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report -> {a.out}")


if __name__ == "__main__":
    main()
