# -*- coding: utf-8 -*-
"""ReAct loop: user message -> tool calls -> observations -> Turkish answer.

    user message
        |
        v  build_prompt -> model.generate
        |
        +-- no <tool_call> --> TURKISH FINAL ANSWER, done
        |
        +-- <tool_call> --> validate --> dispatch --> observation --> loop

Decisions that came out of measurement rather than design:

1. SYSTEM PROMPT VARIANT: see inference/prompts.py. V2 won the offline eval
   (93% vs V1's 90%, wrong_tool_trap 5/5 vs 3/5) but failed in live use by
   suppressing an explicit search request and then inventing an answer. V3 adds
   two escape hatches for that. The asymmetry is deliberate and was the user's
   call: an unnecessary search costs latency, a skipped one costs a wrong answer.

2. ARGUMENT GROUNDING (_ungrounded) IS A REQUIRED LAYER, NOT A NICETY. Both
   prompt variants failed the same way: "Hava nasıl?" produced
   location='Ankara', and "Ankara'dan tren var mı?" produced
   destination="user's destination" with date='2022-03-15'. The model knows it
   does not know - "user's destination" is a placeholder - and fills the slot
   anyway instead of asking. Prompting does not fix this, so it is enforced here.

3. OBSERVATIONS USE THE `ipython` ROLE AND ARE DOUBLE-ENCODED. The training data
   wraps the tool response in a JSON string, quotes and literal \\n included.
   It is an artifact of the Hermes conversion, but the model learned that exact
   shape; not reproducing it puts the model off-distribution.

4. CONTEXT IS TRIMMED FOR LATENCY, NOT MEMORY. Measured: n_ctx=8192 uses only
   544 MiB of KV cache with ~2.7 GB free, so memory is not the constraint.
   Prompt processing at ~175 tok/s is: every 1000 context tokens costs ~6 s.
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import prompts, serve, tools                     # noqa: E402
from inference.grammar import generate_gbnf                     # noqa: E402

# --- prompt building --------------------------------------------------------
# Variants live in inference/prompts.py so eval and production cannot drift.
HEADER, EOT = "<|start_header_id|>{}<|end_header_id|>\n\n", "<|eot_id|>"

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
# Values that betray the model filling a slot it cannot fill (eval E04).
PLACEHOLDER_RE = re.compile(
    r"^(unknown|n/?a|tbd|none|null|todo|string)$"
    r"|[<\[]|\.\.\."
    r"|\b(your|users?'?s?|placeholder|example|specify|insert)\b", re.I)

MAX_ITERATIONS = 6
CONTEXT_BUDGET = 0.70      # of n_ctx; the rest is left for generation
PRUNED_NOTE = "Earlier result removed to save context."
GIVE_UP = ("Bu isteği tamamlayamadım. Sorunu biraz daha açık yazarsan "
        "yeniden deneyebilirim.")


# Turkish diacritics are folded before comparison. Without this the guard fires
# on the model doing the right thing: the user types "Elazıgda", the model
# correctly normalises it to "Elazığ", and a plain string match fails because
# 'ğ' != 'g'. Observed live on the first run.
_FOLD = str.maketrans("çğıİöşüÇĞÖŞÜ", "cgiiosucgosu")


def _norm(s: str) -> str:
    # translate BEFORE lower(): "İ".lower() is two codepoints (i + combining
    # dot) in Python and would slip past the table.
    s = re.sub(r"\s+", " ", s).translate(_FOLD).lower().replace("̇", "")
    return s.strip()


def _grounded(value: str, haystack: str) -> bool:
    """True if the value, or any word of it, was actually said in the conversation."""
    v = _norm(value)
    if v and v in haystack:
        return True
    return any(w in haystack for w in re.findall(r"\w{3,}", v))


class Orchestrator:
    def __init__(self, model: serve.Model | None = None, use_grammar: bool = False,
                max_iterations: int = MAX_ITERATIONS, verbose: bool = False,
                variant: str = prompts.DEFAULT_VARIANT, directive: bool = True):
        self.model = model or serve.Model()
        # Grammar is off by default: the model produced 100% valid JSON without
        # it in eval, so it is insurance for temperature > 0 and long contexts.
        self.grammar = generate_gbnf.build() if use_grammar else None
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.budget = int(self.model.n_ctx * CONTEXT_BUDGET)
        self.system = prompts.system_prompt(tools.SCHEMAS, variant)
        self.directive = directive
        self.history: list[dict] = []

    # -- prompt ------------------------------------------------------------

    def build_prompt(self, messages: list[dict]) -> str:
        p = "<|begin_of_text|>" + HEADER.format("system") + self.system + EOT
        for m in messages:
            p += HEADER.format(m["role"]) + m["content"] + EOT
        # Language + length directive, restated immediately before generation.
        # It lives here rather than in the system prompt because recency is the
        # whole point (see prompts.final_directive). It is appended after the
        # cached prefix, so the system prompt's KV cache is untouched.
        if self.directive:
            p += HEADER.format("system") + prompts.final_directive(messages) + EOT
        return p + HEADER.format("assistant")   # left open: the model speaks next

    @staticmethod
    def observation(name: str, content: dict) -> str:
        """The double-encoded <tool_response> shape the model was trained on."""
        return json.dumps("<tool_response>\n"
                        + json.dumps({"name": name, "content": content})
                        + "\n</tool_response>")

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def extract_call(text: str) -> tuple[str | None, dict, str | None]:
        """Returns (tool name, arguments, error). All None/empty means final answer."""
        m = TOOL_CALL_RE.search(text)
        if not m:
            # An opening tag with no closing one would otherwise pass silently.
            if "<tool_call>" in text:
                return None, {}, "Malformed tool_call: the block was never closed."
            return None, {}, None
        try:
            call = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            return None, {}, f"Malformed tool_call JSON: {e}"
        args = call.get("arguments")
        return call.get("name"), args if isinstance(args, dict) else {}, None

    def _ungrounded(self, name: str, args: dict, messages: list[dict]) -> list[str]:
        """Arguments the model invented instead of asking for. See decision 2.

        Only the user's words and tool results count as grounding. Assistant
        turns are excluded on purpose: otherwise a value the model invented in
        one turn launders itself into "grounded" in the next.
        """
        must_ground = tools.grounded_params(name)
        said = _norm(" ".join(m["content"] for m in messages
                            if m["role"] in ("user", "ipython")))
        problems = []
        for k, v in args.items():
            if not isinstance(v, str) or not v.strip():
                continue
            if PLACEHOLDER_RE.search(v):
                problems.append(f"'{k}' is a placeholder ({v!r}), not a real value")
            elif k in must_ground and not _grounded(v, said):
                problems.append(f"'{k}' was set to {v!r}, which the user never mentioned")
        return problems

    # -- context -----------------------------------------------------------
    def _fit_context(self, messages: list[dict]) -> None:
        """Shorten the oldest observations until the prompt fits the budget.

        Only observation bodies are dropped; the system prompt and the user's
        messages stay, so the large fixed prefix keeps its KV cache.

        The MOST RECENT observation is never pruned - it is what the model needs
        to answer with. If the budget still cannot be met the request proceeds
        oversized rather than answering from a hole.
        """
        if self.model.count_tokens(self.build_prompt(messages)) <= self.budget:
            return
        prunable = [m for m in messages if m["role"] == "ipython" and not m.get("pruned")]
        for m in prunable[:-1]:
            m["content"] = self.observation(m.get("tool", "unknown"), {"note": PRUNED_NOTE})
            m["pruned"] = True
            if self.model.count_tokens(self.build_prompt(messages)) <= self.budget:
                return
        self._log(f"context still over budget ({self.budget} tokens) after pruning; "
                "consider a larger n_ctx or tighter tool-side trimming")

    def _log(self, *a):
        if self.verbose:
            print("  ·", *a, file=sys.stderr)

    # -- main loop ---------------------------------------------------------
    def answer(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        already_called: set[str] = set()

        for step in range(self.max_iterations):
            last = step == self.max_iterations - 1
            out = self.model.generate(self.build_prompt(self.history),
                                    grammar=self.grammar, stop=[EOT])["text"].strip()
            name, args, parse_error = self.extract_call(out)

            if name is None and not parse_error:
                self.history.append({"role": "assistant", "content": out})
                return out                                   # Turkish final answer

            self.history.append({"role": "assistant", "content": out})

            if parse_error:
                result = {"error": "invalid_tool_call", "message": parse_error}
            elif (problems := self._ungrounded(name, args, self.history[:-1])):
                # Do NOT dispatch: ask the user instead of acting on invented input.
                result = {"error": "missing_information",
                        "message": "Do not guess these values. Ask the user for them "
                                    "in Turkish: " + "; ".join(problems)}
            elif (signature := f"{name}:{json.dumps(args, sort_keys=True)}") in already_called:
                result = {"error": "repeated_call",
                        "message": "You already made this exact call. Use the result "
                                    "you already have, or answer the user."}
            else:
                already_called.add(signature)
                result = tools.dispatch(name, args)
                # Log the OUTCOME, not the request: the tool overrides some
                # arguments (google_search treats num_results as a floor), so
                # logging the model's args alone is misleading.
                if "error" in result:
                    outcome = f"error:{result['error']}"
                elif "source_count" in result:
                    kaynak = result.get("provider", "?").lstrip("_")
                    outcome = f"{result['source_count']} kaynak ({kaynak})"
                else:
                    outcome = "ok"
                self._log(f"{name}({args}) -> {outcome}")

            if last:
                result = dict(result, note="This was the last tool call allowed. "
                                        "Answer the user in Turkish now.")
            self.history.append({"role": "ipython", "tool": name or "unknown",
                                "content": self.observation(name or "unknown", result)})
            self._fit_context(self.history)

        # Budget exhausted and still calling tools: one forced final answer.
        out = self.model.generate(self.build_prompt(self.history), stop=[EOT])["text"].strip()
        if self.extract_call(out)[0]:
            out = GIVE_UP
        self.history.append({"role": "assistant", "content": out})
        return out

    def reset(self) -> None:
        self.history.clear()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Turkish ReAct agent (REPL)")
    ap.add_argument("--url", default=serve.DEFAULT_URL)
    ap.add_argument("--grammar", action="store_true", help="enable GBNF constrained decoding")
    ap.add_argument("--verbose", action="store_true", help="log tool calls to stderr")
    ap.add_argument("--variant", default=prompts.DEFAULT_VARIANT, choices=list(prompts.VARIANTS),
                    help="system prompt variant")
    a = ap.parse_args()

    try:
        agent = Orchestrator(serve.Model(a.url), use_grammar=a.grammar,
                            verbose=a.verbose, variant=a.variant)
    except serve.ServerUnavailable as e:
        raise SystemExit(str(e))

    print(f"n_ctx={agent.model.n_ctx}  context budget={agent.budget} tokens  "
        f"grammar={'on' if agent.grammar else 'off'}  prompt={a.variant}")
    print("Sorunuzu yazın (çıkmak için Ctrl+D, geçmişi silmek için /reset).\n")
    while True:
        try:
            line = input("siz > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line == "/reset":
            agent.reset()
            print("geçmiş temizlendi\n")
            continue
        print(f"ajan> {agent.answer(line)}\n")
        


if __name__ == "__main__":
    main()
