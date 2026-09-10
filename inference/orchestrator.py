# -*- coding: utf-8 -*-
"""ReAct loop: user message -> tool calls -> observations -> final answer.

    user message
        |
        v  build_prompt -> model.generate
        |
        +-- no <tool_call> --> FINAL ANSWER, done
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

# "digit operator digit", with ISO dates and year ranges removed first.
# ADVISORY ONLY - deliberately not a gate. Measured over the 7549 unique user
# messages in the training data: the naive form fires on 4.85% of them and this
# narrowed form still fires on 1.44%, almost all false positives (phone numbers,
# "the digits 1-9", pasted passages, "MATLAB code to calculate ..."). Blocking a
# final answer on that would refuse roughly one legitimate turn in seventy.
# It also MISSES arithmetic phrased in words - "17 ile 23'un carpimi kactir?"
# has no operator at all - so a gate built on it would be both noisy and leaky.
# The behaviour is left to the directive; S02 in the test set measures whether
# that is enough, and this warning is how the failure becomes visible meanwhile.
DATE_LIKE_RE = re.compile(r"\d{4}-\d{1,2}(-\d{1,2})?|\b(19|20)\d{2}\s*[-\u2013]\s*(19|20)\d{2}\b")
ARITHMETIC_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?\s*[-+*/\u00d7\u00f7]\s*\d+(?:\.\d+)?(?![\d.])")

MAX_ITERATIONS = 6
CONTEXT_BUDGET = 0.70      # of n_ctx; the rest is left for generation
PRUNED_NOTE = "Earlier result removed to save context."
# Language-aware: the loop can now run in English, and a Turkish fallback in an
# English conversation reads as a crash.
GIVE_UP = {
    "en": "I could not complete that request. Rephrasing it a little may help.",
    "tr": "Bu isteği tamamlayamadım. Sorunu biraz daha açık yazarsan yeniden deneyebilirim.",
}

# The loop writes two observations that tell the model which language to answer
# in. They used to say "Turkish" unconditionally, which contradicted the
# directive on every English turn - and the directive is the one the model
# obeys, so the observation was pure off-distribution noise. The PROSE stays
# English like every other observation (hybrid strategy: tool output English,
# answer localized); only the language it NAMES follows the resolved language.
LANGUAGE_NAMES = {"en": "English", "tr": "Turkish"}


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


class _ProseStream:
    """Forwards a generation to `write`, but only when it turns out to be prose.

    The grammar guarantees a tool call starts with '<' and a plain answer never
    does, so the FIRST non-whitespace character settles it. That one character
    is why the tool_call JSON never reaches the screen: text is held back until
    the branch is known, then either released and streamed, or dropped for good.
    """

    def __init__(self, write):
        self._write = write
        self._prose = None          # None -> undecided
        self._held = ""

    def __call__(self, piece: str) -> None:
        if self._prose is False:
            return
        if self._prose is None:
            self._held += piece
            if not self._held.strip():
                return              # still only whitespace, cannot decide yet
            self._prose = not self._held.lstrip().startswith("<")
            if not self._prose:
                return
            piece, self._held = self._held, ""
        self._write(piece)


class Orchestrator:
    def __init__(self, model: serve.Model | None = None, use_grammar: bool = True,
                max_iterations: int = MAX_ITERATIONS, verbose: bool = False,
                variant: str = prompts.DEFAULT_VARIANT, directive: bool = True,
                language: str | None = prompts.DEFAULT_LANGUAGE, on_text=None):
        self.model = model or serve.Model()
        # Grammar is ON by default. It was off while the agent had only the three
        # tools it was trained on, where the model produced 100% valid JSON
        # unaided. Adding `calculate` - a tool absent from training - broke that:
        # the model emitted "arguments": "16 * 2" as a bare string instead of an
        # object. Measured over the whole set: JSON validity 96% -> 100%,
        # reasoning 40% -> 100%, overall 84% -> 89%, and no category dropped.
        self.grammar = generate_gbnf.build() if use_grammar else None
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.budget = int(self.model.n_ctx * CONTEXT_BUDGET)
        self.directive = directive
        self.language = language
        # When set, prose is streamed to this callback as it is generated.
        # None keeps the buffered behaviour, which is what the eval wants.
        self.on_text = on_text
        # The system prompt names ONE language, chosen from the configured mode
        # rather than per turn: it is the cached prefix, so a per-turn clause
        # would throw away the KV cache every time the language flipped. With
        # the default "en" the prompt no longer mentions Turkish at all, which
        # removes the standing contradiction with the English directive.
        # None -> DEFAULT_LANGUAGE -> "auto" mirrors resolve_language()'s own
        # precedence, so the two cannot disagree about what "unset" means.
        self.system = prompts.system_prompt(
            tools.SCHEMAS, variant,
            language=language or prompts.DEFAULT_LANGUAGE or "auto")
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
            p += HEADER.format("system") + prompts.final_directive(messages, self.language) + EOT
        return p + HEADER.format("assistant")   # left open: the model speaks next

    @staticmethod
    def observation(name: str, content: dict) -> str:
        """The double-encoded <tool_response> shape the model was trained on."""
        return json.dumps("<tool_response>\n"
                        + json.dumps({"name": name, "content": content})
                        + "\n</tool_response>")

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def extract_calls(text: str) -> list[tuple[str | None, dict, str | None]]:
        """One (name, arguments, error) entry per <tool_call> block, in order.

        An empty list means the model wrote a final answer instead of calling.

        findall, NOT search. The grammar has allowed `tool-call (ws-nl
        tool-call)*` since the multi-call fix, and the eval parses every block
        with finditer - but this parser read only the FIRST block and dropped
        the rest with no log, no observation and no error. The model never
        learned its second call had been eaten. Observed live: "use 2 search
        tool" ran one search and answered half the question, which read as a
        model failure and was a harness bug (the same shape as the three
        grammar traps in CLAUDE.md item 14).

        A malformed block becomes an entry carrying `error` rather than
        discarding the whole batch: the well-formed calls beside it are still
        worth dispatching, and the model is told exactly which one was broken.
        """
        blocks = TOOL_CALL_RE.findall(text)
        if not blocks:
            # An opening tag with no closing one would otherwise pass silently.
            if "<tool_call>" in text:
                return [(None, {}, "Malformed tool_call: the block was never closed.")]
            return []
        calls = []
        for raw in blocks:
            try:
                call = json.loads(raw)
            except json.JSONDecodeError as e:
                calls.append((None, {}, f"Malformed tool_call JSON: {e}"))
                continue
            args = call.get("arguments")
            calls.append((call.get("name"), args if isinstance(args, dict) else {}, None))
        return calls

    def _ungrounded(self, name: str, args: dict, messages: list[dict]) -> list[str]:
        """Arguments the model invented instead of asking for. See decision 2.

        Only the user's words and tool results count as grounding. Assistant
        turns are excluded on purpose: otherwise a value the model invented in
        one turn launders itself into "grounded" in the next.
        """
        must_ground = tools.grounded_params(name)
        required = tools.required_params(name)
        said = _norm(" ".join(m["content"] for m in messages
                            if m["role"] in ("user", "ipython")))
        problems = []
        for k, v in list(args.items()):
            if not isinstance(v, str) or not v.strip():
                continue
            if PLACEHOLDER_RE.search(v):
                if k in required:
                    problems.append(f"'{k}' is a placeholder ({v!r}), not a real value")
                else:
                    # A placeholder in an OPTIONAL slot is not a fabrication, it
                    # is an empty hand: the model was told to fill the field and
                    # had nothing real for it. Blocking the whole call there is
                    # wrong. calculate's `unit` is the live case - it is optional,
                    # its description says "always give it", and "4 - 5" has no
                    # unit, so the model reaches for "n/a" / "none" / "null",
                    # all three of which match PLACEHOLDER_RE. That refused
                    # correctly-formed arithmetic, the exact call we want it to
                    # make. Dropping the argument lets the tool answer and emit
                    # its own "say what it counts" message instead.
                    del args[k]
                    self._log(f"dropped optional {name}.{k}={v!r} (placeholder)")
                continue
            if k in must_ground and not _grounded(v, said):
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

    def _check_numbers(self, answer: str) -> None:
        """Warn when the answer states a number no observation contained.

        Advisory only, and only under --verbose. It exists because the model
        produced "256 billion parameters" from sources that said nothing of the
        kind, with no signal that anything was wrong. Gating on it would be
        wrong - a legitimate calculation also yields new numbers - but seeing it
        while debugging is the difference between catching that and not.
        """
        if not self.verbose:
            return
        seen = " ".join(m["content"] for m in self.history if m["role"] == "ipython")
        # rstrip: the character class swallows the sentence's full stop, so
        # "27.4." would be reported and would not match "27.4" in the observation.
        unseen = {n for n in (m.rstrip(".,") for m in re.findall(r"\d[\d.,]*", answer))
                  if len(n) > 2 and n not in seen}
        if unseen:
            self._log(f"not in any observation: {', '.join(sorted(unseen)[:6])}")

        # A number that appears in an EARLIER ANSWER but in no observation is
        # the carry-over case: the model restating something it said in a turn
        # that never asked for it. Observed live, an answer about tensors ended
        # with "The result of the calculation 4-5 is -1" from the turn before.
        #
        # No length filter here. Above, len > 2 keeps "3 sources" from firing on
        # every turn; here the signal is "this exact number is in an old answer
        # of mine and in none of my observations", which is specific enough on
        # its own - and it has to be, because the observed leak was "-1".
        earlier = " ".join(m["content"] for m in self.history[:-1]
                           if m["role"] == "assistant")
        carried = {n for n in (m.rstrip(".,") for m in re.findall(r"-?\d[\d.,]*", answer))
                   if n and n != "-" and n in earlier and n not in seen}
        if carried:
            self._log("carried over from an earlier answer, in no observation: "
                      + ", ".join(sorted(carried)[:6]))

    def _check_arithmetic(self, user_message: str, used: set[str]) -> None:
        """Warn when the user wrote a calculation and `calculate` was never called.

        Advisory, --verbose only; see ARITHMETIC_RE for why this is not a gate.
        Observed live: "weather in Elazig, then what is 4 - 5?" answered the
        arithmetic in prose. The directive says to call the tool, but in a
        compound request the arithmetic is a subordinate clause and loses -
        the same shape as defect G, side instructions being dropped.
        """
        if not self.verbose or "calculate" in used:
            return
        if ARITHMETIC_RE.search(DATE_LIKE_RE.sub(" ", user_message)):
            self._log("the user wrote a calculation and calculate was never called "
                      "- the arithmetic in this answer is unverified")

    def _generate(self, prompt: str, **kw) -> str:
        """Single entry to the model: streams if asked, always logs timings."""
        on_token = _ProseStream(self.on_text) if self.on_text else None
        result = self.model.generate(prompt, on_token=on_token, **kw)
        self._log_timings(result.get("timings") or {})
        return result["text"].strip()

    def _log_timings(self, t: dict) -> None:
        """Prompt processing is this project's bottleneck, so show it.

        Measured at ~175 tok/s against ~40 tok/s of generation: every 1000
        tokens of context costs about six seconds on EVERY later turn. The
        server reports it on every request and it used to be thrown away.
        """
        if not self.verbose or not t:
            return
        self._log("prompt {:.0f} tok {:.1f}s ({:.0f} t/s) · gen {:.0f} tok {:.1f}s ({:.0f} t/s)"
                  .format(t.get("prompt_n", 0), t.get("prompt_ms", 0) / 1000,
                          t.get("prompt_per_second") or 0,
                          t.get("predicted_n", 0), t.get("predicted_ms", 0) / 1000,
                          t.get("predicted_per_second") or 0))

    def _log(self, *a):
        if self.verbose:
            print("  ·", *a, file=sys.stderr)

    # -- main loop ---------------------------------------------------------
    def answer(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        already_called: set[str] = set()
        used_tools: set[str] = set()

        for step in range(self.max_iterations):
            last = step == self.max_iterations - 1
            out = self._generate(self.build_prompt(self.history),
                                grammar=self.grammar, stop=[EOT])
            calls = self.extract_calls(out)

            if not calls:
                self.history.append({"role": "assistant", "content": out})
                self._check_numbers(out)
                self._check_arithmetic(user_message, used_tools)
                return out                                   # final answer

            self.history.append({"role": "assistant", "content": out})
            lang_name = LANGUAGE_NAMES[prompts.resolve_language(self.history, self.language)]
            # Grounding is judged against the conversation as it stood BEFORE
            # this generation. Every call in the batch was written without
            # seeing any of their results, so a result produced here cannot be
            # what grounded a sibling call.
            prior = self.history[:-1]
            if len(calls) > 1:
                self._log(f"{len(calls)} calls in one generation")

            for i, (name, args, parse_error) in enumerate(calls):
                if parse_error:
                    result = {"error": "invalid_tool_call", "message": parse_error}
                    outcome = "error:invalid_tool_call"
                elif (problems := self._ungrounded(name, args, prior)):
                    # Do NOT dispatch: ask the user instead of acting on invented input.
                    result = {"error": "missing_information",
                            "message": "Do not guess these values. Ask the user for them "
                                        f"in {lang_name}: " + "; ".join(problems)}
                    outcome = "blocked:missing_information"
                elif (signature := f"{name}:{json.dumps(args, sort_keys=True)}") in already_called:
                    result = {"error": "repeated_call",
                            "message": "You already made this exact call. Use the result "
                                        "you already have, or answer the user."}
                    outcome = "blocked:repeated_call"
                else:
                    already_called.add(signature)
                    used_tools.add(name)
                    result = tools.dispatch(name, args)
                    # Log the OUTCOME, not the request: the tool overrides some
                    # arguments (google_search treats num_results as a floor), so
                    # logging the model's args alone is misleading.
                    if "error" in result:
                        outcome = f"error:{result['error']}"
                    elif "source_count" in result:
                        provider = result.get("provider", "?").lstrip("_")
                        outcome = f"{result['source_count']} sources ({provider})"
                    else:
                        outcome = "ok"
                # Every branch is logged, not just the dispatched one. A call
                # that is blocked or unparseable is exactly the case where the
                # transcript looks like the model ignored the user.
                self._log(f"{name or '?'}({args}) -> {outcome}")

                # The note goes on the LAST observation of the batch only;
                # once per call would tell the model "this was the last call"
                # several times in a row.
                if last and i == len(calls) - 1:
                    result = dict(result, note="This was the last tool call allowed. "
                                            f"Answer the user in {lang_name} now.")
                self.history.append({"role": "ipython", "tool": name or "unknown",
                                    "content": self.observation(name or "unknown", result)})
            self._fit_context(self.history)

        # Budget exhausted and still calling tools: one forced final answer.
        out = self._generate(self.build_prompt(self.history), stop=[EOT])
        if any(name for name, _, _ in self.extract_calls(out)):
            out = GIVE_UP[prompts.resolve_language(self.history, self.language)]
            if self.on_text:
                self.on_text(out)   # the suppressed call left the screen empty
        self.history.append({"role": "assistant", "content": out})
        return out

    def reset(self) -> None:
        self.history.clear()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Turkish ReAct agent (REPL)")
    ap.add_argument("--url", default=serve.DEFAULT_URL)
    ap.add_argument("--no-grammar", action="store_true",
                    help="disable GBNF constrained decoding (on by default)")
    ap.add_argument("--verbose", action="store_true", help="log tool calls to stderr")
    ap.add_argument("--variant", default=prompts.DEFAULT_VARIANT, choices=list(prompts.VARIANTS),
                    help="system prompt variant")
    ap.add_argument("--lang", default=prompts.DEFAULT_LANGUAGE or "auto", choices=["en", "tr", "auto"],
                    help="answer language; 'auto' mirrors the user (Turkish is suspended by default)")
    ap.add_argument("--no-stream", action="store_true",
                    help="wait for the whole answer instead of streaming it as it is written")
    a = ap.parse_args()

    # Arrow-key history and line editing in the REPL. Importing the module is
    # all it takes; input() picks it up. Absent on some builds, hence the guard.
    try:
        import readline           # noqa: F401
    except ImportError:
        pass

    try:
        # a.lang is passed through verbatim: "en"/"tr" pin the language, "auto"
        # is the sentinel prompts.resolve_language() reads as "mirror the user".
        # It was previously parsed, printed in the banner and then dropped, so
        # --lang tr silently ran in English while the banner said lang=tr.
        agent = Orchestrator(serve.Model(a.url), use_grammar=not a.no_grammar,
                            verbose=a.verbose, variant=a.variant, language=a.lang,
                            on_text=None if a.no_stream
                                    else lambda piece: print(piece, end="", flush=True))
    except serve.ServerUnavailable as e:
        raise SystemExit(str(e))

    print(f"n_ctx={agent.model.n_ctx}  context budget={agent.budget} tokens  "
        f"grammar={'on' if agent.grammar else 'off'}  prompt={a.variant}  lang={a.lang}  "
        f"stream={'off' if a.no_stream else 'on'}")
    print("Type your question (Ctrl+D to quit, /reset to clear the history).\n")
    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line == "/reset":
            agent.reset()
            print("history cleared\n")
            continue
        # A failed turn must not take the conversation with it: the history is
        # the expensive part, and losing it to one timeout is worse than the
        # timeout. Ctrl+C now cancels the turn rather than the session.
        try:
            if agent.on_text:
                print("agent> ", end="", flush=True)
                agent.answer(line)
                print("\n")
            else:
                print(f"agent> {agent.answer(line)}\n")
        except KeyboardInterrupt:
            print("\n[cancelled - history kept]\n")
        except serve.ServerUnavailable as e:
            print(f"\n[{e}]\n", file=sys.stderr)
        except Exception as e:                       # noqa: BLE001 - the REPL must survive
            print(f"\n[{type(e).__name__}: {e}]\n", file=sys.stderr)



if __name__ == "__main__":
    main()
