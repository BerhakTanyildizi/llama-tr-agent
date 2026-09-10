# -*- coding: utf-8 -*-
"""ReAct loop invariants. No server, no network - a FakeModel replays scripted output.

Why these exist: the eval measures the MODEL. Nothing measured the HARNESS, and
the harness is where this project's bugs actually live - calls dropped by the
parser, a placeholder guard refusing correct calls, a grammar that forbade plain
answers, an eval that iterated the characters of a string. Every test below
locks down a bug that shipped at least once.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import orchestrator, prompts, tools                    # noqa: E402


def call_block(name, **args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'


class FakeModel:
    """Replays a list of generations. The last entry repeats if the loop asks again."""
    n_ctx = 8192

    def __init__(self, outputs, n_ctx=8192):
        self.outputs = list(outputs)
        self.n_ctx = n_ctx
        self.prompts = []
        self.i = 0

    def count_tokens(self, text):
        return len(text) // 4                      # deterministic stand-in

    def generate(self, prompt, on_token=None, **kw):
        self.prompts.append(prompt)
        out = self.outputs[min(self.i, len(self.outputs) - 1)]
        self.i += 1
        if on_token is not None:
            # The real client calls on_token once per server-sent chunk. A fake
            # that accepts the argument and never calls it would let a broken
            # stream pass, so it is emitted here in pieces - small ones, so a
            # tag like <tool_call> straddles a boundary the way it really does.
            for start in range(0, len(out), 5):
                on_token(out[start:start + 5])
        return {"text": out, "timings": {}}


def agent(outputs, **kw):
    kw.setdefault("use_grammar", False)
    return orchestrator.Orchestrator(FakeModel(outputs), **kw)


def observations(a):
    """The decoded content of every observation in the history, in order."""
    out = []
    for m in a.history:
        if m["role"] == "ipython":
            inner = json.loads(m["content"])
            out.append(json.loads(inner.split("\n")[1])["content"])
    return out


# ---------------------------------------------------------------- parsing ---
class ExtractCalls(unittest.TestCase):
    """`search` used to take the first block and drop the rest with no trace."""

    def test_plain_answer_is_not_a_call(self):
        self.assertEqual(orchestrator.Orchestrator.extract_calls("Tensors are arrays."), [])

    def test_single_call(self):
        calls = orchestrator.Orchestrator.extract_calls(call_block("calculate", expression="4-5"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "calculate")
        self.assertEqual(calls[0][1], {"expression": "4-5"})
        self.assertIsNone(calls[0][2])

    def test_every_block_is_returned(self):
        """The live "use 2 searches" failure: only the first survived."""
        text = call_block("google_search", query="a") + "\n" + call_block("google_search", query="b")
        calls = orchestrator.Orchestrator.extract_calls(text)
        self.assertEqual([c[1]["query"] for c in calls], ["a", "b"])

    def test_malformed_block_does_not_discard_its_siblings(self):
        text = call_block("calculate", expression="1+1") + '<tool_call>{"name":}</tool_call>'
        calls = orchestrator.Orchestrator.extract_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "calculate")
        self.assertIsNone(calls[0][2])
        self.assertIsNotNone(calls[1][2])

    def test_unclosed_tag_is_reported_not_ignored(self):
        calls = orchestrator.Orchestrator.extract_calls('text <tool_call>{"name": "x"')
        self.assertEqual(len(calls), 1)
        self.assertIn("never closed", calls[0][2])

    def test_arguments_as_a_bare_string_become_an_empty_dict(self):
        """The model emitted "arguments": "16 * 2" once; iterating it is nonsense."""
        text = '<tool_call>{"name": "calculate", "arguments": "16 * 2"}</tool_call>'
        self.assertEqual(orchestrator.Orchestrator.extract_calls(text)[0][1], {})


# ------------------------------------------------------------- dispatching ---
class Dispatching(unittest.TestCase):
    def test_every_call_gets_its_own_observation(self):
        a = agent([call_block("calculate", expression="1+1")
                   + "\n" + call_block("calculate", expression="2+2"), "done"])
        a.answer("add things")
        results = [o["result"] for o in observations(a)]
        self.assertEqual(results, [2, 4])

    def test_duplicate_call_is_blocked_not_dispatched(self):
        a = agent([call_block("calculate", expression="1+1")
                   + "\n" + call_block("calculate", expression="1+1"), "done"])
        a.answer("add")
        obs = observations(a)
        self.assertEqual(obs[0]["result"], 2)
        self.assertEqual(obs[1]["error"], "repeated_call")

    def test_a_blocked_call_does_not_stop_its_siblings(self):
        a = agent([call_block("get_weather", location="Paris")
                   + "\n" + call_block("calculate", expression="1+1"), "done"])
        a.answer("what is 1+1?")                      # Paris was never mentioned
        obs = observations(a)
        self.assertEqual(obs[0]["error"], "missing_information")
        self.assertEqual(obs[1]["result"], 2)

    def test_plain_generation_ends_the_turn(self):
        a = agent(["Just an answer."])
        self.assertEqual(a.answer("hi"), "Just an answer.")
        self.assertEqual(observations(a), [])

    def test_loop_gives_up_instead_of_looping_forever(self):
        a = agent([call_block("calculate", expression="1+1")], max_iterations=2)
        out = a.answer("go")
        self.assertEqual(out, orchestrator.GIVE_UP["en"])


# --------------------------------------------------------------- grounding ---
class Grounding(unittest.TestCase):
    def setUp(self):
        self.a = agent(["x"])
        self.msgs = [{"role": "user", "content": "What is the weather?"}]

    def test_city_the_user_never_named_is_refused(self):
        problems = self.a._ungrounded("get_weather", {"location": "Ankara"}, self.msgs)
        self.assertTrue(problems)

    def test_city_the_user_named_is_accepted(self):
        msgs = [{"role": "user", "content": "Weather in Ankara?"}]
        self.assertEqual(self.a._ungrounded("get_weather", {"location": "Ankara"}, msgs), [])

    def test_diacritics_are_folded_before_comparison(self):
        """User types Elazig, the model correctly normalises to Elazığ."""
        msgs = [{"role": "user", "content": "Elazigda hava nasil?"}]
        self.assertEqual(self.a._ungrounded("get_weather", {"location": "Elazığ"}, msgs), [])

    def test_a_remembered_fact_grounds_the_call(self):
        """The profile is the user's words - /remember is typed by the user and
        never written by the model (item 33). Leaving it out of the haystack was
        not a stricter check but a wrong one: with "i am from Elazig/Turkey"
        remembered, "the weather in my city" built a correct
        get_weather(location='Elazig') and the guard refused it, so the agent
        asked the user for a city it had been told to remember."""
        a = agent(["x"], profile=("i am from Elazig/Turkey",))
        msgs = [{"role": "user", "content": "How is the weather in my city"}]
        self.assertEqual(a._ungrounded("get_weather", {"location": "Elazig"}, msgs), [])

    def test_a_profile_without_the_value_still_blocks(self):
        """The guard must not have been loosened into uselessness: with no city
        remembered the model still reaches for 'Ankara' (item 8) and is stopped.
        Measured live at 6/6 blocked."""
        a = agent(["x"], profile=("My name is Berhak", "I am a software engineer"))
        self.assertTrue(a._ungrounded("get_weather", {"location": "Ankara"}, self.msgs))

    def test_placeholder_in_a_required_slot_blocks(self):
        problems = self.a._ungrounded("calculate", {"expression": "unknown"}, self.msgs)
        self.assertTrue(problems)

    def test_placeholder_in_an_optional_slot_is_dropped_not_blocked(self):
        """calculate.unit is optional and "4-5" has no unit; n/a must not refuse the call."""
        for value in ("n/a", "none", "null"):
            args = {"expression": "4-5", "unit": value}
            self.assertEqual(self.a._ungrounded("calculate", args, self.msgs), [], value)
            self.assertNotIn("unit", args, value)

    def test_a_real_unit_survives(self):
        args = {"expression": "4-5", "unit": "bytes"}
        self.assertEqual(self.a._ungrounded("calculate", args, self.msgs), [])
        self.assertEqual(args["unit"], "bytes")

    def test_the_model_cannot_launder_its_own_invention(self):
        """Assistant turns are excluded from the grounding pool on purpose."""
        msgs = [{"role": "user", "content": "What is the weather?"},
                {"role": "assistant", "content": "Shall I check Ankara?"}]
        self.assertTrue(self.a._ungrounded("get_weather", {"location": "Ankara"}, msgs))


# ----------------------------------------------------------------- context ---
class Context(unittest.TestCase):
    def test_observation_uses_the_double_encoded_training_shape(self):
        raw = orchestrator.Orchestrator.observation("get_weather", {"temperature": 20})
        inner = json.loads(raw)                       # one layer of JSON string
        self.assertTrue(inner.startswith("<tool_response>\n"))
        self.assertTrue(inner.endswith("\n</tool_response>"))
        body = json.loads(inner[len("<tool_response>\n"):-len("\n</tool_response>")])
        self.assertEqual(body["name"], "get_weather")

    def test_pruning_keeps_the_most_recent_observation(self):
        a = agent(["x"], )
        a.model.n_ctx = 64                            # force the budget to bite
        a.budget = 16
        a.history = [{"role": "user", "content": "q"}]
        for i in range(3):
            a.history.append({"role": "assistant", "content": f"call {i}"})
            a.history.append({"role": "ipython", "tool": "calculate",
                              "content": orchestrator.Orchestrator.observation(
                                  "calculate", {"result": i, "padding": "x" * 400})})
        a._fit_context(a.history)
        obs = [m for m in a.history if m["role"] == "ipython"]
        self.assertTrue(obs[0].get("pruned"), "the oldest observation should be pruned")
        self.assertFalse(obs[-1].get("pruned"), "the newest observation must survive")

    def test_the_directive_is_appended_after_the_history(self):
        """It must sit past the cached prefix, closest to the generation point."""
        a = agent(["x"])
        p = a.build_prompt([{"role": "user", "content": "hello"}])
        self.assertLess(p.index("hello"), p.index("Write your answer now"))
        self.assertTrue(p.rstrip().endswith("<|end_header_id|>"))


# ------------------------------------------------------------- advisories ---
class Advisories(unittest.TestCase):
    def test_arithmetic_warning_is_silent_without_verbose(self):
        a = agent(["x"], verbose=False)
        self.assertIsNone(a._check_arithmetic("what is 4 - 5?", set()))

    def test_arithmetic_pattern_ignores_dates_and_ranges(self):
        for text in ("the meeting is on 2022-05-15", "compare 2020-2024 sales",
                     "tell me about GPT-4 and Llama-3.1"):
            cleaned = orchestrator.DATE_LIKE_RE.sub(" ", text)
            self.assertIsNone(orchestrator.ARITHMETIC_RE.search(cleaned), text)

    def test_arithmetic_pattern_catches_a_real_expression(self):
        for text in ("what is 4 - 5?", "compute 17*23", "how much is 32*1024/16?"):
            cleaned = orchestrator.DATE_LIKE_RE.sub(" ", text)
            self.assertIsNotNone(orchestrator.ARITHMETIC_RE.search(cleaned), text)


# ---------------------------------------------------------------- streaming ---
class ProseStreaming(unittest.TestCase):
    """The screen must never show the tool_call JSON, only the answer."""

    def drain(self, pieces):
        seen = []
        stream = orchestrator._ProseStream(seen.append)
        for piece in pieces:
            stream(piece)
        return "".join(seen)

    def test_prose_is_forwarded_whole(self):
        self.assertEqual(self.drain(["Ten", "sors ", "are ", "arrays."]),
                         "Tensors are arrays.")

    def test_a_tool_call_is_suppressed_entirely(self):
        self.assertEqual(self.drain(["<tool", "_call>{", '"name"', "}</tool_call>"]), "")

    def test_leading_whitespace_does_not_force_a_decision(self):
        """A chunk of pure whitespace must be held, not read as prose."""
        self.assertEqual(self.drain(["\n", "  ", "<tool_call>{}"]), "")
        self.assertEqual(self.drain(["\n", "  ", "Hello"]), "\n  Hello")

    def test_an_angle_bracket_later_in_prose_is_kept(self):
        """The grammar allows '<' after the first character; the stream must too."""
        self.assertEqual(self.drain(["3 is ", "less: 3 ", "< 5"]), "3 is less: 3 < 5")

    def test_streaming_matches_the_returned_text(self):
        seen = []
        a = agent(["Tensors are arrays."], on_text=seen.append)
        returned = a.answer("what is a tensor?")
        self.assertEqual("".join(seen), returned)

    def test_nothing_is_streamed_while_a_tool_runs(self):
        seen = []
        a = agent([call_block("calculate", expression="1+1"), "The answer is 2."],
                  on_text=seen.append)
        a.answer("what is 1+1?")
        self.assertEqual("".join(seen), "The answer is 2.")

    def test_buffered_mode_streams_nothing(self):
        a = agent(["plain"])
        self.assertIsNone(a.on_text)


class Timings(unittest.TestCase):
    def test_timings_are_silent_without_verbose(self):
        a = agent(["x"], verbose=False)
        self.assertIsNone(a._log_timings({"prompt_n": 10, "prompt_ms": 100}))

    def test_missing_timing_fields_do_not_crash(self):
        a = agent(["x"], verbose=True)
        self.assertIsNone(a._log_timings({}))
        self.assertIsNone(a._log_timings({"prompt_n": 5}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
