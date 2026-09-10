# -*- coding: utf-8 -*-
"""GBNF generation.

Every bug this file guards against arrived disguised as a model failure:
the root that forbade plain answers, the single-call root, the fixed grammar
file that hid unseen tools, and the prose rule that made '<' unwritable.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import tools                                           # noqa: E402
from inference.grammar import generate_gbnf                           # noqa: E402


class Root(unittest.TestCase):
    def setUp(self):
        self.grammar = generate_gbnf.build()
        self.rules = {line.split(" ::= ")[0]: line.split(" ::= ", 1)[1]
                      for line in self.grammar.splitlines() if " ::= " in line}

    def test_a_plain_answer_is_representable(self):
        """The root once accepted only <tool_call>, forcing a call on every question."""
        self.assertIn("prose", self.rules["root"])

    def test_more_than_one_call_is_representable(self):
        self.assertIn("*", self.rules["root"].split("|")[0])

    def test_prose_may_contain_an_angle_bracket(self):
        """`[^<]+` made '<' unwritable: the model answered "3 <= 5" for "3 < 5",
        a different mathematical claim, and could not write code or markup."""
        self.assertNotEqual(self.rules["prose"].strip(), "[^<]+")
        self.assertIn('"<"', self.rules["prose"])

    def test_prose_still_cannot_start_with_an_angle_bracket(self):
        """The root stays decidable from the first character."""
        self.assertTrue(self.rules["prose"].strip().startswith("[^<]"))

    def test_prose_covers_newlines(self):
        """Multi-line answers must survive; this is why '.' is not used."""
        self.assertNotIn(".", self.rules["prose"].replace("[^<]", "").replace('"<"', ""))


class BuiltFromTheCallSite(unittest.TestCase):
    def test_all_registry_tools_are_offered(self):
        grammar = generate_gbnf.build()
        for name in tools.REGISTRY:
            self.assertIn(name, grammar, name)

    def test_a_subset_produces_a_grammar_limited_to_that_subset(self):
        """Building once from the global registry scored unseen_positive 0/11:
        records could not select the tools they declared."""
        only_calc = generate_gbnf.build([tools.REGISTRY["calculate"].SCHEMA])
        self.assertIn("calculate", only_calc)
        self.assertNotIn("get_weather", only_calc)

    def test_an_unseen_schema_is_honoured(self):
        schema = {"type": "function", "function": {
            "name": "book_flight", "description": "Book a flight.",
            "parameters": {"type": "object", "properties": {
                "destination": {"type": "string", "description": "City."},
                "seats": {"type": "integer", "description": "How many."}},
                "required": ["destination"]}}}
        grammar = generate_gbnf.build([schema])
        self.assertIn("book_flight", grammar)
        self.assertIn("destination", grammar)
        self.assertIn("seats", grammar)

    def test_enum_values_become_alternatives(self):
        grammar = generate_gbnf.build([tools.REGISTRY["get_weather"].SCHEMA])
        self.assertIn("celsius", grammar)
        self.assertIn("fahrenheit", grammar)

    def test_a_tool_without_parameters_is_valid(self):
        schema = {"type": "function", "function": {
            "name": "ping", "description": "Ping.",
            "parameters": {"type": "object", "properties": {}}}}
        self.assertIn("ping", generate_gbnf.build([schema]))


class Shape(unittest.TestCase):
    def test_rule_names_carry_no_underscores(self):
        """GBNF rule names accept letters, digits and dashes only."""
        for line in generate_gbnf.build().splitlines():
            if " ::= " in line:
                self.assertNotIn("_", line.split(" ::= ")[0], line)

    def test_every_referenced_rule_is_defined(self):
        import re
        grammar = generate_gbnf.build()
        defined, referenced = set(), set()
        for line in grammar.splitlines():
            if " ::= " not in line:
                continue
            name, body = line.split(" ::= ", 1)
            defined.add(name.strip())
            body = re.sub(r'"(?:[^"\\]|\\.)*"', " ", body)      # drop literals
            body = re.sub(r"\[(?:[^\]\\]|\\.)*\]", " ", body)   # drop char classes
            referenced |= set(re.findall(r"[A-Za-z][A-Za-z0-9-]*", body))
        self.assertEqual(referenced - defined, set())

    def test_the_checked_in_file_matches_the_registry(self):
        """tool_call.gbnf is a snapshot; production builds dynamically. Drift
        between them is a trap, so the file is kept honest here."""
        path = Path(generate_gbnf.__file__).with_name("tool_call.gbnf")
        if not path.exists():
            self.skipTest("no snapshot checked in")
        self.assertEqual(path.read_text(encoding="utf-8"), generate_gbnf.build(),
                         "run: python inference/grammar/generate_gbnf.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
