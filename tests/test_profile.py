# -*- coding: utf-8 -*-
"""The /remember profile: how a fact is stored, and whose voice it is in.

The bug this file exists for: a fact is stored verbatim, in the voice the user
typed it in - a voice that ADDRESSES the assistant. Replayed inside a system
turn it is read in the assistant's voice and both pronouns flip referent, so
"My name is Berhak" became the model's own name and "your name is NEXUS" became
the user's. Observed live, exactly that pair, swapped.

Nothing here talks to the server. eval measures the model; these measure the
harness, and the profile had no harness test at all before this - which is how a
prompt block that lied about half its lines shipped.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import orchestrator, prompts, session, tools           # noqa: E402
from test_orchestrator import FakeModel                              # noqa: E402

EN = [{"role": "user", "content": "What is my name?"}]
FACTS = ("My name is Berhak", "I am a software engineer", "your name is NEXUS")


class Attribution(unittest.TestCase):
    def test_first_person_is_about_the_user(self):
        for fact in ("My name is Berhak", "I am a software engineer",
                     "i am from Elazig/Turkey", "my cat is called Pamuk"):
            self.assertEqual(prompts.attribute(fact), prompts.ABOUT_USER, fact)

    def test_second_person_is_about_the_assistant(self):
        """The half the old heading actively got wrong: it called this a fact
        about the user, which is how "what is my name" answered "NEXUS"."""
        for fact in ("your name is NEXUS", "you are called NEXUS",
                     "You should answer briefly"):
            self.assertEqual(prompts.attribute(fact), prompts.ABOUT_ASSISTANT, fact)

    def test_ambiguous_and_impersonal_facts_are_not_guessed_at(self):
        """Naming a party here would repeat the original mistake in miniature."""
        for fact in ("I want you to be brief", "the deadline is Friday",
                     "prefers metric units"):
            self.assertEqual(prompts.attribute(fact), prompts.NEUTRAL, fact)

    def test_pronouns_are_matched_as_whole_words(self):
        """"im" inside "important" and "i" inside anything would misclassify."""
        self.assertEqual(prompts.attribute("shipping is important"), prompts.NEUTRAL)
        self.assertEqual(prompts.attribute("uses Yourdon notation"), prompts.NEUTRAL)


class Rendering(unittest.TestCase):
    def render(self, facts=FACTS):
        return prompts.system_prompt(tools.SCHEMAS, language="en", profile=facts)

    def test_the_profile_is_in_the_system_turn_not_the_directive(self):
        """THE placement regression. Beside the directive the facts occupy the
        last slot before the model speaks, and it answers whatever is there:
        measured 5/6 turns opened with a stale answer from three turns earlier
        and 2/6 derailed into "Hello! I'm glad you're here". From the cached
        prefix: 0/6 and 0/6, with the memory still usable (23/24 vs 24/24)."""
        directive = prompts.final_directive(EN, "en")
        self.assertEqual(directive, prompts.DIRECTIVES["en"])
        for fact in FACTS:
            self.assertNotIn(fact, directive)
            self.assertIn(fact, self.render())

    def test_every_line_carries_a_label(self):
        """Measured 18/32 -> 31/32 live. A rule stated once in the heading has
        to be carried across every line; a label two words away does not."""
        block = prompts.profile_block(FACTS)
        for line in block.splitlines():
            if line.startswith("- "):
                self.assertRegex(line, r"^- (about the user|about you, the assistant"
                                    r"|the user's note): ")

    def test_the_two_persons_land_on_different_labels(self):
        text = self.render()
        self.assertIn(f'- {prompts.ABOUT_USER}: "My name is Berhak"', text)
        self.assertIn(f'- {prompts.ABOUT_ASSISTANT}: "your name is NEXUS"', text)

    def test_facts_are_quoted_and_otherwise_untouched(self):
        """Quoting marks them as reported speech - that is what makes the
        pronouns someone else's - and bounds a hand-editable file's contents."""
        for fact in FACTS:
            self.assertIn(f'"{fact}"', self.render())

    def test_the_heading_no_longer_calls_every_line_a_user_fact(self):
        self.assertNotIn("Background facts about the user", self.render())

    def test_no_profile_leaves_both_halves_byte_identical(self):
        """eval_post_quant.py passes no profile to either; its recorded scores
        depend on both texts not moving."""
        self.assertEqual(prompts.final_directive(EN, "en"), prompts.DIRECTIVES["en"])
        self.assertEqual(prompts.system_prompt(tools.SCHEMAS, language="en"),
                        prompts.system_prompt(tools.SCHEMAS, language="en", profile=()))

    def test_remembering_rebuilds_the_prefix(self):
        """/remember has to take effect on the very next turn, and the facts now
        live in the prefix - so assigning .profile must rebuild the system turn."""
        a = orchestrator.Orchestrator(FakeModel(["x"]), use_grammar=False)
        self.assertNotIn("Berhak", a.system)
        a.profile = ("My name is Berhak",)
        self.assertIn("Berhak", a.system)


class Store(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "profile.txt"
        self.profile = session.Profile(self.path)

    def write(self, n):
        self.path.write_text("\n".join(f"fact {i}" for i in range(1, n + 1)) + "\n",
                            encoding="utf-8")

    def test_load_returns_the_whole_file(self):
        self.write(session.MAX_FACTS + 5)
        self.assertEqual(len(self.profile.load()), session.MAX_FACTS + 5)

    def test_for_prompt_caps_to_the_newest(self):
        self.write(session.MAX_FACTS + 5)
        carried = self.profile.for_prompt()
        self.assertEqual(len(carried), session.MAX_FACTS)
        self.assertEqual(carried[-1], f"fact {session.MAX_FACTS + 5}")

    def test_remembering_never_truncates_the_file(self):
        """load() capped at the first 20 and add() saved the last 20, so one
        /remember on a 25-line hand-edited file destroyed six facts in silence.
        The file is documented as hand-editable; rewriting it is the one thing
        this must not do."""
        self.write(session.MAX_FACTS + 5)
        before = self.profile.load()
        self.profile.add("a new fact")
        self.assertEqual(self.profile.load(), before + ["a new fact"])

    def test_forgetting_past_the_cap_keeps_the_rest(self):
        self.write(session.MAX_FACTS + 5)
        self.assertEqual(self.profile.remove(1), "fact 1")
        self.assertEqual(len(self.profile.load()), session.MAX_FACTS + 4)

    def test_a_write_that_did_not_happen_is_reported(self):
        """The REPL printed "remembered (4 facts)" either way, because it only
        ever saw the count. Same class of untruth as item 22."""
        self.profile.add("My name is Berhak")
        self.assertIsNotNone(self.profile.add("My name is Berhak")[1])
        self.assertIsNotNone(self.profile.add("   ")[1])
        self.assertIsNone(self.profile.add("I like tea")[1])

    def test_whitespace_is_collapsed_so_a_fact_stays_one_line(self):
        facts, _ = self.profile.add("  My   name\tis  Berhak ")
        self.assertEqual(facts, ["My name is Berhak"])

    def test_a_missing_file_is_not_an_error(self):
        self.assertEqual(self.profile.load(), [])
        self.assertEqual(self.profile.for_prompt(), [])
        self.assertIsNone(self.profile.remove(1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
