# -*- coding: utf-8 -*-
"""Language resolution and prompt assembly.

The rule this file defends: while DEFAULT_LANGUAGE is "en" the prompt must not
mention Turkish at all. The system prompt used to say "if the user speaks
Turkish, answer in Turkish" unconditionally, ~1500 tokens before a directive
saying LANGUAGE: English - a standing contradiction on every Turkish message.
"""
import datetime
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import prompts, tools                                  # noqa: E402

TR = [{"role": "user", "content": "Ankara'da hava nasıl?"}]
EN = [{"role": "user", "content": "What is the weather in Ankara?"}]


class ResolveLanguage(unittest.TestCase):
    def test_pinned_language_wins_over_the_user(self):
        self.assertEqual(prompts.resolve_language(TR, "en"), "en")
        self.assertEqual(prompts.resolve_language(EN, "tr"), "tr")

    def test_auto_mirrors_the_user(self):
        """"auto" needs its own value: None already means "use DEFAULT_LANGUAGE"."""
        self.assertEqual(prompts.resolve_language(TR, "auto"), "tr")
        self.assertEqual(prompts.resolve_language(EN, "auto"), "en")

    def test_none_falls_back_to_the_module_default(self):
        self.assertEqual(prompts.resolve_language(TR, None), prompts.DEFAULT_LANGUAGE)

    def test_an_explicit_request_beats_detection(self):
        msgs = TR + [{"role": "user", "content": "answer in english please"}]
        self.assertEqual(prompts.resolve_language(msgs, "auto"), "en")

    def test_a_signalless_message_does_not_reset_the_language(self):
        """"Yes" inside a Turkish thread used to flip the directive to English."""
        msgs = TR + [{"role": "assistant", "content": "..."}, {"role": "user", "content": "Yes"}]
        self.assertEqual(prompts.resolve_language(msgs, "auto"), "tr")

    def test_detect_language_returns_none_without_signal(self):
        self.assertIsNone(prompts.detect_language("Yes"))

    def test_result_is_always_a_real_language(self):
        self.assertIn(prompts.resolve_language([], "auto"), ("en", "tr"))


class Directive(unittest.TestCase):
    def test_directive_matches_the_resolver(self):
        for force in ("en", "tr", "auto", None):
            expected = prompts.DIRECTIVES[prompts.resolve_language(TR, force)]
            self.assertEqual(prompts.final_directive(TR, force), expected, force)

    def test_both_directives_carry_the_coverage_rule(self):
        """Added after a compound request lost half its task silently."""
        self.assertIn("COVERAGE", prompts.DIRECTIVES["en"])
        self.assertIn("KAPSAMA", prompts.DIRECTIVES["tr"])

    def test_coverage_rule_forbids_dropping_a_part_in_silence(self):
        self.assertIn("silently", prompts.DIRECTIVES["en"])

    def test_directive_forbids_restating_an_earlier_answer(self):
        """A previous turn's answer leaked into an unrelated one."""
        self.assertIn("earlier turn", prompts.DIRECTIVES["en"])
        self.assertIn("önceki turda", prompts.DIRECTIVES["tr"])


class SystemPrompt(unittest.TestCase):
    def test_english_prompt_never_mentions_turkish(self):
        self.assertNotIn("Turkish", prompts.system_prompt(tools.SCHEMAS, language="en"))

    def test_turkish_option_still_exists(self):
        self.assertIn("Turkish", prompts.system_prompt(tools.SCHEMAS, language="tr"))

    def test_the_eval_default_keeps_the_mirroring_wording(self):
        """eval_post_quant.py calls system_prompt() with no language argument.
        Changing what it gets would silently invalidate every recorded score."""
        text = prompts.system_prompt(tools.SCHEMAS)
        self.assertIn("If the user speaks Turkish, answer in Turkish", text)

    def test_date_is_locale_independent(self):
        """strftime("%b") yields "Tem" under a Turkish locale and breaks the template."""
        text = prompts.system_prompt(tools.SCHEMAS, today=datetime.date(2025, 7, 9))
        self.assertIn("Today Date: 09 Jul 2025", text)

    def test_tool_schemas_are_embedded(self):
        text = prompts.system_prompt(tools.SCHEMAS, language="en")
        for name in tools.REGISTRY:
            self.assertIn(name, text, name)

    def test_every_variant_builds(self):
        for variant in prompts.VARIANTS:
            self.assertIn("<tools>", prompts.system_prompt(tools.SCHEMAS, variant, language="en"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
