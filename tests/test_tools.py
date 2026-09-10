# -*- coding: utf-8 -*-
"""Tool registry, schema validation and the pure parts of each tool.

Nothing here touches the network. get_weather is exercised only on the branches
that return before the first HTTP call; google_search only through its pure
helpers. A test that needed the internet would be skipped in CI and would
therefore protect nothing.
"""
import locale
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import tools                                           # noqa: E402
from inference.tools import calculate, get_system_time, get_weather   # noqa: E402
from inference.tools import google_search as gs                       # noqa: E402


class Registry(unittest.TestCase):
    def test_every_module_exposes_the_contract(self):
        for name, module in tools.REGISTRY.items():
            self.assertTrue(hasattr(module, "SCHEMA"), name)
            self.assertTrue(hasattr(module, "run"), name)
            self.assertTrue(hasattr(module, "GROUNDED_PARAMS"), name)
            self.assertEqual(module.SCHEMA["function"]["name"], name)

    def test_missing_required_parameter_is_rejected(self):
        self.assertIn("expression", tools.validate("calculate", {}))

    def test_unknown_parameter_is_rejected(self):
        self.assertIn("nonsense", tools.validate("calculate", {"expression": "1", "nonsense": 1}))

    def test_unknown_tool_is_rejected(self):
        self.assertIn("Unknown tool", tools.validate("teleport", {}))

    def test_enum_is_enforced(self):
        self.assertIsNotNone(tools.validate("get_weather", {"location": "X", "unit": "kelvin"}))
        self.assertIsNone(tools.validate("get_weather", {"location": "X", "unit": "celsius"}))

    def test_bool_is_not_accepted_as_an_integer(self):
        """isinstance(True, int) is True in Python - this trap was hit once in eval."""
        self.assertIsNotNone(tools.validate("get_weather", {"location": "X", "days_ahead": True}))

    def test_required_params_reads_the_schema(self):
        self.assertEqual(tools.required_params("calculate"), frozenset({"expression"}))
        self.assertEqual(tools.required_params("get_system_time"), frozenset())

    def test_grounded_params_are_declared_per_tool(self):
        self.assertIn("location", tools.grounded_params("get_weather"))
        self.assertEqual(tools.grounded_params("google_search"), frozenset())

    def test_dispatch_never_raises(self):
        """Every failure must come back as an observation the model can read."""
        for name, args in [("teleport", {}), ("calculate", {}),
                           ("calculate", {"expression": "1/0"}),
                           ("calculate", {"expression": "import os"})]:
            result = tools.dispatch(name, args)
            self.assertIsInstance(result, dict)
            self.assertIn("error", result)


class Calculate(unittest.TestCase):
    def test_arithmetic_is_exact(self):
        for expression, expected in [("4 - 5", -1), ("17*23", 391),
                                     ("16 * 1024**3 / 2", 8589934592.0)]:
            self.assertEqual(calculate.run(expression, "unit")["result"], expected, expression)

    def test_no_arbitrary_code_execution(self):
        for expression in ("__import__('os').system('echo hi')", "open('/etc/passwd')",
                           "[].__class__", "lambda: 1"):
            self.assertIn("error", calculate.run(expression), expression)

    def test_source_contains_no_bare_eval(self):
        import re
        source = Path(calculate.__file__).read_text(encoding="utf-8")
        code = "\n".join(l for l in source.splitlines() if not l.strip().startswith("#"))
        code = re.sub(r'""".*?"""', "", code, flags=re.S)
        self.assertIsNone(re.search(r"(^|[^_\w.])(eval|exec)\s*\(", code))

    def test_huge_exponent_is_refused_not_computed(self):
        self.assertIn("error", calculate.run("9**9**9"))

    def test_division_by_zero_is_named(self):
        self.assertEqual(calculate.run("1/0")["error"], "division_by_zero")

    def test_missing_unit_still_returns_the_number(self):
        result = calculate.run("17*23")
        self.assertEqual(result["result"], 391)
        self.assertIn("message", result)

    def test_unit_note_does_not_ask_the_user_for_anything(self):
        """The model relayed the old note verbatim: "Please provide the unit"."""
        message = calculate.run("4-5")["message"].lower()
        self.assertNotIn("call again", message)
        self.assertNotIn("provide", message)


class Weather(unittest.TestCase):
    """Only the branches that return before any HTTP request."""

    def test_day_beyond_the_horizon_is_refused_not_clamped(self):
        self.assertEqual(get_weather.run("Ankara", days_ahead=99)["error"], "day_out_of_range")

    def test_non_numeric_day_is_named(self):
        self.assertEqual(get_weather.run("Ankara", days_ahead="soon")["error"],
                         "invalid_days_ahead")

    def test_schema_documents_the_day_range(self):
        days = get_weather.SCHEMA["function"]["parameters"]["properties"]["days_ahead"]
        self.assertIn("7", days["description"])


class SystemTime(unittest.TestCase):
    def test_invalid_timezone_is_an_observation_not_a_crash(self):
        self.assertEqual(get_system_time.run("Mars/Olympus")["error"], "invalid_timezone")

    def test_weekday_is_english_whatever_the_os_locale_says(self):
        """strftime("%A") returned "Çarşamba" under tr_TR."""
        try:
            locale.setlocale(locale.LC_TIME, "tr_TR.UTF-8")
        except locale.Error:
            self.skipTest("tr_TR.UTF-8 locale not installed")
        try:
            self.assertIn(get_system_time.run()["weekday"], get_system_time.WEEKDAYS)
        finally:
            locale.setlocale(locale.LC_TIME, "C")


class SearchHelpers(unittest.TestCase):
    def test_markdown_chrome_is_stripped_and_prose_kept(self):
        raw = ("![logo](x.png)\nSkip to content\nInstagram Facebook-f Youtube\n"
               "[Home](/) » Explore\nThe fjords of Bergen are deep and cold.\n")
        self.assertEqual(gs._strip_markdown(raw), "The fjords of Bergen are deep and cold.")

    def test_domain_is_normalised(self):
        self.assertEqual(gs._domain("https://www.Example.com/a/b"), "example.com")

    def test_api_keys_are_redacted(self):
        self.assertNotIn("tvly-abcd1234efgh", gs._redact("error tvly-abcd1234efgh happened"))

    def test_a_bot_challenge_is_not_an_empty_result(self):
        """Reading HTTP 202 + CAPTCHA as "nothing found" made the agent lie."""
        self.assertTrue(gs._blocked(202, "please solve the following challenge"))
        self.assertTrue(gs._blocked(200, "<div>captcha</div>"))
        self.assertFalse(gs._blocked(200, "<html>real results</html>"))

    def test_num_results_is_a_floor(self):
        self.assertGreaterEqual(gs.MIN_RESULTS, 3)
        self.assertLessEqual(gs.MIN_RESULTS, gs.MAX_RESULTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
