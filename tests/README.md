# tests/

Harness tests. **No server, no network, no third-party package** — 114 tests in about a hundredth of a
second.

```bash
python -m unittest discover -s tests -v       # everything
python -m unittest tests.test_grammar -v      # one module
python -m unittest tests.test_tools.Calculate -v
```

## Why they exist

`eval/` scores the **model**. Nothing scored the **harness** — and most of the bugs found in this project
lived there:

| bug | how it looked |
|:--|:--|
| The parser took the first call and dropped the rest | *"use 2 searches"* ran one → read as a model failure |
| The grammar root accepted only `<tool_call>` | A plain answer was impossible → read as over-triggering |
| The eval loaded the grammar from a fixed file | `unseen_positive` scored 0/11 → read as failed generalization |
| `prose ::= [^<]+` | The model wrote `3 ≤ 5` instead of `3 < 5` → read as bad arithmetic |
| A placeholder in an *optional* slot blocked the call | A correctly built `calculate` call was refused |
| The eval iterated a string argument's characters | `extra:1, extra:7, extra:*` in the report |
| A retired observation still showed its tool call | The model learned searching was pointless and stopped |

**Every one looked like the model failing. None of them was.** Each test here pins a bug that shipped at
least once.

## Modules

| file | covers |
|:--|:--|
| `test_orchestrator.py` | call parsing (single, multiple, malformed), dispatch, argument grounding, the required-vs-optional placeholder split, context pruning, directive placement, advisory warnings |
| `test_tools.py` | registry contract, schema validation, `dispatch` never raises, `calculate` safety (no `eval()`), locale-independent weekday, page extraction (content, not navigation menus) |
| `test_profile.py` | `/remember`: whose voice a fact is in, per-line labelling, the profile living in the prefix rather than the directive, and storage that never truncates the file |
| `test_prompts.py` | language resolution precedence (`en`/`tr`/`auto`/`None`), directive clauses, the system prompt's language clause, **the eval default staying byte-identical**, locale-independent dates |
| `test_grammar.py` | the root allowing prose and repeated calls, `prose` accepting `<`, generation from the call site's tool bundle, no undefined rules, snapshot drift |

## Rules

- **No network.** `get_weather` is tested only on the branches that return *before* any HTTP request
  (`days_ahead=99`, a bad type). `google_search` only through its pure helpers. A test that needs the
  network is skipped in CI and protects nothing.
- **No model.** `FakeModel` replays scripted generations: the loop really runs, the model does not.
- **Multi-turn where it matters.** A change that touches an observation is tested across several turns —
  a single-turn test once passed while the agent had stopped searching entirely.
- Every test name says what it protects; the reason lives in the docstring.

## If `test_grammar` fails

The `tool_call.gbnf` snapshot has fallen behind the registry:

```bash
python inference/grammar/generate_gbnf.py
```
