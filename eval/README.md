# eval/

The evaluation layer required by spec sections 5.3 and 5.4.

## test_set.jsonl

**86 records, written by hand.** 26 of them are built on tool schemas the model has **never seen** in
training. The set is held out and never mixed back in.

**Why it has to exist.** It is the only way to separate *"the model learned to read a JSON schema and build
the call"* from *"the model memorized the tool names it was trained on"*. Measured: **24/26 (92%)** on
unseen schemas against **48/56 (85%)** on seen ones. That the gap essentially closed is this project's
main finding.

Every tool name claimed as unseen was checked against all 2,985 tool names in the training data — intuitive
picks like `convert_currency` and `translate_text` turned out to be present and were replaced. The set was
also screened for verbatim overlap and at a 0.70 similarity threshold; five real leaks were found and
rewritten.

Category spread (`validate_test_set.py` prints it on every run):

| category | count | what it measures |
|:--|:--|:--|
| `unseen_positive` / `unseen_negative` | 11 / 5 | correct call from an unseen schema / staying frugal with one on the table |
| `seen_positive` / `seen_negative` | 6 / 8 | baseline inside the training distribution / not calling when nothing is needed |
| `boundary_positive` / `boundary_ambiguous` | 6 / 4 | the line between needing a call and not |
| `wrong_tool_trap` | 5 | keyword collisions that pull toward the wrong tool |
| `missing_argument` | 4 | asking instead of fabricating |
| `reasoning` | 5 | arithmetic routed to `calculate` |
| `observation_final` | 6 | the final turn after an observation |
| `forecast` | 5 | reading `days_ahead` out of the question |
| `explicit_search` / `query_language` | 4 / 3 | honouring an explicit search request / query language |
| `multi_call` / `multi_turn` | 2 / 3 | two calls in one turn / carrying context |
| `user_language_edge` | 2 | language detection on short or mixed messages |
| `compound_request` | 7 | several tasks in one message — is every one answered |

Each record carries its own `tools` bundle, the `messages` that set it up, an `expect` block naming the
call and arguments that count as correct, and a `scoring` mode — `strict` when exactly one behaviour is
right, `report` when more than one is defensible.

Records where more than one behaviour is defensible are **reported, not scored** (`scoring: report`).
Inventing a reference answer to make a metric look complete would have made the metric worse.

> [!WARNING]
> **The generator is gone.** `uret_test_set.py` lived in a `scratchpad/` directory that was deleted with its
> session. `test_set.jsonl` is now the source itself — it can be edited by hand, but
> `validate_test_set.py` must be run after **every** edit. That is exactly why the validator was moved into
> the repository.

## Files

| file | what it does |
|:--|:--|
| `test_set.jsonl` | the 86-record held-out set |
| `validate_test_set.py` | structural checks: schema/type/enum agreement, required-parameter coverage, role order, `ipython` double-encoding, training leakage, eval-vs-production schema drift, `INTENTIONAL_DRIFT` review. Every check here caught a real defect at least once. |
| `eval_post_quant.py` | scores the quantized GGUF model against the set |

```bash
python eval/validate_test_set.py                       # this first
python eval/eval_post_quant.py --grammar --out r.json  # then this
```

`--grammar` defaults to **off** here (it is on in production): the model's unaided behaviour has to be
measurable too. The grammar is built per record from **that record's own tools**, never from the production
registry — a fixed grammar forbids the unseen tools a record declares and drops `unseen_positive` to 0/11.
That is a harness bug, and it looks exactly like a model failure.

## ❌ The bf16 baseline cannot be taken

A baseline runner was drafted for the merged bfloat16 model, but the merged model was deleted by the
quantize pipeline **before** any measurement was taken, so it was never run and never could be — the draft
has since been removed. Every number here is therefore **absolute**, not relative to the pre-quantization
model: *"what did Q4_K_M cost?"* is unanswerable in this repository.

## `compound_request` — and the limit of a single-shot eval

It came out of two live failures: *"the weather, then what is 4 - 5?"* answered the arithmetic from memory;
*"search tensors and RAG"* never searched RAG and never mentioned it.

The eval is single-shot — it does not run the loop, so it cannot directly ask *"was the whole request
satisfied by the end of the turn?"*, because calling the second tool on a later iteration is also valid.
So each scenario is **two records**: a fresh one scoring the first generation (call count **reported**), and
a continuation that puts the first observation in the history and leaves exactly one correct move
(**strict**).

Two scoring fields came with it:

- **`args_contains`** — looks at the *subject* of a free-text argument. `must: {"query": "*"}` only says
  "not empty", and would pass a model that simply repeated its previous query.
- **`must_not_repeat`** — strings that must not appear in the output. Catches a previous turn's answer being
  carried into an unrelated one.

## ⚠️ The eval does not measure the agent's full prompt

`eval_post_quant.py` takes the **system prompt** from `inference/prompts.py`, so the variants have a single
source — but it does not append the **last-moment directive** the orchestrator renders immediately before
generation. So the `final_language` and `observation_final` numbers here are raw model behaviour measured
**without** the mechanism that actually enforces the language. That is a deliberate boundary: the eval
measures the model's own tendency, not the agent's. Adding the directive would require re-running every
measurement.
