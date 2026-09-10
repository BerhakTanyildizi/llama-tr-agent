# -*- coding: utf-8 -*-
"""System prompt: the single source of truth for orchestrator AND eval.

Both must build the prompt identically or the evaluation measures something the
product does not use. Defining it twice is exactly how that drift starts.

CALL_FORMAT and the first half of PREAMBLE are copied verbatim from the training
data; changing them moves the model off its training distribution. The language
clause is NOT - see LANGUAGE_CLAUSES below.

The VARIANTS differ only in how frugal the model is told to be about calling
tools. They exist because the base training data is skewed ~71:1 toward calling
a tool (a side effect of the loss masking: single-turn "answer directly" Hermes
examples were masked out entirely, so only ~97 examples ever taught restraint).
"""
import datetime, re

PREAMBLE = ("You are a function calling AI model. You are provided with function "
    "signatures within <tools></tools> XML tags. You may call one or more "
    "functions to assist with the user query. Don't make assumptions about "
    "what values to plug into functions.")

# The language clause is NOT part of the training preamble (checked: 889 Hermes
# examples carry no language sentence at all, and the 94 Turkish ones say
# "always write your final answer to the user in Turkish"). It is a project
# addition, so it is free to follow the configured language instead of naming
# both languages on every turn.
#
# Why that matters: the mirroring text used to be unconditional, so while
# DEFAULT_LANGUAGE = "en" the system prompt still said "if the user speaks
# Turkish, answer in Turkish" ~1500 tokens before a directive that says
# LANGUAGE: English. Any Turkish user message put the two in direct conflict.
# Recency means the directive wins, but a contradiction the model has to
# resolve is not a free thing to leave in the prompt.
#
# Keyed by the CONFIGURED language, never the per-turn resolved one: this text
# sits in the cached prefix, and making it vary per turn would invalidate the
# system prompt's KV cache on every language flip.
LANGUAGE_CLAUSES = {
    "en": (" After receiving the tool results, always answer the user in English, "
        "in natural and fluent language, regardless of the language of the tool "
        "output or of the user's message."),
    "tr": (" After receiving the tool results, always write your final answer to "
        "the user in Turkish, in natural and fluent language, regardless of the "
        "language of the tool output."),
    # Mirroring. The exact wording that shipped before this split, kept so the
    # eval keeps measuring the bytes it measured before.
    "auto": (" After receiving the tool results, always answer the user in the same "
                "language as the user's latest message. If the user speaks English, "
                "answer in English. If the user speaks Turkish, answer in Turkish. Do "
                "not switch languages unless the user explicitly asks you to."),
}

CALL_FORMAT = ("For each function call return a json object with function name and "
                "arguments within <tool_call></tool_call> XML tags.")

VARIANTS = {
# Training prompt as-is. Baseline: 90% overall, but calls a tool for half of
# the questions that need none.
"V0": "",

# One sentence of restraint. Measured 90% overall, wrong_tool_trap 3/5.
"V1": (" Only call a function when the answer requires real-time, external, "
        "or user-specific information that you cannot know. If you can answer "
        "from your own knowledge, answer directly in the user's language without calling "
        "any function."),

# Stronger restraint. Measured 93% overall, wrong_tool_trap 5/5.
# WITHDRAWN FROM PRODUCTION: in live use it suppressed an explicit search
# request ("Kuantum salıncak teorisi nedir, arar mısın?") because the phrase
# "definitions, explanations, science" routes such questions to a direct
# answer - and the model then invented a theory that does not exist.
"V2": (" Most user questions do NOT require a function call. Call a function "
        "ONLY for live data you cannot know: current weather, the current date "
        "or time, and facts that changed after your knowledge cutoff. For "
        "everything else - definitions, explanations, history, science, math, "
        "advice, opinions - answer directly in the user's language and do NOT call any "
        "function."),

# V2 plus two escape hatches, added after the live failure above. The
# asymmetry is deliberate: an unnecessary search costs latency, a skipped
# search costs a confidently wrong answer.
"V3": (" Most user questions do NOT require a function call: answer directly "
        "in the user's language whenever you reliably know the answer - history, science, "
        "math, definitions, advice and opinions. Call a function in three "
        "cases: (1) the answer depends on live data you cannot know, such as "
        "current weather, the current date or time, prices, news, or anything "
        "that changed after your knowledge cutoff; (2) the user explicitly "
        "asks you to search or look something up; (3) you do not recognise "
        "the subject well enough to answer confidently - in that case search "
        "instead of guessing. Never invent facts about a topic you do not know. "
        "IMPORTANT: When you perform a search, do not just list the resulting links. "
        "The search tool automatically extracts and includes the text content of the "
        "resulting pages. Read this content carefully, and combine this external "
        "information with your internal knowledge to provide a comprehensive, detailed, "
        "and directly useful answer to the user. "
        "The sources are in whatever language the web returned them in. Do NOT "
        "translate them sentence by sentence: read them, then write the answer "
        "from scratch in natural, grammatical prose in the user's language. Never "
        "leave half-translated phrases or foreign words inside a sentence when a "
        "normal word exists. "
        "Length: answer in at least four to six full sentences. Explain the concept, "
        "give the reason or mechanism behind it, and add a concrete example where it "
        "helps. A one-line definition is not an acceptable answer."),
}

DEFAULT_VARIANT = "V3"

# Written out by hand: strftime("%b") follows the OS locale and yields "Tem"
# under a Turkish locale, which breaks the Llama-3.1 template (CLAUDE.md item 6).
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def system_prompt(schemas: list[dict], variant: str = DEFAULT_VARIANT,
                today: datetime.date | None = None,
                language: str | None = None,
                profile: tuple[str, ...] = ()) -> str:
    """Builds the system turn. `today` is injectable so tests stay reproducible.

    `language` picks the LANGUAGE_CLAUSES entry: "en"/"tr" name one language,
    "auto" (and the None default) keep the mirroring wording the eval was
    measured against.

    `profile` is the /remember facts. With none - which is how eval calls it -
    the text is byte-identical to what every recorded score was measured on.
    """
    import json
    d = today or datetime.date.today()
    clause = LANGUAGE_CLAUSES.get(language or "auto", LANGUAGE_CLAUSES["auto"])
    text = (f"Cutting Knowledge Date: December 2023\n"
        f"Today Date: {d.day:02d} {MONTHS[d.month - 1]} {d.year}\n\n"
        + PREAMBLE + clause + VARIANTS[variant] + "\n<tools>\n"
        + json.dumps(schemas, ensure_ascii=False)
        + "\n</tools>\n" + CALL_FORMAT)
    return text + "\n\n" + profile_block(profile) if profile else text


# --- last-moment directive --------------------------------------------------
# Measured: Turkish final turns in the training data have a MEDIAN of 118
# characters and 95% are under 300. Because the two-layer masking made the
# Turkish subset the only unmasked source of final-turn behaviour, the model
# learned "a final answer is ~120 characters". Wording alone does not move a
# learned prior (same story as num_results=1); PLACEMENT does. This directive
# is rendered immediately before the assistant header, ~1500 tokens closer to
# the generation point than the system prompt, and it is where the language
# mirroring is enforced too - the model only switched language when told
# explicitly, so it is told explicitly on every turn.
_TR_CHARS = "çğıöşüÇĞİÖŞÜ"
_TR_WORDS = ("bir", "ve", "için", "ile", "bu", "şu", "var", "yok", "nasıl",
                "nedir", "mı", "mi", "mu", "ne", "bana", "daha")
_EN_WORDS = ("the", "and", "is", "are", "of", "to", "in", "that", "with",
                "for", "you", "your", "what", "how", "explain", "search")


def detect_language(text: str) -> str | None:
    """'tr', 'en', or None when there is no signal either way.

    Returning None matters. The old version fell back to "tr" on a tie, so
    "Yes" and "Just shut up" inside an all-English conversation were each
    handed a Turkish directive and the agent flip-flopped mid-thread.
    """
    low = text.lower()
    words = set(low.replace("?", " ").replace(",", " ").split())
    tr = sum(c in _TR_CHARS for c in text) + len(words & set(_TR_WORDS))
    en = len(words & set(_EN_WORDS))
    if tr == en:
        return None
    return "en" if en > tr else "tr"


# The length clause used to read "at least 5-8 full sentences ... a one or two
# sentence answer is not acceptable", unconditionally, on every turn. Because
# this directive sits closest to the generation point it WON - including over
# the user. Observed live: "Hi what a beauty day" drew a nine-sentence essay,
# and "You don't need to talk at length about everything" drew the same essay
# again. Depth is now proportional, and an explicit request for brevity wins.
DIRECTIVES = {
    "tr": ("Cevabını şimdi yaz. DİL: Türkçe. "
           "UZUNLUK: derinliği soruya göre ayarla. Açıklama istenen bir soruda "
           "kavramı açıkla, nedenini ver, somut bir örnek ekle. Selamlaşma, "
           "onay ya da sohbet cümlelerine kısa karşılık ver. Kullanıcı kısa "
           "konuşmanı istediyse KISA konuş - bu talimat onun isteğini geçersiz "
           "kılmaz. Sorudaki her kısıta harfiyen uy: istenen ton, istenen madde "
           "sayısı, istenen ayrıntı düzeyi. "
           "KAPSAMA: kullanıcı birden fazla şey sorduysa HEPSİNE cevap ver. "
           "Birini yapamadıysan hangisini ve neden yapamadığını söyle - sorunun "
           "bir kısmını sessizce atlama. Yalnızca ŞU AN sorulanı cevapla: önceki "
           "turda verdiğin bir cevabı tekrar etme ya da yeniden anlatma. "
           "SAYILAR: hesaplamadan ÖNCE birimleri ayrı bir satıra yaz - örneğin "
           "'1 GB = 1024**3 bayt; FP16 = değer başına 2 bayt' - sonra calculate "
           "aracını o birimlerdeki ifadeyle çağır ve sonucun neyi saydığını "
           "söyle. Birimsiz sayı cevap değildir. Önceki turda söylediğin bir "
           "sayıyı doğrulanmış kabul etme, yeniden hesapla."),
    "en": ("Write your answer now. LANGUAGE: English. "
           "LENGTH: match the depth to the question. When you are asked to "
           "explain something, explain it, give the reason behind it and add a "
           "concrete example. Answer greetings, acknowledgements and small talk "
           "briefly. If the user has asked you to be shorter, BE SHORTER - this "
           "instruction does not override them. "
           "Follow every constraint in the question literally: a requested tone, "
           "a requested number of items, a requested level of detail. "
           "COVERAGE: if the user asked for several things, answer EVERY one of "
           "them. If you could not do one of them, say which one and why - never "
           "drop a part of the question silently. Answer only what is being asked "
           "now: do not restate or re-answer something you already answered in an "
           "earlier turn. "
           "NUMBERS: before calculating anything, first write the units on their "
           "own line - for example '1 GB = 1024**3 bytes; FP16 = 2 bytes per "
           "value' - then call the calculate tool with an expression in those "
           "units and say what the result counts. A bare number with no unit is "
           "not an answer. Never treat a number you stated in an earlier turn as "
           "verified; recompute it."),
}


# An explicit request beats statistical detection and STICKS until the user
# asks again. Two reasons, both observed live: "İn english" was detected as
# Turkish (the Turkish 'İ' inflates the Turkish score), and a user who asks to
# switch languages means it for the rest of the conversation, not for one turn.
# Deliberately narrow patterns: "İngilizce nasıl öğrenilir?" is a question ABOUT
# English, not a request to answer in it.
_ASK_EN = re.compile(r"\bin english\b|\bspeak english\b"
                    r"|\bingilizce (olarak )?(cevap|yaz|anlat|konus|soyle)", re.I)
_ASK_TR = re.compile(r"\bin turkish\b|\bspeak turkish\b"
                    r"|\bturkce (olarak )?(cevap|yaz|anlat|konus|soyle)", re.I)
_FOLD = str.maketrans("ıİğĞşŞçÇöÖüÜ", "iigGsScCoOuU")


def language_preference(messages: list[dict]) -> str | None:
    """Most recent explicit language request, if any."""
    for m in reversed(messages):
        if m["role"] != "user":
            continue
        folded = m["content"].translate(_FOLD)
        if _ASK_EN.search(folded):
            return "en"
        if _ASK_TR.search(folded):
            return "tr"
    return None


# TURKISH IS CURRENTLY SUSPENDED (DEFAULT_LANGUAGE = "en").
# The agent is being validated in English first: every remaining defect found so
# far has been of the form "works in English, fails in Turkish" - days_ahead
# extraction from "yarın", explicit language requests, search query language.
# Isolating the language variable makes the English baseline measurable before
# Turkish is layered back on. Set DEFAULT_LANGUAGE = None to restore mirroring,
# or pass --lang tr / --lang auto.
DEFAULT_LANGUAGE: str | None = "en"


def resolve_language(messages: list[dict], force: str | None = None) -> str:
    """The answer language for this turn. Always returns 'en' or 'tr'.

    Precedence: forced mode > explicit user request > detected language of the
    last user message > the language of the most recent message that HAD a
    signal > "en". The walk backwards is what stops a bare "Yes" from resetting
    the language mid-conversation.

    `force` accepts "auto" as well as "en"/"tr". It needs its own value because
    None already means "nothing was forced, use DEFAULT_LANGUAGE" - and while
    DEFAULT_LANGUAGE is "en" that collision made --lang auto unreachable: it
    resolved to English exactly like the pinned default.

    Exposed separately from final_directive because the orchestrator needs the
    LANGUAGE, not the directive text, for its own user-facing strings and for
    the language it names in an observation. Deriving it twice is how those
    drift apart from the directive the model is actually reading.
    """
    lang = force or DEFAULT_LANGUAGE
    if lang == "auto":                       # mirror the user, as does None
        lang = None
    if lang is None:
        lang = language_preference(messages)
    if lang is None:
        for m in reversed(messages):
            if m["role"] == "user":
                lang = detect_language(m["content"])
                if lang:
                    break
    return lang or "en"


# WHERE THIS BLOCK GOES DECIDED MORE THAN WHAT IT SAYS.
#
# It used to be rendered beside the final directive, i.e. as the last thing in
# the prompt before the model speaks - chosen because item 11 measured that this
# model follows what sits next to the generation point, and because /remember
# stayed instant there (a fact added mid-conversation does not rewrite the
# cached prefix).
#
# Both reasons were real and the placement was still wrong. That slot is the
# strongest position in the prompt, and filling it with a list of facts on EVERY
# turn put the standing block in competition with the question. Measured live
# against the transcript that exposed it, 6 turns per cell:
#
#     profile position   stale answer leaked   turn derailed   memory usable
#     beside directive          5/6                2/6            24/24
#     in the system prompt      0/6                0/6            23/24
#     no profile at all         0/6                1/6             1/24
#
# The symptom was not subtle: asked for the latest Python version, the model
# opened with the iPhone answer from three turns earlier; told "no, the latest
# iPhone is 18", it replied "Hello! I'm glad you're here". With the facts in the
# cached prefix instead, both vanish and the memory still works - 23/24 against
# 24/24 is the same number. The bottom row is the control: turn the profile off
# and the failures also go, but so does the feature (1/24).
#
# So the profile is BACKGROUND, and background belongs where the rest of the
# background lives. The cost is real and small: /remember now invalidates the
# ~1100-token prefix once, about six seconds at the measured 175 tok/s, at the
# moment the user types it. The old placement paid instead on every single turn,
# with answers aimed at the wrong thing. Item 31 measured the same force from
# the other side - the profile winning over "what did I just tell you?" - and
# fixed it by ordering profile before directive. That was a smaller dose of this
# bug, and the ordering fix treated the symptom.
#
# No restriction on using the facts. The first version said "never recite them
# back and never treat them as the question", written against a model that
# opens every answer with "As I recall, your name is ...". But the one question
# a profile exists to answer - "what is my name" - is answered BY reciting a
# fact, so the clause forbade the behaviour it was added to support.
# "told you" is avoided on purpose: it collides with "what did I just tell you",
# which the model then answered from this list instead of the previous turn.
#
# WHY EVERY LINE CARRIES A LABEL. A fact is stored verbatim, in the voice the
# user typed it in - a voice that ADDRESSES the assistant. Replayed inside a
# system turn it is read in the assistant's voice, and both pronouns flip
# referent on the way:
#     "My name is Berhak"  -> the model claims the name as its own
#     "your name is NEXUS" -> the model files it as the user's name
# Observed live, exactly that pair, swapped: "What is your name?" -> "My name is
# Berhak", "What is my name?" -> "Your name is NEXUS". The old header made the
# second half worse rather than better, because "facts about the user" asserts
# that a fact addressed TO the assistant is about the user.
#
# Measured live against the four facts above, 8 turns per question at the REPL's
# own temperature of 0.5:
#     old header, lines unlabelled                       18/32
#     per-line label                                     29/32
#     per-line label + quoted  (shipped)                 31/32
# The label is the same lesson as items 11 and 31, a third time: the model
# follows what sits NEXT TO the text it applies. A pronoun rule stated once in a
# heading has to be carried across four lines; a label two words away does not.
# Measured separately at 5 turns per question, a heading that explained the
# pronouns but left the lines unlabelled scored 15/20 and answered "what is my
# name" 1/5 - WORSE than doing nothing - so the label was not a free guess.
#
# The quotes are worth their two characters twice over: they mark the fact as
# reported speech, which is what makes its pronouns someone else's, and they
# bound a hand-editable file's contents so a fact cannot read as prompt
# structure.
#
# REJECTED, AND EXPENSIVE TO REDISCOVER: re-anchoring the pronouns instead of
# labelling them - rendering the facts in the assistant's own voice as "Your
# name is Berhak" / "My name is NEXUS". It reads correctly to a human and scored
# 16/32, INVERTED: 0/8 on "what is your name", where the model answered "my name
# is Berhak". In a SYSTEM turn "Your name is X" is the conventional way to name
# the ASSISTANT, so the rewrite collided head-on with the strongest prior the
# model has about that position. The same convention is why the original bug's
# second half was not really a model error at all: "your name is NEXUS" WAS the
# assistant's name, and the old heading was the thing asserting otherwise.
#
# HARNESS NOTE: the first scorer for all of this failed any answer that
# mentioned the other party's name, so "Hi Berhak! I'm NEXUS" was counted wrong
# for "what is your name" - the model was right and the measurement was not. It
# understated every variant and understated the chatty ones most, which is to
# say it would have picked the wrong winner. Items 14, 15 and 17 again: grade
# the claim, not the presence of a token.
PROFILE_BLOCK = ("Facts the user asked you to remember, quoted in the user's own words "
                 "and each labelled with who it is about:\n{}")

# Whole words only: "im" would match inside "important" and "i" inside anything.
_FIRST_PERSON = re.compile(r"\b(i|i'm|im|me|my|mine|myself|we|our|us)\b", re.I)
_SECOND_PERSON = re.compile(r"\b(you|you're|your|yours|yourself)\b", re.I)

ABOUT_USER = "about the user"
ABOUT_ASSISTANT = "about you, the assistant"
NEUTRAL = "the user's note"


def attribute(fact: str) -> str:
    """Label one stored fact with who it is about, from its grammatical person.

    Only the unambiguous halves are named. A line carrying BOTH persons ("I
    want you to be brief") is about the user AND about the assistant, and one
    carrying NEITHER ("the deadline is Friday") is about neither; guessing on
    those would re-introduce, in miniature, the very lie this fixes - the old
    header called every line a fact about the user, including the one that was
    not. They get a truthful third label instead of a coin flip.

    This is deliberately not a classifier. It reads grammatical person, which
    is decidable, and declines to infer intent, which is not. Same reasoning as
    item 19, where a measured-but-leaky arithmetic gate was left uninstalled.
    """
    first, second = _FIRST_PERSON.search(fact), _SECOND_PERSON.search(fact)
    if first and not second:
        return ABOUT_USER
    if second and not first:
        return ABOUT_ASSISTANT
    return NEUTRAL


def profile_block(profile: tuple[str, ...]) -> str:
    """The remembered facts, labelled. Rendered into the system turn."""
    return PROFILE_BLOCK.format(
        "\n".join(f'- {attribute(fact)}: "{fact}"' for fact in profile))


def final_directive(messages: list[dict], force: str | None = None) -> str:
    """The last-moment directive, in the language resolve_language() picks.

    Nothing but the directive goes here any more. Whatever occupies this slot
    is what the model answers, so it holds the instruction for THIS turn and
    nothing standing - see the placement note above PROFILE_BLOCK.
    """
    return DIRECTIVES[resolve_language(messages, force)]
