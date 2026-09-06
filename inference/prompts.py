# -*- coding: utf-8 -*-
"""System prompt: the single source of truth for orchestrator AND eval.

Both must build the prompt identically or the evaluation measures something the
product does not use. Defining it twice is exactly how that drift starts.

PREAMBLE and CALL_FORMAT are copied verbatim from the training data. Changing
them moves the model off its training distribution.

The VARIANTS differ only in how frugal the model is told to be about calling
tools. They exist because the base training data is skewed ~71:1 toward calling
a tool (a side effect of the loss masking: single-turn "answer directly" Hermes
examples were masked out entirely, so only ~97 examples ever taught restraint).
"""
import datetime, re

PREAMBLE = ("You are a function calling AI model. You are provided with function "
       "signatures within <tools></tools> XML tags. You may call one or more "
       "functions to assist with the user query. Don't make assumptions about "
       "what values to plug into functions. After receiving the tool results, "
       "always answer the user in the same language as the user's latest message. "
       "If the user speaks English, answer in English. If the user speaks Turkish, "
       "answer in Turkish. Do not switch languages unless the user explicitly asks "
       "you to.")

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
              today: datetime.date | None = None) -> str:
    """Builds the system turn. `today` is injectable so tests stay reproducible."""
    import json
    d = today or datetime.date.today()
    return (f"Cutting Knowledge Date: December 2023\n"
       f"Today Date: {d.day:02d} {MONTHS[d.month - 1]} {d.year}\n\n"
       + PREAMBLE + VARIANTS[variant] + "\n<tools>\n"
       + json.dumps(schemas, ensure_ascii=False)
       + "\n</tools>\n" + CALL_FORMAT)


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


def detect_language(text: str) -> str:
    """'tr' or 'en'. Deliberately crude: only used to pick a directive."""
    low = text.lower()
    words = set(low.replace("?", " ").replace(",", " ").split())
    tr = sum(c in _TR_CHARS for c in text) + len(words & set(_TR_WORDS))
    en = len(words & set(_EN_WORDS))
    return "en" if en > tr else "tr"


DIRECTIVES = {
    "tr": ("Cevabını şimdi yaz. DİL: Türkçe. UZUNLUK: en az 5-8 tam cümle. "
           "Kavramı açıkla, arkasındaki nedeni veya mekanizmayı ver, işe "
           "yarayan somut bir örnek ekle. Tek iki cümlelik yüzeysel cevap "
           "kabul edilmez."),
    "en": ("Write your answer now. LANGUAGE: English. LENGTH: at least 5-8 "
           "full sentences. Explain the concept, give the reason or mechanism "
           "behind it, and add a concrete example. A one or two sentence "
           "answer is not acceptable."),
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


def final_directive(messages: list[dict]) -> str:
    """Explicit request if the user made one, otherwise mirror the last message."""
    lang = language_preference(messages)
    if lang is None:
        son = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        lang = detect_language(son)
    return DIRECTIVES[lang]
