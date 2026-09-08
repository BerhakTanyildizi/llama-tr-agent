# -*- coding: utf-8 -*-
"""Arithmetic tool.

Added after a live failure. Asked how much memory FP16 weights need in INT4,
the model answered "1 FP16 = 2 INT4, so 16 GB becomes 32 GB" - the ratio is 4,
the operation is division, and the answer was 4 GB. The next turn then reused
that wrong ratio as a premise and produced "256 billion 7B parameters fit in
32 GB". An 8B model is weak at multi-step numeric reasoning; that is a known
limit, not something a prompt fixes, so the arithmetic moves into code.

SAFETY: eval() is never used. The expression is parsed to an AST and only the
whitelisted node types below are evaluated, so a tool_call cannot run arbitrary
code no matter what the model emits.

UNITS: the first version took only an expression, and that fixed the arithmetic
without fixing the reasoning. Live, the model called it with '(16*1024)/4' and
reported "4,096 units", and with '32*1024/16' for a question whose answer is
about 16 billion. The ratio was right and the units were nonsense.

Two things address that here. The description states the byte constants, because
schema descriptions were measured to actually steer this model. And `unit` makes
the model name what it is counting - a step it skipped entirely when it could
emit a bare expression, and the same step it performed correctly on the one
occasion it reasoned in prose instead of calling the tool.
"""
import ast
import math
import operator

SCHEMA = {"type": "function", "function": {"name": "calculate", "description": "Evaluate an arithmetic expression and return the exact result. Use it for every calculation instead of computing in your head. For memory maths, work in BYTES: 1 GB = 1024**3 bytes, FP16 = 2 bytes per value, INT8 = 1, INT4 = 0.5.", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "The expression to evaluate, in a single consistent unit, e.g. '32 * 1024**3 / 2'."}, "unit": {"type": "string", "description": "What the result is measured in, e.g. 'bytes', 'GB', 'parameters'. Always give it, otherwise the number is meaningless."}}, "required": ["expression"]}}}

# The expression is composed by the model from the question, not copied out of
# the user's words, so grounding it against the conversation would reject it.
GROUNDED_PARAMS: frozenset[str] = frozenset()

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}

_FUNCS = {"abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10}

MAX_EXPONENT = 64   # 9**9**9 would otherwise hang the process


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow):
            exponent = _eval(node.right)
            if abs(exponent) > MAX_EXPONENT:
                raise ValueError(f"exponent {exponent} is too large")
            return _eval(node.left) ** exponent
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"unsupported element: {type(node).__name__}")


def run(expression: str, unit: str | None = None) -> dict:
    try:
        value = _eval(ast.parse(str(expression), mode="eval").body)
    except ZeroDivisionError:
        return {"error": "division_by_zero",
                "message": f"'{expression}' divides by zero."}
    except Exception as e:
        return {"error": "invalid_expression",
                "message": (f"Could not evaluate '{expression}': {e}. Allowed: numbers, "
                            f"+ - * / // % **, parentheses, and {', '.join(_FUNCS)}.")}
    if isinstance(value, float) and not math.isfinite(value):
        return {"error": "not_finite", "message": f"'{expression}' is not a finite number."}
    # Rounded so the model is not handed 4.000000000000001 to read out loud.
    out = {"expression": str(expression),
        "result": round(value, 10) if isinstance(value, float) else value,
        "unit": unit or "unspecified"}
    if not unit:
        out["message"] = ("No unit was given, so this number means nothing on its own. "
                        "Say what it counts, or call again with the unit.")
    return out
