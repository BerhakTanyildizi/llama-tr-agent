# -*- coding: utf-8 -*-
"""Model access layer. The orchestrator talks to the model ONLY through this.

DELIBERATE DEVIATION FROM THE SPEC: the spec described this file as an
in-process llama-cpp-python Llama() object. This machine has no CUDA toolkit
(nvcc) and llama-cpp-python builds from source, while llama.cpp's prebuilt
Vulkan binaries already work. So the model runs as a separate llama-server
process and this file is an HTTP client to it.

What we lost: nothing. Every setting the spec listed exists as a server flag
(q8_0 KV cache, flash attention, -ngl 99) and prompt/prefix caching works over
HTTP (measured: 14 tokens processed instead of 440 on the second request).
What we gained: the model process survives across sessions, not just turns;
restarting a script no longer reloads the model.

MEASURED (RTX 4060 Laptop, Vulkan, Q4_K_M, n_ctx=8192):
weights 4403 MiB + KV cache 544 MiB + compute 108 MiB = 5055 / 7774 MiB
KV cost 68 KB/token | generation ~40 tok/s | prompt processing ~175 tok/s
-> ~2.7 GB free; n_ctx could go up to 32768.
The real bottleneck is not memory but PROMPT PROCESSING: at 175 tok/s every
1000 tokens costs ~6 seconds. That is why the orchestrator must trim context.
"""
import json, urllib.error, urllib.request

DEFAULT_URL = "http://127.0.0.1:8080"

START_COMMAND = """cd ~/llama-bin/llama-b10632 && LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./llama-server \\
-m ~/Desktop/MyAgent/training/outputs/gguf/llama31-8b-tr-Q4_K_M.gguf \\
-c 8192 -np 1 -ngl 99 --device Vulkan1 -fa on \\
--cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port 8080"""


class ServerUnavailable(RuntimeError):
    pass


class Model:
    def __init__(self, url: str = DEFAULT_URL, timeout: int = 600):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.n_ctx = self._props().get("n_ctx", 8192)

    def _request(self, path: str, body: dict | None = None, timeout: int | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.url + path, data=data,
                                    headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.URLError as e:
            raise ServerUnavailable(f"cannot reach llama-server ({self.url}): {e}\n"
                                    f"start it with:\n{START_COMMAND}") from None

    def _props(self) -> dict:
        p = self._request("/props", timeout=10)
        return p.get("default_generation_settings", p)

    def count_tokens(self, text: str) -> int:
        """For context budget tracking. Not an estimate - the model's own tokenizer."""
        return len(self._request("/tokenize", {"content": text}, timeout=30)["tokens"])

    def generate(self, prompt: str, max_tokens: int = 768, grammar: str | None = None,
                stop: list[str] | None = None, temperature: float = 0.5,
                top_k: int = 40, top_p: float = 0.9,
                repeat_penalty: float = 1.1, repeat_last_n: int = 256) -> dict:
        """Sampling defaults are for CHAT, not for measurement.

        At temperature 0 the agent locked into repeating one sentence verbatim
        across turns (observed live). Greedy decoding is right for eval, where
        reproducibility is the point - eval_post_quant.py sets its own
        temperature=0 and is unaffected by these defaults.

        repeat_penalty is the direct fix for that lock-up; top_k/top_p keep the
        sampling from wandering into low-probability tokens. 0.7 broke the loop
        but degraded Turkish grammar when summarising English sources, so the
        temperature was lowered to 0.5 - the anti-loop work is repeat_penalty's,
        not the temperature's.
        """
        body = {"prompt": prompt, "n_predict": max_tokens, "temperature": temperature,
                "top_k": top_k, "top_p": top_p, "repeat_penalty": repeat_penalty,
                "repeat_last_n": repeat_last_n,
                "cache_prompt": True, "stop": stop or []}
        if grammar:
            body["grammar"] = grammar
        c = self._request("/completion", body)
        return {"text": c["content"], "timings": c.get("timings", {})}
