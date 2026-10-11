"""
llm -- the only module that calls the Anthropic API.

* The configured model ID is checked against GET /v1/models and the run aborts on
  a mismatch. A model ID is NEVER substituted.
* Every call is priced from measured usage. An unpriced model aborts rather than
  costing $0.00; add it under [model.pricing] in project.toml.
* Cost is estimated before submission (input tokens counted exactly, output
  assumed) and the run aborts, having sent nothing, if the estimate exceeds the
  stage's max_usd.
"""

from __future__ import annotations

import sys
import time

try:
    import anthropic
except ImportError:
    anthropic = None

# USD per million tokens (base, non-batch); cache reads as a multiple of input.
PRICING = {"claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read_mult": 0.05}}
# Least thinking each model allows. Opus 5.5 cannot disable thinking (400 on
# `disabled`); effort=low is its only lever, and its thinking bills as output.
REQUEST_SHAPE = {"claude-opus-5-5": {"output_config": {"effort": "low"}}}
CACHE_WRITE_MULT, BATCH_MULT = 1.25, 0.5


class LLM:
    def __init__(self, cfg):
        if anthropic is None:
            sys.exit("anthropic is required:  pip install anthropic")
        m = cfg["model"]
        self.model = m["id"]
        self.price = {**PRICING.get(self.model, {}), **(m.get("pricing") or {})}
        if not {"input", "output"} <= set(self.price):
            sys.exit(f"no price for {self.model}: add [model.pricing] input/output "
                     f"(USD per million tokens) to project.toml")
        self.price.setdefault("cache_read_mult", 0.1)
        self.shape = REQUEST_SHAPE.get(self.model, m.get("request", {}))
        self.client = anthropic.Anthropic()
        try:
            served = {x.id for x in self.client.models.list()}
        except Exception as e:                                  # noqa: BLE001
            sys.exit(f"could not reach GET /v1/models to validate {self.model}: {e}")
        if self.model not in served:
            sys.exit(f"ABORT: model {self.model!r} is not served by this account; nothing "
                     f"submitted. Available: {', '.join(sorted(served))}. This tool never "
                     f"substitutes a model ID.")

    def cost(self, usage: dict, batch: bool) -> float:
        p, inp = self.price, self.price["input"] / 1e6
        return (BATCH_MULT if batch else 1.0) * (
            usage.get("input_tokens", 0) * inp
            + usage.get("cache_creation_input_tokens", 0) * inp * CACHE_WRITE_MULT
            + usage.get("cache_read_input_tokens", 0) * inp * p["cache_read_mult"]
            + usage.get("output_tokens", 0) * p["output"] / 1e6)

    def estimate(self, requests: dict[str, dict], out_tokens: int, batch: bool,
                 sample: int = 60) -> float:
        """Input counted exactly (on up to `sample` requests, extrapolated); output assumed."""
        ids = list(requests)
        step = max(1, len(ids) // sample)
        counted = [self.client.messages.count_tokens(
            model=self.model, system=requests[i]["system"], tools=requests[i]["tools"],
            messages=requests[i]["messages"]).input_tokens for i in ids[::step]]
        tokens_in = sum(counted) / len(counted) * len(ids)
        return (BATCH_MULT if batch else 1.0) * (
            tokens_in * self.price["input"] + len(ids) * out_tokens * self.price["output"]) / 1e6

    def params(self, system: str, tools: list[dict], user: str, max_tokens: int) -> dict:
        return {"model": self.model, "max_tokens": max_tokens,
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "tools": tools, "tool_choice": {"type": "auto"},
                "messages": [{"role": "user", "content": user}], **self.shape}

    def run(self, requests: dict[str, dict], batch: bool, on_result, poll: int = 30) -> None:
        """Send each request; call on_result(custom_id, message_or_None, error_or_None, batch)."""
        if not batch:
            for cid, p in requests.items():
                try:
                    with self.client.messages.stream(**p) as s:
                        on_result(cid, s.get_final_message(), None, False)
                except anthropic.APIError as e:
                    on_result(cid, None, f"api_error:{type(e).__name__}", False)
            return
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        b = self.client.messages.batches.create(requests=[
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**p))
            for cid, p in requests.items()])
        print(f"[batch] {b.id}: {len(requests)} requests")
        while (b := self.client.messages.batches.retrieve(b.id)).processing_status != "ended":
            c = b.request_counts
            print(f"  [batch] processing={c.processing} succeeded={c.succeeded} errored={c.errored}", flush=True)
            time.sleep(poll)
        for res in self.client.messages.batches.results(b.id):
            if res.result.type == "succeeded":
                on_result(res.custom_id, res.result.message, None, True)
            else:
                on_result(res.custom_id, None, f"batch_{res.result.type}", True)


def usage(message) -> dict:
    u = message.usage
    return {k: getattr(u, k, 0) or 0 for k in ("input_tokens", "output_tokens",
            "cache_creation_input_tokens", "cache_read_input_tokens")}


def tool_input(message, name: str) -> tuple[dict | None, str | None]:
    stop = getattr(message, "stop_reason", None)
    if stop in ("refusal", "max_tokens"):
        return None, stop
    blocks = [b for b in message.content or [] if getattr(b, "type", "") == "tool_use"
              and getattr(b, "name", "") == name]
    if len(blocks) != 1:
        return None, "no_tool_use" if not blocks else "multiple_tool_use"
    return (blocks[0].input, None) if isinstance(blocks[0].input, dict) else (None, "tool_input_not_object")


def confirm(n: int, est: float, cap: float, yes: bool) -> bool:
    print(f"[cost] estimated ${est:.2f} for {n} request(s); cap ${cap:g}")
    if est > cap:
        print(f"*** ABORT: estimate ${est:.2f} exceeds max_usd ${cap:g}. Nothing was submitted.")
        return False
    return yes or input(f"Submit {n} request(s) for ~${est:.2f}? [y/N] ").strip().lower() in ("y", "yes")
