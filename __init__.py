"""Advisor strategy for Hermes Agent.

A cheap executor model runs the conversation. When a turn needs real judgment, a
stronger *advisor* model reads the transcript and returns short guidance. The
executor keeps the turn; the main model is never swapped, so the cached prompt
prefix survives (Hermes' core invariant).

Three entry points:
  * ``advisor`` tool       the executor asks for advice itself
  * ``/advisor <q>``       you ask, from any surface
  * Jev gate (pre_llm_call) classifies each user message; above the threshold it
    either nudges the executor to call the tool (``nudge``) or fetches the advice
    up front and injects it into the turn (``auto``)
"""

from __future__ import annotations

import logging

try:
    from . import core
except ImportError:  # imported bare (pytest collecting the repo root as a package)
    core = None

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "advisor",
    "description": (
        "Consult a stronger senior model before committing. It reads this conversation and returns "
        "a short plan, risks, and corrections. Use it for hard reasoning, ambiguous requirements, "
        "architecture or debugging dead-ends, and anything irreversible (money, deletes, prod, "
        "outbound messages). Don't use it for routine work."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What you want advice on, plus what you've tried or are about to do.",
            }
        },
        "required": ["question"],
    },
}


def register(ctx) -> None:
    if core is None:
        raise ImportError("advisor must be loaded as a package")
    core.bind(ctx)
    ctx.register_tool(
        name="advisor", toolset="advisor", schema=TOOL_SCHEMA,
        handler=core.handle_tool, description="Consult the advisor model", emoji="🧭",
    )
    ctx.register_hook("pre_llm_call", core.on_pre_llm_call)
    ctx.register_command(
        "advisor", handler=core.handle_slash,
        description="Ask the advisor model about the current conversation", args_hint="<question>",
    )
    logger.info("advisor: registered tool, /advisor, pre_llm_call gate")
