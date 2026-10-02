# hermes-advisor

The [advisor strategy](https://claude.com/blog/the-advisor-strategy) for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Run your conversations on a cheap, low-effort model. When a turn actually needs judgment, a stronger model (Opus at high effort by default) reads the conversation and hands back a short plan. Your executor keeps the turn.

**The main model never switches**, so Hermes' prompt cache stays warm. Model-swapping routers (`/model --once`, smart-routing patches) pay a full cache rebuild every time they escalate. This plugin does not.

## Three ways in

| | What happens |
|---|---|
| `advisor` tool | The executor calls it when it decides it's stuck or the stakes are high. |
| `/advisor <question>` | You ask, from CLI, desktop, Discord, Telegram... `/advisor status` shows config. |
| Jev gate (optional) | Each user message is scored by [Jev](https://typesafe.ai) (~0.5 s, one typed yes/no). Above the threshold it either nudges the executor to consult (`nudge`) or fetches the advice first and injects it into the turn (`auto`). |

The gate fails open: no key, a timeout, or an API error means no escalation, never a blocked turn.

## Install

```bash
hermes plugins install https://github.com/superlowburn/hermes-advisor --enable
# optional, for automatic escalation
hermes config set TYPESAFE_API_KEY <key>
```

## Configure

```yaml
plugins:
  entries:
    advisor:
      model: claude-opus-5-5     # advisor model ("" = your main model)
      provider: ""               # "" = your main provider
      reasoning_effort: high
      mode: auto                 # off | nudge | auto
      threshold: 0.7             # Jev probability needed to escalate
      min_message_chars: 25      # don't score "ok" / "thanks"
      transcript_chars: 60000    # tail of the conversation the advisor reads
      platforms: []              # e.g. [discord] to gate only Discord
```

Pair it with a cheap main model, e.g. `hermes config set model.default claude-sonnet-5-5` and low reasoning effort.

## What it costs

An escalation is one advisor call: transcript tail in, ~400 words out. A turn that doesn't escalate costs one Jev call (or nothing with `mode: off`). Advisor calls go through Hermes' own auxiliary client, so they use your configured auth, fallback, and usage accounting.

## Privacy

With the gate on, each user message over `min_message_chars` is sent to TypeSafe's API for scoring. The advisor sees the conversation tail and goes to whichever provider you configure.

MIT licensed.
