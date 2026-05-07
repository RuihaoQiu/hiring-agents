"""Chainlit UI for Hiring Agents — v3 conversational agent."""

from __future__ import annotations

import csv
import io
import json

import chainlit as cl
from langchain_core.messages import HumanMessage

from hiring_agents.agent.agent import get_agent

_FILLED = "●"
_EMPTY = "○"


def _dots(score: int, max_score: int = 5) -> str:
    return _FILLED * score + _EMPTY * (max_score - score)


def _candidate_md(c: dict) -> str:
    work = "\n".join(
        f"- **{e['title']}** at {e['company']} · {e['start_year']}–{e['end_year'] or 'present'}"
        for e in sorted(c["work_history"], key=lambda x: x["start_year"], reverse=True)
    )
    skills = "  ".join(f"`{s}`" for s in c["skills"]) or "—"
    return (
        f"**{c['candidate_id']}** · {c['current_title']} · {c['current_employer']}"
        f" · {c['location']} · {c['total_yoe']} yrs · {_dots(c['score'])}\n\n"
        f"{work}\n\n"
        f"**Skills:** {skills}"
    )


async def _render_candidates(raw: str) -> None:
    try:
        candidates = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return
    if not candidates:
        return

    if candidates[0].get("filters_relaxed"):
        await cl.Message(
            content="⚠️ No results with original filters — seniority requirement was relaxed."
        ).send()

    for c in candidates:
        entry = {k: c[k] for k in (
            "candidate_id", "current_title", "current_employer",
            "location", "total_yoe", "score", "gaps", "suggestion",
        )}
        await cl.Message(
            content=_candidate_md(c),
            actions=[cl.Action(name="add_shortlist", value=c["candidate_id"], payload=entry, label="+ Shortlist")],
        ).send()

    shortlist = cl.user_session.get("shortlist") or []
    await cl.Message(
        content=f"Shortlist: **{len(shortlist)}** saved",
        actions=[
            cl.Action(name="export_shortlist", value="export", payload={}, label="Export CSV ↓"),
            cl.Action(name="clear_shortlist", value="clear", payload={}, label="Clear Shortlist ✕"),
        ],
    ).send()


async def _handle_stream_token(
    event: dict, response_msg: cl.Message | None
) -> cl.Message | None:
    chunk = event["data"]["chunk"]
    token = chunk.content if isinstance(chunk.content, str) else ""
    if token:
        if response_msg is None:
            response_msg = cl.Message(content="")
            await response_msg.send()
        await response_msg.stream_token(token)
    return response_msg


async def _handle_tool_start(event: dict, tool_steps: dict[str, cl.Step]) -> None:
    name = event.get("name", "")
    run_id = event.get("run_id", "")
    step = cl.Step(name=name.replace("_", " ").title(), type="tool")
    await step.__aenter__()
    step.input = json.dumps(event["data"].get("input", {}), indent=2)
    tool_steps[run_id] = step


async def _handle_tool_end(event: dict, tool_steps: dict[str, cl.Step]) -> None:
    name = event.get("name", "")
    run_id = event.get("run_id", "")
    step = tool_steps.pop(run_id, None)
    output = event["data"].get("output", "")
    output_str = output.content if hasattr(output, "content") else str(output)
    if step:
        step.output = output_str[:500] + "…" if len(output_str) > 500 else output_str
        await step.__aexit__(None, None, None)
    if name == "search_candidates":
        await _render_candidates(output_str)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Find best candidates",
            message="Find me a senior Python backend engineer in Berlin with Kubernetes experience",
            icon="/public/group.svg",
        ),
        cl.Starter(
            label="Search by job description",
            message="I'd like to search by job description",
            icon="/public/article_person.svg",
        ),
    ]


@cl.on_chat_start
async def on_start():
    cl.user_session.set("shortlist", [])


@cl.on_message
async def on_message(message: cl.Message) -> None:
    config = {"configurable": {"thread_id": cl.context.session.id}}
    agent = get_agent()
    response_msg: cl.Message | None = None
    tool_steps: dict[str, cl.Step] = {}

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=message.content)]},
        config=config,
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            response_msg = await _handle_stream_token(event, response_msg)
        elif kind == "on_tool_start":
            await _handle_tool_start(event, tool_steps)
        elif kind == "on_tool_end":
            await _handle_tool_end(event, tool_steps)

    if response_msg:
        await response_msg.update()


@cl.action_callback("add_shortlist")
async def on_add_shortlist(action: cl.Action) -> None:
    shortlist: list[dict] = cl.user_session.get("shortlist") or []
    entry: dict = action.payload
    if any(c["candidate_id"] == entry["candidate_id"] for c in shortlist):
        await cl.Message(content=f"**{entry['candidate_id']}** is already in the shortlist.").send()
        return
    shortlist.append(entry)
    cl.user_session.set("shortlist", shortlist)
    await cl.Message(
        content=f"Added **{entry['candidate_id']}** to shortlist ({len(shortlist)} saved)."
    ).send()


@cl.action_callback("export_shortlist")
async def on_export_shortlist(action: cl.Action) -> None:
    shortlist: list[dict] = cl.user_session.get("shortlist") or []
    if not shortlist:
        await cl.Message(content="Shortlist is empty — add candidates first.").send()
        return

    fields = [
        "candidate_id", "current_title", "current_employer",
        "location", "total_yoe", "score", "gaps", "suggestion",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for c in shortlist:
        writer.writerow({**c, "gaps": "; ".join(c.get("gaps") or [])})

    await cl.Message(
        content=f"Exported **{len(shortlist)}** candidates.",
        elements=[cl.File(name="shortlist.csv", content=buf.getvalue().encode(), display="inline")],
    ).send()


@cl.action_callback("clear_shortlist")
async def on_clear_shortlist(action: cl.Action) -> None:
    cl.user_session.set("shortlist", [])
    await cl.Message(content="Shortlist cleared.").send()
