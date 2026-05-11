"""Chainlit UI for Hiring Agents — v3 conversational agent."""

from __future__ import annotations

import json

import chainlit as cl
from langchain_core.messages import HumanMessage

from hiring_agents.agent.agent import get_agent
from hiring_agents.llm.tracing import create_trace_id, get_langchain_handler, observe_trace

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
        await cl.Message(content=_candidate_md(c)).send()


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
    if name in ("search_candidates", "show_more_candidates"):
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



@cl.on_message
async def on_message(message: cl.Message) -> None:
    session_id = cl.context.session.id
    config = {"configurable": {"thread_id": session_id}}
    agent = get_agent()
    response_msg: cl.Message | None = None
    tool_steps: dict[str, cl.Step] = {}

    trace_id = create_trace_id()
    handler = get_langchain_handler(trace_id=trace_id)
    if handler:
        config["callbacks"] = [handler]

    with observe_trace(
        name="conversation_turn",
        session_id=session_id,
        trace_id=trace_id,
        metadata={"user_message": message.content[:500]},
    ):
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


