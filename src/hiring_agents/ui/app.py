"""Chainlit UI for Hiring Agents."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import chainlit as cl
import httpx

from hiring_agents.config import SENIORITY_VOCAB, UI_API_BASE_URL
from hiring_agents.data_gen.axes import LOCATIONS

_ANY = "Any"
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


async def _run_search(query: str, mode: str, hard_filters: dict | None = None) -> None:
    payload: dict[str, Any] = {"query": query, "mode": mode}
    if hard_filters:
        payload["hard_filters"] = hard_filters

    ranked: list[dict] = []
    retrieved_count = 0
    filters_relaxed = False
    step_closed = False
    step = cl.Step(name="Searching…")
    await step.__aenter__()

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{UI_API_BASE_URL}/search/stream", json=payload) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    event = json.loads(line)

                    if event["type"] == "normalized":
                        norm = event["data"]
                        loc = ", ".join(norm["hard_filters"].get("location_keywords") or []) or "—"
                        sen = ", ".join(norm["hard_filters"].get("seniority") or []) or "—"
                        step.output = (
                            f"**Skills:** {', '.join(norm['core_skills']) or '—'}  \n"
                            f"**Location:** {loc}  \n**Seniority:** {sen}"
                        )

                    elif event["type"] == "retrieved":
                        retrieved_count = event["count"]
                        filters_relaxed = event["filters_relaxed"]
                        step.name = f"Retrieved {retrieved_count}"
                        await step.__aexit__(None, None, None)
                        step_closed = True

                        if filters_relaxed:
                            await cl.Message(
                                content="⚠️ No results with original filters — seniority requirement was relaxed."
                            ).send()

                    elif event["type"] == "candidate":
                        c = event["data"]
                        ranked.append(c)
                        entry = {k: c[k] for k in (
                            "candidate_id", "current_title", "current_employer",
                            "location", "total_yoe", "score", "gaps", "suggestion",
                        )}
                        await cl.Message(
                            content=_candidate_md(c),
                            actions=[cl.Action(name="add_shortlist", value=c["candidate_id"], payload=entry, label="+ Shortlist")],
                        ).send()

    except httpx.HTTPStatusError as exc:
        await cl.Message(content=f"❌ API error {exc.response.status_code}").send()
        return
    except httpx.ConnectError:
        await cl.Message(
            content=f"❌ Cannot connect to API at {UI_API_BASE_URL}. Is `make api` running?"
        ).send()
        return
    finally:
        if not step_closed:
            await step.__aexit__(None, None, None)

    if not ranked:
        await cl.Message(content="No candidates matched the filters.").send()
        return

    await cl.Message(
        content=f"**{len(ranked)}** candidates · {retrieved_count} retrieved"
    ).send()

    shortlist = cl.user_session.get("shortlist") or []
    await cl.Message(
        content=f"Shortlist: **{len(shortlist)}** saved",
        actions=[
            cl.Action(name="export_shortlist", value="export", payload={}, label="Export CSV ↓"),
            cl.Action(name="clear_shortlist", value="clear", payload={}, label="Clear Shortlist ✕"),
        ],
    ).send()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Job Description Search",
            message="__jd__",
            icon="/public/article_person.svg",
        ),
        cl.Starter(
            label="Strict Search",
            message="__strict__",
            icon="/public/discover_tune.svg",
        ),
    ]


@cl.on_chat_start
async def on_start():
    cl.user_session.set("shortlist", [])
    cl.user_session.set("strict_step", None)
    cl.user_session.set("strict_job_title", "")
    cl.user_session.set("jd_next", False)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    content = message.content.strip()

    if content == "__jd__":
        cl.user_session.set("jd_next", True)
        await cl.Message(content="Paste your job description:").send()
        return

    if content == "__strict__":
        cl.user_session.set("strict_step", "job_title")
        await cl.Message(
            content="**Strict Search** — no filter relaxation.\n\nEnter job title:"
        ).send()
        return

    if cl.user_session.get("strict_step") == "job_title":
        cl.user_session.set("strict_job_title", content)
        cl.user_session.set("strict_step", "awaiting_location")
        await cl.Message(
            content="Select location:",
            actions=[
                cl.Action(name="select_location", value=loc, payload={}, label=loc) for loc in LOCATIONS
            ] + [cl.Action(name="select_location", value=_ANY, payload={}, label="Any")],
        ).send()
        return

    if cl.user_session.get("jd_next"):
        cl.user_session.set("jd_next", False)
        await _run_search(content, "jd")
        return

    # Reset any stale strict state if user types freely
    cl.user_session.set("strict_step", None)
    await _run_search(content, "keyword")


@cl.action_callback("select_location")
async def on_select_location(action: cl.Action) -> None:
    loc = action.value
    await cl.Message(
        content="Select seniority:",
        actions=[
            cl.Action(name="select_seniority", value=s, payload={}, label=s.capitalize())
            for s in SENIORITY_VOCAB
        ] + [cl.Action(name="select_seniority", value=_ANY, payload={}, label="Any")],
    ).send()
    cl.user_session.set("strict_location", loc)


@cl.action_callback("select_seniority")
async def on_select_seniority(action: cl.Action) -> None:
    cl.user_session.set("strict_step", None)
    job_title = cl.user_session.get("strict_job_title") or "software engineer"
    location = cl.user_session.get("strict_location") or _ANY
    seniority = action.value

    hard_filters: dict[str, Any] = {}
    if location != _ANY:
        hard_filters["location_keywords"] = [location]
    if seniority != _ANY:
        hard_filters["seniority"] = [seniority]

    await _run_search(job_title, "strict", hard_filters or None)


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
