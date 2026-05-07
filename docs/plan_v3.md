# Hiring Agents — v3 Plan

V3 reframes the system as a true conversational AI agent. Instead of a fixed search
UI with predefined modes, the recruiter talks freely to an agent that understands
requirements, decides when and how to search, presents results with commentary, and
refines the search through dialogue.

The retrieve → rerank pipeline becomes a tool the agent calls — not the primary
interface.

## What v3 is and isn't

**v3 is:**
- A LangGraph ReAct agent with tool calling, replacing the hand-coded routing in v2
- Free-form conversation: the recruiter describes the role naturally; the agent
  decides what to search for and how to filter
- Agent streams text responses token by token (like a chat assistant)
- Tool calls rendered as Chainlit Steps so the recruiter sees what the agent is doing
- Multi-turn memory: agent remembers candidates shown earlier in the conversation
- Shortlist managed via conversation ("add the first candidate to my shortlist")
- Same pipeline under the hood (normalize → retrieve → rerank) — called as a tool

**v3 is not:**
- A change to the pipeline or eval harness (those remain unchanged)
- Persistent storage across sessions (shortlist still lives in session state)
- Multi-agent orchestration
- A replacement for the FastAPI backend (kept for eval / CLI usage)

## Architecture

```
recruiter
   │ natural language
   ▼
Chainlit UI  ─── stream events ──► token-by-token text response
   │
   ▼
LangGraph ReAct agent
   │ tool calls
   ├─► search_candidates()  ──► normalize → retrieve → rerank (direct call, no HTTP)
   ├─► add_to_shortlist()   ──► cl.user_session
   ├─► get_shortlist()      ──► cl.user_session
   └─► export_shortlist()   ──► CSV bytes → cl.File
```

The agent runs inside the Chainlit app process. Tools call the pipeline functions
directly (not via HTTP), so no round-trip to FastAPI is needed for the UI path.
FastAPI is kept for eval and direct API access.

## Agent design

### System prompt

The agent acts as a recruiting assistant. Key behaviours:
- Ask one clarifying question if the role is too vague before searching
- Always explain what it searched for after calling a tool
- Present the top 3 candidates by name with a one-line reason each; offer to show more
- Suggest filter adjustments if results are weak
- Proactively offer to add candidates to the shortlist

### Tools

**`search_candidates(query, location, seniority, strict)`**
```
query:     str   — role description or full JD text
location:  str | None — city/region filter
seniority: list[str] | None — one or more of junior/mid/senior/staff/principal
strict:    bool = False — if True, skip filter relaxation on empty results

Returns: JSON list of top candidates (id, name, title, employer, location,
         yoe, score, skills, gaps, suggestion)
```
Internally: normalizes query, embeds, retrieves top-k, rerranks, returns ranked list.
The agent picks which fields to highlight in its prose response.

**`add_to_shortlist(candidate_id, candidate_name)`**
Appends to `cl.user_session["shortlist"]`. Returns confirmation string.

**`get_shortlist()`**
Returns the current shortlist as a formatted string. Agent uses this to answer
"what's in my shortlist?" without searching again.

**`clear_shortlist()`**
Resets `cl.user_session["shortlist"]` to `[]`.

### LangGraph setup

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_react_agent(
    model=ChatOpenAI(model=AGENT_MODEL, temperature=AGENT_TEMPERATURE, streaming=True),
    tools=[search_candidates, add_to_shortlist, get_shortlist, clear_shortlist],
    state_modifier=AGENT_SYSTEM_PROMPT,
    checkpointer=MemorySaver(),
)
```

Each Chainlit session maps to a LangGraph thread via `session.id`, giving the agent
memory across conversation turns within a session.

## Chainlit UI

`ui/app.py` simplifies significantly: no more mode routing, no more Starters
(or just one "Get started" hint), no more action callbacks for search modes.

```python
@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    msg = cl.Message(content="")
    await msg.send()

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=message.content)]},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                await msg.stream_token(token)

        elif kind == "on_tool_start":
            tool_name = event["name"]
            tool_input = event["data"].get("input", {})
            step = cl.Step(name=f"🔍 {tool_name}", type="tool")
            await step.__aenter__()
            step.input = str(tool_input)
            cl.user_session.set("current_step", step)

        elif kind == "on_tool_end":
            step = cl.user_session.get("current_step")
            if step:
                result = event["data"].get("output", "")
                step.output = _format_tool_output(result)
                await step.__aexit__(None, None, None)
                # If search result, render candidate cards
                if event["name"] == "search_candidates":
                    await _render_candidates(result)

    await msg.update()
```

Candidate cards are still rendered as `cl.Message` with `+ Shortlist` action
buttons. The shortlist export CSV button is shown after any search result.

The `_format_tool_output` helper truncates long search results for the Step
display (full data still goes to the agent context).

## Directory layout (delta from v2)

```
src/hiring_agents/
  agent/
    __init__.py
    agent.py       # create_react_agent, MemorySaver, agent singleton
    tools.py       # search_candidates, add_to_shortlist, get_shortlist, clear_shortlist
    prompts.py     # AGENT_SYSTEM_PROMPT
  ui/
    app.py         # rewritten: stream agent events, no mode routing
```

Everything else stays: `normalize.py`, `retrieve.py`, `rerank.py`, `graph.py`,
`api/`, `config.py`, eval harness, Chainlit branding.

## Config additions

```python
AGENT_MODEL: str = "gpt-4o"
AGENT_TEMPERATURE: float = 0.3
```

## Build order

### Step 1 — Agent tools

1. Create `src/hiring_agents/agent/tools.py`:
   - `search_candidates` as a LangChain `@tool` (async): calls normalize →
     retrieve → rerank directly (no HTTP), returns JSON string of ranked candidates
   - `add_to_shortlist`, `get_shortlist`, `clear_shortlist` as sync tools
     (access `cl.user_session` via Chainlit context)
2. Create `src/hiring_agents/agent/prompts.py`: `AGENT_SYSTEM_PROMPT`
3. Unit test: call `search_candidates.invoke({"query": "python engineer berlin"})`
   with ingested data, assert non-empty result.

**Milestone:** `search_candidates` tool returns ranked candidates when called
directly; other shortlist tools read/write session state correctly.

### Step 2 — LangGraph agent

1. Create `src/hiring_agents/agent/agent.py`: `build_agent()` returns compiled
   `create_react_agent` with all tools and `MemorySaver` checkpointer.
2. Add `AGENT_MODEL` / `AGENT_TEMPERATURE` to `config.py`.
3. Smoke test: run agent with a test query, assert it calls `search_candidates`
   exactly once and returns a non-empty response.

**Milestone:** agent correctly routes a free-text hiring query to the search tool
and returns a coherent text response with candidate names.

### Step 3 — Chainlit UI rewrite

1. Rewrite `src/hiring_agents/ui/app.py`:
   - `@cl.on_chat_start`: init shortlist, instantiate agent config
   - `@cl.on_message`: stream `agent.astream_events`, render text tokens +
     tool Steps + candidate cards
   - Keep `_candidate_md`, shortlist action callbacks, export CSV logic
   - Remove: mode routing, `_run_search`, `_call_api`, Starters for modes
2. Add a single Starter with a "Find candidates" prompt to give users a hint.
3. Manual smoke test: keyword query, JD paste, multi-turn refinement ("show
   more senior ones"), shortlist add + export.

**Milestone:** full end-to-end conversation works; agent streams text; tool calls
appear as Steps; candidate cards render with shortlist buttons; export CSV works.

## What stays vs. what changes

| Concern | v2 | v3 |
|---|---|---|
| Search modes | 3 fixed modes (keyword / jd / strict) | Agent decides based on context |
| Filter logic | Hard-coded in UI routing | Agent chooses query + filters per turn |
| UI entry point | Starters → mode routing | Free-form message |
| Response | Candidate cards only | Agent text + candidate cards |
| Streaming | HTTP NDJSON to UI | LangGraph `astream_events` |
| Memory | None (stateless per query) | LangGraph `MemorySaver` per session |
| Shortlist | `cl.Action` buttons on cards | Buttons + agent can call `add_to_shortlist` |
| FastAPI `/search` | Primary path | Kept for eval/CLI; not used by UI |

## Risks

| Risk | Mitigation |
|---|---|
| Agent calls search 3× per turn (cost) | System prompt: search once per turn unless explicitly asked to refine |
| Agent hallucinates candidate details | Tool returns structured data; agent instructed to quote tool output, not invent |
| MemorySaver grows unbounded | Acceptable for PoC; truncate at ~20 turns if needed |
| LangChain `@tool` + async + `cl.user_session` context | Test shortlist tools carefully; Chainlit session context must be active |
| `create_react_agent` doesn't call tools | Fallback: custom ReAct loop if prebuilt doesn't behave |

## Definition of done for v3

1. Recruiter can type a free-form role description and receive a streamed response
   with candidate recommendations, no mode selection required.
2. Agent correctly calls `search_candidates` and presents results with commentary.
3. Multi-turn refinement works: "show me only senior ones in Berlin" re-searches
   with updated filters in the same conversation.
4. Shortlist: add via agent command or action button; export CSV works.
5. Tool calls visible as collapsed Steps in the Chainlit UI.
6. `uv run pytest` green; eval metrics unchanged.
