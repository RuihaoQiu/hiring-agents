# Hiring Agents — v2 Plan

A recruiter-grade product layer on top of the v1 API. The goal of v2 is to add
job-description search, a strict search mode, a Chainlit conversational UI,
and a per-session shortlist with CSV export.

## What v2 is and isn't

**v2 is:**
- A Chainlit chat UI replacing the Streamlit UI: queries are conversation
  turns, two Starter buttons trigger JD mode and Strict mode
- Three search modes: `keyword` (default), `jd` (paste a full job description),
  `strict` (keyword search, no filter relaxation)
- A per-session shortlist (cl.user_session): add candidates via Action buttons,
  export to CSV as an inline file download
- A JD-specific normalization prompt that parses requirements sections into the
  existing `NormalizedQuery` schema
- The same 500 synthetic candidates, eval harness, and v1 pipeline unchanged

**v2 is not:**
- Persistent storage (shortlist lives in Chainlit session state, cleared on
  page reload)
- Multi-role or multi-user shortlists
- Real candidate data, auth, or production deployment
- BM25 hybrid retrieval or PDF export

Target: one person, ~3 focused steps.

## Non-negotiable principles (inherited from v1)

All v1 principles apply. Additions:

11. **(new) Shortlist is session-only.** Lives in `cl.user_session`. No server
    state, no API endpoints for shortlist.
12. **(new) Mode is explicit in every request.** The graph branches on `mode`;
    no implicit behaviour from missing fields.

## Stack changes (delta from v1)

| Concern | v1 | v2 |
|---|---|---|
| UI framework | Streamlit | Chainlit >= 2.0.0 |
| UI interaction | stateless query box | `@cl.on_message` + `cl.user_session` history |
| Mode starters | none | `@cl.set_starters` with two starters |
| Progress feedback | `st.spinner` | `cl.Step` per pipeline stage |
| Add to shortlist | none | `cl.Action` buttons per candidate card |
| Shortlist export | none | `cl.File` inline download (CSV from session state) |
| New deps | — | `chainlit>=2.0.0` (remove `streamlit`) |

## Schema changes

`SearchRequest` gains two fields:
```python
mode: Literal["keyword", "jd", "strict"] = "keyword"
hard_filters: HardFilters | None = None  # pre-built by UI for strict mode
```

When `mode="strict"` and `hard_filters` is provided, the normalize node skips
the LLM call and wraps the filters directly into a `NormalizedQuery`.

`SearchResponse` is unchanged — the mode distinction is handled upstream.

## Graph changes

`PipelineState` gains:
```python
mode: Literal["keyword", "jd", "strict"]
```

Conditional edge for filter relaxation gains a strict-mode guard:

```python
def _should_relax(state: PipelineState) -> str:
    if state.get("mode") == "strict":
        return "rerank"          # never relax in strict mode
    if len(state["retrieved"]) == 0 and not state.get("filters_relaxed"):
        return "relax_filters"
    return "rerank"
```

`normalize` node branches on `mode`:
- `"jd"` → `normalize_jd(raw_query)` (new prompt, same output schema)
- `"keyword"` / `"strict"` → `normalize_query(raw_query)` (existing)

## New: JD normalization

`normalize.py` gets `normalize_jd(text: str) -> NormalizedQuery`. Same schema,
different prompt: parse "Requirements" / "Qualifications" / "Must have" sections;
extract must-have skills as `core_skills`, preferred as `nice_to_haves`,
seniority and location signals. No new schema fields needed.

## Directory layout (delta from v1)

```
hiring-agents/
├── pyproject.toml          # chainlit>=2.0.0 replaces streamlit
├── .chainlit/
│   └── config.toml         # assistant name, theme path, cot="full"
├── public/
│   ├── theme.json          # teal primary: "175 84% 32%"
│   └── custom.css          # hide Chainlit watermark
├── src/hiring_agents/
│   ├── normalize.py        # + normalize_jd()
│   ├── graph.py            # + mode field + strict guard
│   ├── api/
│   │   └── models.py       # + mode field on SearchRequest
│   └── ui/
│       └── app.py          # full rewrite: Chainlit app
└── Makefile                # ui target → chainlit run src/hiring_agents/ui/app.py
```

`src/hiring_agents/ui/app.py` is replaced. The old Streamlit code is deleted.

## Chainlit UI design

**Starters** (shown on empty chat screen):

```
[ 📋 Job Description ]   [ 🔍 Strict Search ]
```

Clicking a starter sends a sentinel message (`__jd__` or `__strict__`) that the
`@cl.on_message` handler intercepts before hitting the API.

**Default flow** (keyword):
```
user:      "senior python engineer berlin"
assistant: [Step: Normalizing]
           [Step: Retrieving — 10 candidates]
           [Step: Ranking]
           ── Candidate 1 ──────────────────────────
           **Alice M.** · Senior Engineer · Acme · Berlin · 6 yrs
           Score: ●●●●○  Summary: ...
           Skills: Python, FastAPI, ...  Gaps: ...
           [+ Shortlist]
           ── Candidate 2 ...
           ── Shortlist: 0 saved  [Export CSV]
```

**JD flow:**
1. User clicks "Job Description" starter
2. Assistant: "Paste your job description:"
3. User pastes full JD text
4. Same search flow with `mode="jd"`

**Strict flow** (structured form, conversational):
1. User clicks "Strict Search" starter
2. Assistant: "Enter job title:" → user types free text
3. Assistant: "Select location:" → `cl.Action` buttons for each location option
   (from `LOCATIONS` in `data_gen/axes.py`; "Any" option to skip)
4. Assistant: "Select seniority:" → `cl.Action` buttons: junior / mid / senior /
   staff / principal / any
5. Assistant runs search with the three collected values mapped to `HardFilters`;
   `mode="strict"` disables filter relaxation fallback
6. If 0 results: explicit message "No candidates matched these filters — try
   broadening location or seniority"

Strict mode bypasses `normalize_query` entirely — the form fields map directly
to `HardFilters` and a canonical summary is synthesised from the three values.
A new `SearchRequest` field carries the pre-built filters so the graph's
normalize node is skipped:

```python
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: Literal["keyword", "jd", "strict"] = "keyword"
    hard_filters: HardFilters | None = None  # pre-built by UI for strict mode
```

When `mode="strict"` and `hard_filters` is provided, the normalize node wraps
the fields into a `NormalizedQuery` without calling the LLM.

**Shortlist:**
- `cl.user_session` holds `shortlist: list[dict]`
- `cl.Action(name="add_shortlist", ...)` button on each candidate card
- `@cl.action_callback("add_shortlist")` appends to session shortlist
- After every search, a summary line shows shortlist count and an Export button
- Export: write CSV in memory, send as `cl.File(display="inline")`

**Progress steps** (cl.Step per stage):
```
▶ Normalizing query...      ✓ mode: keyword | skills: Python | seniority: senior
▶ Retrieving candidates...  ✓ 10 retrieved (filters: Berlin, senior)
▶ Ranking...                ✓ 3 ranked
```

## Build order

### Step 1 — Search modes + JD normalization

1. Add `mode: Literal["keyword", "jd", "strict"] = "keyword"` to
   `SearchRequest` in `api/models.py`.
2. Add `normalize_jd(text: str) -> NormalizedQuery` to `normalize.py` with
   JD-specific prompt.
3. Update `graph.py`: thread `mode` through `PipelineState`; branch normalize
   node on mode; add strict guard to conditional edge.
4. Tests: `test_normalize_jd.py` (smoke, mocked LLM); `test_graph.py` — assert
   strict mode never sets `filters_relaxed`; assert jd mode calls normalize_jd.

**Milestone:** `POST /search` with `mode="jd"` returns ranked results from JD
text; `mode="strict"` with pre-built `hard_filters` skips the LLM normalize
call and returns empty ranked list (not an error) when no candidates match.

### Step 2 — Chainlit UI (search + progress)

1. Add `chainlit>=2.0.0` to `pyproject.toml`; remove `streamlit`; `uv sync`.
2. Add `.chainlit/config.toml`, `public/theme.json`, `public/custom.css`.
3. Rewrite `src/hiring_agents/ui/app.py` as Chainlit app:
   - `@cl.set_starters`: JD and Strict starters
   - `@cl.on_chat_start`: init `mode="keyword"`, `shortlist=[]` in session
   - `@cl.on_message`: sentinel routing → mode state → search → render results
   - `_call_api(query, mode)` via httpx (same as before)
   - `_render_results(data)`: one `cl.Message` per candidate with `cl.Action`
   - Three `cl.Step` blocks for Normalizing / Retrieving / Ranking progress
4. Update Makefile `ui` target:
   `chainlit run src/hiring_agents/ui/app.py --port 8501`
5. Tests: `test_ui.py` — update `_call_api` tests; no Chainlit rendering tests.

**Milestone:** chat UI at `localhost:8501` shows conversation turns, progress
steps, and candidate cards with teal theme.

### Step 3 — Shortlist + CSV export

1. Add `@cl.action_callback("add_shortlist")` to `ui/app.py`: append to
   `cl.user_session` shortlist, send confirmation message.
2. Add "Export CSV" `cl.Action` in the post-search summary message: on click,
   write CSV to a temp file and send as `cl.File(display="inline")`.
3. Add "Clear Shortlist" action for starting a new role search.
4. Tests: no new tests (session state is Chainlit-internal); manual smoke test.

**Milestone:** recruiter adds candidates across multiple searches, exports a
CSV file with all shortlisted candidates.

## `config.py` additions

```python
CHAINLIT_PORT: int = 8501
```

Chainlit credentials / telemetry opt-out set via `CHAINLIT_TELEMETRY=false` in
`.env`.

## Risks summary

| Risk | Likelihood | Mitigation |
|---|---|---|
| JD normalization over/under-extracts skills | Medium | Smoke test on 3 real JD samples; tune prompt |
| cl.Action state lost if user refreshes page | Low | Acceptable for v2; document limitation |
| Chainlit version churn (>=2.0) | Medium | Pin exact minor `==2.x.y` after install |
| Strict mode returning 0 results confuses recruiter | Low | Show explicit "No candidates matched strict filters" message |

## Definition of done for v2

1. `POST /search` with `mode="keyword"`, `"jd"`, and `"strict"` all return
   valid `SearchResponse` JSON.
2. Chainlit UI at `localhost:8501`: keyword flow, JD mode, strict mode, candidate
   cards with progress steps all work end-to-end.
3. Shortlist: add via action buttons, export CSV as inline file download.
4. `uv run pytest` green.
5. Eval metrics unchanged from v1 (keyword path unaffected by new modes).
