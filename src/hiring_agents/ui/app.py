from __future__ import annotations

import httpx
import streamlit as st

from hiring_agents.config import UI_API_BASE_URL

_TEAL = "#0D9488"
_MUTED = "#717182"
_BORDER = "rgba(0,0,0,0.1)"

_CSS = f"""
<style>
/* Card-style expanders */
div[data-testid="stExpander"] {{
    border: 1px solid {_BORDER};
    border-radius: 10px;
    margin-bottom: 8px;
    overflow: hidden;
    background: #ffffff;
}}
div[data-testid="stExpander"] summary {{
    padding: 14px 18px;
    font-size: 0.95rem;
}}
div[data-testid="stExpander"] summary:hover {{
    background: #f9f9fb;
}}
div[data-testid="stExpanderDetails"] {{
    border-top: 1px solid {_BORDER};
    padding: 4px 18px 16px;
}}
/* Section labels */
.section-label {{
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {_MUTED};
    margin: 14px 0 4px;
}}
/* Skill tags */
.skill-tag {{
    display: inline-block;
    background: #f3f3f5;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.82rem;
    margin: 2px 3px 2px 0;
    color: #333;
}}
/* Score dots */
.score-dots {{
    font-size: 1.1rem;
    letter-spacing: 3px;
}}
/* Filters-relaxed banner */
.relax-banner {{
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.88rem;
    color: #9a3412;
    margin-bottom: 16px;
}}
</style>
"""


def _dots_html(score: int, max_score: int = 5) -> str:
    filled = f'<span style="color:{_TEAL}">{"●" * score}</span>'
    empty = f'<span style="color:#d5d5d5">{"●" * (max_score - score)}</span>'
    return f'<span class="score-dots">{filled}{empty}</span>'


def _call_api(query: str) -> dict:
    resp = httpx.post(f"{UI_API_BASE_URL}/search", json={"query": query}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _render_query_details(normalized: dict) -> None:
    with st.expander("Query details"):
        hf = normalized["hard_filters"]
        cols = st.columns(3)
        cols[0].markdown(f"**Seniority**  \n{', '.join(hf['seniority']) if hf.get('seniority') else '—'}")
        cols[1].markdown(f"**Location**  \n{', '.join(hf['location_keywords']) if hf.get('location_keywords') else '—'}")
        cols[2].markdown(f"**Core skills**  \n{', '.join(normalized['core_skills']) or '—'}")
        if normalized["nice_to_haves"]:
            st.markdown(f"**Nice-to-haves:** {', '.join(normalized['nice_to_haves'])}")
        st.caption(normalized["canonical_summary"])


def _render_card(c: dict) -> None:
    label = (
        f"**{c['candidate_id']}**"
        f"  ·  {c['current_title']}"
        f"  ·  {c['current_employer']}"
        f"  ·  {c['location']}"
        f"  ·  {c['total_yoe']} yrs"
        f"  ·  {'●' * c['score']}{'●' * (5 - c['score'])}"
    )
    with st.expander(label):
        st.markdown(_dots_html(c["score"]), unsafe_allow_html=True)

        st.markdown('<p class="section-label">Summary</p>', unsafe_allow_html=True)
        st.write(c["summary"])

        st.markdown('<p class="section-label">Experience</p>', unsafe_allow_html=True)
        for e in sorted(c["work_history"], key=lambda x: x["start_year"], reverse=True):
            end = e["end_year"] or "present"
            st.markdown(f"**{e['title']}** at {e['company']}  ·  {e['start_year']}–{end}")
            if e["description"]:
                st.caption(e["description"])

        st.markdown('<p class="section-label">Skills</p>', unsafe_allow_html=True)
        tags = "".join(f'<span class="skill-tag">{s}</span>' for s in c["skills"])
        st.markdown(tags, unsafe_allow_html=True)

        if c["gaps"]:
            st.markdown('<p class="section-label">Gaps</p>', unsafe_allow_html=True)
            for g in c["gaps"]:
                st.markdown(f"— {g}")

        st.markdown('<p class="section-label">Suggestion</p>', unsafe_allow_html=True)
        st.markdown(f"*{c['suggestion']}*")


def main() -> None:
    st.set_page_config(page_title="Hiring Agents", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("Hiring Agents")

    query = st.text_input("Search", placeholder="Search candidates — e.g. senior Python engineer in Berlin", label_visibility="collapsed")
    search = st.button("Search", type="primary", disabled=not query.strip())

    if search and query.strip():
        with st.spinner("Searching…"):
            try:
                data = _call_api(query.strip())
            except httpx.HTTPStatusError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
                return
            except httpx.ConnectError:
                st.error(f"Could not connect to API at {UI_API_BASE_URL}. Is `make api` running?")
                return

        _render_query_details(data["normalized"])
        st.markdown("---")

        if data["filters_relaxed"]:
            st.markdown(
                '<div class="relax-banner">No results matched the original filters — seniority requirement was relaxed.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"**{len(data['ranked'])}** candidates  ·  {data['retrieved_count']} retrieved")
        st.markdown("")

        for c in data["ranked"]:
            _render_card(c)


if __name__ == "__main__":
    main()
