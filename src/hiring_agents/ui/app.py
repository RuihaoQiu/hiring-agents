from __future__ import annotations

import httpx
import streamlit as st

from hiring_agents.config import UI_API_BASE_URL


def _stars(score: int, max_score: int = 5) -> str:
    return "★" * score + "☆" * (max_score - score)


def _call_api(query: str) -> dict:
    resp = httpx.post(f"{UI_API_BASE_URL}/search", json={"query": query}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _render_sidebar(normalized: dict) -> None:
    st.sidebar.header("Normalized query")
    hf = normalized["hard_filters"]
    st.sidebar.markdown(f"**Seniority:** {', '.join(hf['seniority']) if hf.get('seniority') else '—'}")
    st.sidebar.markdown(f"**Location:** {', '.join(hf['location_keywords']) if hf.get('location_keywords') else '—'}")
    st.sidebar.markdown(f"**Core skills:** {', '.join(normalized['core_skills']) or '—'}")
    if normalized["nice_to_haves"]:
        st.sidebar.markdown(f"**Nice-to-haves:** {', '.join(normalized['nice_to_haves'])}")
    with st.sidebar.expander("Canonical summary"):
        st.write(normalized["canonical_summary"])


def _render_results(data: dict) -> None:
    if data["filters_relaxed"]:
        st.warning("No results with original filters — seniority requirement was relaxed.")

    ranked = data["ranked"]
    st.subheader(f"Top {len(ranked)} candidates  (retrieved {data['retrieved_count']})")

    for c in ranked:
        label = f"{_stars(c['score'])} **{c['candidate_id']}** — {c['one_line_summary']}"
        with st.expander(label):
            met = [m for m in c["must_have_matches"] if m["met"]]
            missed = [m for m in c["must_have_matches"] if not m["met"]]
            if met:
                st.markdown("**Must-haves met**")
                for m in met:
                    st.markdown(f"- ✓ {m['requirement']}: {m['evidence']}")
            if missed:
                st.markdown("**Must-haves missed**")
                for m in missed:
                    st.markdown(f"- ✗ {m['requirement']}")
            if c["gaps"]:
                st.markdown("**Gaps**")
                for g in c["gaps"]:
                    st.markdown(f"- {g}")


def main() -> None:
    st.set_page_config(page_title="Hiring Agents", layout="wide")
    st.title("Hiring Agents")

    query = st.text_input("Search query", placeholder="e.g. senior python engineer in Berlin")
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

        _render_sidebar(data["normalized"])
        _render_results(data)


if __name__ == "__main__":
    main()
