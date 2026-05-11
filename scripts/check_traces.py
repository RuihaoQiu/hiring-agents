"""Print a summary of recent Langfuse traces with their child observations."""

import os
from datetime import timezone

from dotenv import load_dotenv

load_dotenv()

from langfuse.api import LangfuseAPI  # noqa: E402

LIMIT = int(os.getenv("TRACE_LIMIT", "5"))


def main() -> None:
    lf = LangfuseAPI(
        base_url=os.environ["LANGFUSE_BASE_URL"],
        username=os.environ["LANGFUSE_PUBLIC_KEY"],
        password=os.environ["LANGFUSE_SECRET_KEY"],
    )

    traces = lf.trace.list(limit=LIMIT)
    if not traces.data:
        print("No traces found.")
        return

    for t in traces.data:
        ts = t.timestamp.astimezone(timezone.utc).strftime("%H:%M:%S")
        session = (t.session_id or "—")[:24]
        cost = f"${t.total_cost:.4f}" if t.total_cost else "—"
        latency = f"{t.latency:.1f}s" if t.latency else "—"
        print(f"\n{'─'*60}")
        print(f"  {ts}  {t.name}  |  session={session}  |  {cost}  {latency}")
        full = lf.trace.get(t.id)
        obs = sorted(full.observations or [], key=lambda x: x.start_time or "")
        for o in obs:
            indent = "    ├─ " if o.parent_observation_id else "  ├─ "
            name = o.name or o.type or "?"
            model = f"  [{o.model}]" if getattr(o, "model", None) else ""
            ocost = f"  ${o.total_cost:.4f}" if getattr(o, "total_cost", None) else ""
            olatency = f"  {o.latency:.1f}s" if getattr(o, "latency", None) else ""
            print(f"{indent}{name}{model}{ocost}{olatency}")

    print(f"\n{'─'*60}")
    print(f"Showing {len(traces.data)} most recent traces. Set TRACE_LIMIT=N to change.")


if __name__ == "__main__":
    main()
