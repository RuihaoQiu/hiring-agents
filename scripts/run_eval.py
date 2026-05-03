from __future__ import annotations

from hiring_agents.eval.harness import run_eval
from hiring_agents.logging_setup import configure_logging


def main() -> None:
    configure_logging()
    run_eval()


if __name__ == "__main__":
    main()
