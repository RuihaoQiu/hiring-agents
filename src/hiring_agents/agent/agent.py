from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from hiring_agents.agent.prompts import AGENT_SYSTEM_PROMPT
from hiring_agents.agent.tools import search_candidates, show_more_candidates
from hiring_agents.config import AGENT_MODEL, AGENT_TEMPERATURE

_checkpointer = MemorySaver()
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        llm = ChatOpenAI(model=AGENT_MODEL, temperature=AGENT_TEMPERATURE, streaming=True)
        _agent = create_react_agent(
            model=llm,
            tools=[search_candidates, show_more_candidates],
            prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=_checkpointer,
        )
    return _agent
