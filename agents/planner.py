from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from schemas.state import AppState


class PlanOut(BaseModel):
    steps: List[str] = Field(..., description="Ordered steps to complete the task.")


SYSTEM = """You are Planner Agent.
You ONLY create an execution plan. Do NOT do research. Do NOT draft the deliverable.

Rules:
- Produce 3-6 steps, ordered.
- Steps should map to: research -> writing -> verification.
- Output MUST be valid JSON per the schema.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "User task:\n{user_task}\n\nCreate the plan JSON now."),
    ]
)


def run_planner(state: AppState) -> AppState:
    llm = ChatOpenAI(model=state.meta.get("model", "gpt-4o-mini"), temperature=0)
    structured = llm.with_structured_output(PlanOut)

    out: PlanOut = structured.invoke(
        PROMPT.format_messages(user_task=state.user_task)
    )

    state.plan = out.steps
    state.log("planner", "created plan", f"{len(out.steps)} steps")
    return state
