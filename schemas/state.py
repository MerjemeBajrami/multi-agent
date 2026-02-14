from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timezone


class AgentLogEntry(BaseModel):
    timestamp: str
    agent: str
    action: str
    outcome: str

    @staticmethod
    def now(agent: str, action: str, outcome: str) -> "AgentLogEntry":
        return AgentLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=agent,
            action=action,
            outcome=outcome,
        )


class Citation(BaseModel):
    doc_id: str
    location: str
    snippet: str


class ResearchFact(BaseModel):
    fact: str
    citations: List[Citation]


class ResearchNotes(BaseModel):
    status: Literal["ok", "Not found in sources"]
    facts: List[ResearchFact] = Field(default_factory=list)


class AppState(BaseModel):
    # Inputs
    user_task: str = ""

    # Orchestration artifacts
    plan: List[str] = Field(default_factory=list)

    # Research artifacts
    research_notes: Optional[ResearchNotes] = None

    # Writing artifacts
    draft_output: Optional[str] = None

    # Final artifacts
    final_output: Optional[str] = None

    # Flattened citations for display (optional convenience)
    citations: List[Citation] = Field(default_factory=list)

    # Traceability
    agent_logs: List[AgentLogEntry] = Field(default_factory=list)

    # Control for verifier loop
    verifier_fail_count: int = 0
    verifier_max_retries: int = 2

    # Extra metadata (optional)
    meta: Dict[str, Any] = Field(default_factory=dict)

    def log(self, agent: str, action: str, outcome: str) -> None:
        self.agent_logs.append(AgentLogEntry.now(agent, action, outcome))
