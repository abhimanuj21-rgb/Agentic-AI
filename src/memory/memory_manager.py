"""Session memory and conversation persistence manager."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ConversationTurn:
    """Represents a single query-response turn in a session."""
    timestamp: str
    query: str
    response: str
    context_sources: List[str]


class SessionMemoryManager:
    """Manages short-term and persistent session history for multi-turn conversations."""

    def __init__(self, storage_path: str = "./data/memory/session_history.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[ConversationTurn] = self._load()

    def _load(self) -> List[ConversationTurn]:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return [ConversationTurn(**turn) for turn in data]
            except Exception:
                return []
        return []

    def save(self) -> None:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([asdict(t) for t in self.history], f, indent=2)
        except Exception as e:
            print(f"[Warning] Failed to save session memory: {e}")

    def add_turn(self, query: str, response: str, context_sources: Optional[List[str]] = None) -> None:
        """Record a conversation turn."""
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            query=query,
            response=response,
            context_sources=context_sources or [],
        )
        self.history.append(turn)
        self.save()

    def get_recent_context(self, max_turns: int = 3) -> str:
        """Format the most recent conversation turns for prompt context injection."""
        if not self.history:
            return ""

        recent = self.history[-max_turns:]
        lines = ["--- Prior Session Conversation History ---"]
        for t in recent:
            lines.append(f"User: {t.query}")
            lines.append(f"Agent: {t.response[:300]}...")
            if t.context_sources:
                lines.append(f"Sources cited: {', '.join(t.context_sources)}")
            lines.append("")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset conversation memory."""
        self.history = []
        if self.storage_path.exists():
            self.storage_path.unlink()
