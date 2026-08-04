"""DirectorAgent port and reject/retry orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import CreativeBrief, MoviePlan
from .validation import ValidationReport, validate_movie_plan


class DirectorAgent(Protocol):
    """The only port allowed to author a MoviePlan."""

    def create_movie_plan(
        self,
        brief: CreativeBrief,
        direction: str,
        *,
        feedback: str = "",
    ) -> MoviePlan: ...

    def revise_movie_plan(
        self,
        brief: CreativeBrief,
        plan: MoviePlan,
        feedback: str,
    ) -> MoviePlan: ...


@dataclass(frozen=True, slots=True)
class DirectorAttempt:
    attempt: int
    report: ValidationReport


class DirectorOutputRejected(ValueError):
    """Raised when the LLM failed the contract after bounded retries."""

    def __init__(self, attempts: tuple[DirectorAttempt, ...]) -> None:
        self.attempts = attempts
        last = attempts[-1].report if attempts else ValidationReport(("no attempt",))
        super().__init__(last.feedback())


class DirectorOrchestrator:
    """Ask the DirectorAgent again; never patch its output locally."""

    def __init__(self, agent: DirectorAgent, *, max_attempts: int = 2) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.agent = agent
        self.max_attempts = max_attempts

    def create_movie_plan(
        self,
        brief: CreativeBrief,
        direction: str,
    ) -> MoviePlan:
        feedback = ""
        attempts: list[DirectorAttempt] = []
        for attempt in range(1, self.max_attempts + 1):
            candidate = self.agent.create_movie_plan(
                brief,
                direction,
                feedback=feedback,
            )
            report = validate_movie_plan(
                candidate,
                brief,
            )
            attempts.append(DirectorAttempt(attempt, report))
            if report.valid:
                return candidate
            feedback = report.feedback()
        raise DirectorOutputRejected(tuple(attempts))

    def revise_movie_plan(
        self,
        brief: CreativeBrief,
        plan: MoviePlan,
        feedback: str,
    ) -> MoviePlan:
        if not feedback.strip():
            raise ValueError("director revision feedback is required")
        attempts: list[DirectorAttempt] = []
        current_feedback = feedback
        for attempt in range(1, self.max_attempts + 1):
            candidate = self.agent.revise_movie_plan(
                brief,
                plan,
                current_feedback,
            )
            report = validate_movie_plan(
                candidate,
                brief,
            )
            attempts.append(DirectorAttempt(attempt, report))
            if report.valid:
                return candidate
            current_feedback = report.feedback()
        raise DirectorOutputRejected(tuple(attempts))
