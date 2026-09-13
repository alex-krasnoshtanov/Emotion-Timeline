"""One transcription-and-scoring run, and the store that keeps track of them.

A run takes minutes — download, ffmpeg, Whisper, three model loads — so the
browser cannot wait on a request. The job goes on a worker thread, the page polls
for its state, and this module is everything about that which is worth testing:
the state machine, the progress log, the eviction rule.

**In memory, single process, deliberately.** The alternative is a queue, a broker
and a database for a tool one person runs on their own laptop to look at one
video. Restarting the server loses the jobs, which is the correct trade at this
size — and the artefacts each run writes are on disk regardless.

The log is capped and the store evicts oldest-first, because both are unbounded
otherwise and a progress line arrives for every batch of a long transcription.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: Enough to see what happened without letting one run grow without limit.
MAX_LOG_LINES = 400

#: Finished jobs are kept so a reloaded page can still find its result.
MAX_JOBS = 20


class State(StrEnum):
    """Where a run has got to. A string enum so it serialises as its own name."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    @property
    def finished(self) -> bool:
        return self in (State.DONE, State.FAILED)


@dataclass
class Job:
    """One run, its progress, and whatever it produced."""

    id: str
    source: str
    kind: str
    state: State = State.QUEUED
    log: list[str] = field(default_factory=list)
    error: str | None = None
    result: dict[str, Any] | None = None

    def say(self, message: str) -> None:
        """Record a line of progress, dropping the oldest if the log is full."""
        self.log.append(message)
        if len(self.log) > MAX_LOG_LINES:
            # Keep the tail: the useful part of a long run is what it is doing
            # now, and the head is always the same three lines.
            del self.log[: len(self.log) - MAX_LOG_LINES]

    def start(self) -> None:
        self.state = State.RUNNING

    def finish(self, result: dict[str, Any]) -> None:
        self.state = State.DONE
        self.result = result

    def fail(self, error: str) -> None:
        self.state = State.FAILED
        # The browser shows this, so it has to be the message and not a stack.
        self.error = error.strip().splitlines()[-1] if error.strip() else "failed"

    def summary(self) -> dict[str, Any]:
        """What the polling endpoint returns. Never the whole timeline."""
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "state": self.state.value,
            "finished": self.state.finished,
            "log": list(self.log),
            "error": self.error,
            "scenes": len(self.result["timeline"]) if self.result else 0,
        }


class JobStore:
    """Thread-safe, bounded, in memory. Nothing more is warranted here."""

    def __init__(self, limit: int = MAX_JOBS) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._limit = limit

    def create(self, source: str, kind: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], source=source, kind=kind)
        with self._lock:
            self._jobs[job.id] = job
            self._evict()
        return job

    def _evict(self) -> None:
        """Drop the oldest finished jobs once the store is over its limit.

        Running jobs are never evicted: a worker thread holds a reference and
        would go on writing to a job the store had forgotten.
        """
        while len(self._jobs) > self._limit:
            removable = [key for key, job in self._jobs.items() if job.state.finished]
            if not removable:
                return
            del self._jobs[removable[0]]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def __iter__(self) -> Iterator[Job]:
        with self._lock:
            return iter(list(self._jobs.values()))
