"""Provenance: a two-node W3C PROV-lite graph (spec §11.1).

Agents do things; activities are the things done.  Objects point at an activity
through their ``prov`` attribute.  Two node types are enough to describe the
workflow that actually dominates real curation --- a model pre-annotates, a
human corrects, a second human reviews --- which a "review status" field cannot
describe at all, because it records *that* something was reviewed without
recording what produced the thing being reviewed.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5.document_fields import check_known
from medh5.errors import MEDH5ValidationError

AGENT_TYPES = ("person", "software", "organization", "model")

AGENT_FIELDS = frozenset(
    {"id", "type", "name", "role", "version", "qualification", "organization"}
)
ACTIVITY_FIELDS = frozenset(
    {
        "id",
        "type",
        "agent",
        "started",
        "ended",
        "tool",
        "inputs",
        "outputs",
        "params",
    }
)
"""The schema's `agent` and `activity` properties; both are closed objects."""

ACTIVITY_TYPES = (
    "import",
    "annotate",
    "review",
    "predict",
    "resample",
    "register",
    "derive",
    "deidentify",
    "transcode",
    "other",
)

RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})$"
)


def check_timestamp(value: str, *, where: str) -> str:
    """Validate an RFC 3339 timestamp (E604)."""
    if not RFC3339.match(value):
        raise MEDH5ValidationError(
            f"{where}: {value!r} is not an RFC 3339 timestamp", code="E604"
        )
    return value


@dataclass(frozen=True, slots=True)
class Agent:
    """Someone or something that acts: a rater, a model, a tool, a site."""

    id: str
    type: str
    name: str
    role: str | None = None
    version: str | None = None
    qualification: str | None = None
    organization: str | None = None
    """The id of an ``organization`` agent this one belongs to (§11.1).

    A schema field, and the one thing the old ``extra`` mapping was ever used
    for --- where it failed E005 at ``commit()``.
    """

    def __post_init__(self) -> None:
        if self.type not in AGENT_TYPES:
            raise MEDH5ValidationError(
                f"agent {self.id!r}: unknown type {self.type!r}; "
                f"expected one of {list(AGENT_TYPES)}",
                code="E603",
            )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "type": self.type, "name": self.name}
        for key in ("role", "version", "qualification", "organization"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Agent:
        check_known(doc, AGENT_FIELDS, what="agent")
        return cls(
            id=str(doc["id"]),
            type=str(doc["type"]),
            name=str(doc["name"]),
            role=doc.get("role"),
            version=doc.get("version"),
            qualification=doc.get("qualification"),
            organization=doc.get("organization"),
        )


@dataclass(frozen=True, slots=True)
class Activity:
    """One thing an agent did, with what it consumed and what it produced."""

    id: str
    type: str
    agent: str | None = None
    started: str | None = None
    ended: str | None = None
    tool: str | None = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in ACTIVITY_TYPES:
            raise MEDH5ValidationError(
                f"activity {self.id!r}: unknown type {self.type!r}; "
                f"expected one of {list(ACTIVITY_TYPES)}",
                code="E603",
            )
        for key in ("started", "ended"):
            value = getattr(self, key)
            if value is not None:
                check_timestamp(value, where=f"activity {self.id!r}.{key}")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "type": self.type}
        for key in ("agent", "started", "ended", "tool"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.inputs:
            out["inputs"] = list(self.inputs)
        if self.outputs:
            out["outputs"] = list(self.outputs)
        if self.params:
            out["params"] = dict(self.params)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Activity:
        check_known(doc, ACTIVITY_FIELDS, what="activity")
        return cls(
            id=str(doc["id"]),
            type=str(doc["type"]),
            agent=doc.get("agent"),
            started=doc.get("started"),
            ended=doc.get("ended"),
            tool=doc.get("tool"),
            inputs=tuple(doc.get("inputs") or ()),
            outputs=tuple(doc.get("outputs") or ()),
            params=dict(doc.get("params") or {}),
        )


class Provenance:
    """The agents and activities of one sample, with reference resolution."""

    __slots__ = ("_activities", "_agents")

    def __init__(
        self,
        agents: Sequence[Agent] = (),
        activities: Sequence[Activity] = (),
    ) -> None:
        self._agents = {a.id: a for a in agents}
        self._activities = {a.id: a for a in activities}

    def __bool__(self) -> bool:
        return bool(self._agents or self._activities)

    def __repr__(self) -> str:
        return (
            f"Provenance({len(self._agents)} agents, "
            f"{len(self._activities)} activities)"
        )

    @property
    def agents(self) -> tuple[Agent, ...]:
        return tuple(self._agents.values())

    @property
    def activities(self) -> tuple[Activity, ...]:
        return tuple(self._activities.values())

    def agent(self, agent_id: str) -> Agent:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise KeyError(f"unknown agent {agent_id!r}") from None

    def activity(self, activity_id: str) -> Activity:
        try:
            return self._activities[activity_id]
        except KeyError:
            raise KeyError(f"unknown activity {activity_id!r}") from None

    def has_activity(self, activity_id: str) -> bool:
        return activity_id in self._activities

    def has_agent(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def add_agent(self, agent: Agent, *, replace: bool = False) -> Agent:
        """Add an agent; an id already in the graph is refused unless *replace*.

        Assigning into the dict silently overwrote: ``person("Alice",
        agent_id="s2")`` followed by ``software("tool")`` --- whose automatic
        id is also ``s2`` --- left one agent, the software, and every
        activity that named Alice now named the tool.  The file validated,
        because every reference resolved; it resolved to the wrong node.
        """
        if not replace and agent.id in self._agents:
            raise MEDH5ValidationError(
                f"agent {agent.id!r} is already declared "
                f"({self._agents[agent.id].name!r}); pass replace=True to "
                "rewrite it deliberately"
            )
        self._agents[agent.id] = agent
        return agent

    def add_activity(self, activity: Activity, *, replace: bool = False) -> Activity:
        """Add an activity; an id already in the graph is refused unless *replace*."""
        if not replace and activity.id in self._activities:
            raise MEDH5ValidationError(
                f"activity {activity.id!r} is already declared "
                f"({self._activities[activity.id].type!r}); pass replace=True to "
                "rewrite it deliberately"
            )
        self._activities[activity.id] = activity
        return activity

    def activities_by_type(self, activity_type: str) -> tuple[Activity, ...]:
        return tuple(a for a in self._activities.values() if a.type == activity_type)

    def produced_by(self, object_path: str) -> tuple[Activity, ...]:
        """Every activity claiming *object_path* among its outputs."""
        return tuple(a for a in self._activities.values() if object_path in a.outputs)

    def dangling_agent_refs(self) -> tuple[tuple[str, str], ...]:
        """``(activity_id, agent_id)`` pairs whose agent is not declared (E605)."""
        return tuple(
            (a.id, a.agent)
            for a in self._activities.values()
            if a.agent is not None and a.agent not in self._agents
        )

    def __iter__(self) -> Iterator[Activity]:
        return iter(self._activities.values())

    def to_json(self) -> dict[str, Any]:
        return {
            "agents": [a.to_json() for a in self._agents.values()],
            "activities": [a.to_json() for a in self._activities.values()],
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | None) -> Provenance:
        if not doc:
            return cls()
        return cls(
            agents=[Agent.from_json(a) for a in doc.get("agents", ())],
            activities=[Activity.from_json(a) for a in doc.get("activities", ())],
        )


__all__ = [
    "ACTIVITY_FIELDS",
    "ACTIVITY_TYPES",
    "AGENT_FIELDS",
    "AGENT_TYPES",
    "Activity",
    "Agent",
    "Provenance",
    "check_timestamp",
]
