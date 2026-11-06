from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class SubQuestion:
    id: str
    heading: str
    query: str


@dataclass
class Source:
    id: str
    url: str
    title: str
    snippet: str = ""
    origin: str = ""          # wikipedia | web
    subq: str = ""
    text: str = ""            # fetched full text (not serialised to clients)
    status: str = "found"     # found | fetched | failed

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("text")
        return d


@dataclass
class Evidence:
    id: str
    source_id: str
    subq: str
    quote: str
    score: float = 0.0


@dataclass
class Claim:
    text: str
    evidence: list[str]
    verified: bool = False
    support: float = 0.0


@dataclass
class Section:
    heading: str
    claims: list[Claim] = field(default_factory=list)
