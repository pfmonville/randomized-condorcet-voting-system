from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

IntVote = Sequence[int | Sequence[int]]
StrVote = Sequence[str | Sequence[str]]
Vote = IntVote | StrVote
Ballots = Sequence[Vote]
BatchBallots = Iterable[Ballots]


@dataclass
class NodeData:
    name: str
    proba: str | None
    winner: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LinkData:
    source: int
    target: int
    label: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraphData:
    nodes: list[NodeData]
    links: list[LinkData]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WinnerInfo:
    name: str
    round: int
    chances: float | str


@dataclass
class EliminatedInfo:
    name: str
    round: int
    chances: str
