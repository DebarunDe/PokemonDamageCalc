"""Input and output types for the calculator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTABLE_STATS = STATS[1:]

# Champions replaces EVs with Stat Points: up to 32 in a stat, 66 in total.
MAX_SP_PER_STAT = 32
MAX_SP_TOTAL = 66

STATUSES = {"": "", "none": "", "brn": "brn", "par": "par", "psn": "psn", "tox": "tox", "slp": "slp", "frz": "frz"}
WEATHERS = {"sun": "Sun", "rain": "Rain", "sand": "Sand", "snow": "Snow"}
TERRAINS = {"electric": "Electric", "grassy": "Grassy", "psychic": "Psychic", "misty": "Misty"}


class CalcError(ValueError):
    """Raised for invalid input, such as an unknown Pokemon or an illegal SP spread."""


def _check_stats(values: dict[str, int], allowed: tuple[str, ...], what: str) -> dict[str, int]:
    checked = {}
    for stat, value in values.items():
        key = stat.lower()
        if key not in allowed:
            raise CalcError(f"Unknown stat '{stat}' in {what}; expected one of {', '.join(allowed)}")
        if not isinstance(value, int):
            raise CalcError(f"{what} for {key} must be a whole number, got {value!r}")
        checked[key] = value
    return checked


@dataclass
class PokemonSet:
    """One Pokemon's build. Level is always 50 in Champions."""

    species: str
    nature: str | None = None
    ability: str | None = None
    item: str | None = None
    sp: dict[str, int] = field(default_factory=dict)
    boosts: dict[str, int] = field(default_factory=dict)
    status: str = ""
    cur_hp_percent: float | None = None

    def __post_init__(self) -> None:
        self.sp = _check_stats(self.sp, STATS, "SP")
        for stat, value in self.sp.items():
            if not 0 <= value <= MAX_SP_PER_STAT:
                raise CalcError(f"SP for {stat} must be between 0 and {MAX_SP_PER_STAT}, got {value}")
        total = sum(self.sp.values())
        if total > MAX_SP_TOTAL:
            raise CalcError(f"{self.species} has {total} SP; the maximum is {MAX_SP_TOTAL}")

        self.boosts = _check_stats(self.boosts, BOOSTABLE_STATS, "boost")
        for stat, value in self.boosts.items():
            if not -6 <= value <= 6:
                raise CalcError(f"Boost for {stat} must be between -6 and +6, got {value}")

        status = (self.status or "").lower()
        if status not in STATUSES:
            raise CalcError(f"Unknown status '{self.status}'; expected one of {', '.join(s for s in STATUSES if s)}")
        self.status = STATUSES[status]

        if self.cur_hp_percent is not None and not 0 < self.cur_hp_percent <= 100:
            raise CalcError(f"Current HP must be above 0% and at most 100%, got {self.cur_hp_percent}")


@dataclass
class Field:
    """Battle conditions.

    Reflect, Light Screen, Aurora Veil and Friend Guard protect the defender's
    side. Helping Hand powers up the attacker.
    """

    weather: str | None = None
    terrain: str | None = None
    reflect: bool = False
    light_screen: bool = False
    doubles: bool = False
    aurora_veil: bool = False
    helping_hand: bool = False
    friend_guard: bool = False

    def __post_init__(self) -> None:
        self.weather = _normalize(self.weather, WEATHERS, "weather")
        self.terrain = _normalize(self.terrain, TERRAINS, "terrain")
        if not self.doubles:
            for name in ("helping_hand", "friend_guard"):
                if getattr(self, name):
                    raise CalcError(f"{name.replace('_', ' ').title()} needs an ally, so it only applies in doubles")


def _normalize(value: str | None, options: dict[str, str], what: str) -> str | None:
    if not value:
        return None
    key = value.lower().removesuffix(" terrain")
    if key not in options:
        raise CalcError(f"Unknown {what} '{value}'; expected one of {', '.join(options.values())}")
    return options[key]


@dataclass(frozen=True)
class KOChance:
    chance: float | None
    n: int
    text: str


@dataclass(frozen=True)
class MoveDamage:
    """One move's damage range from `Calculator.calculate_many`."""

    move: str
    type: str
    category: str
    min: int
    max: int
    min_percent: float
    max_percent: float


@dataclass(frozen=True)
class Result:
    """The outcome of one attack. Percentages are of the defender's max HP."""

    description: str
    move: dict[str, Any]
    attacker: dict[str, Any]
    defender: dict[str, Any]
    rolls: list[int]
    min: int
    max: int
    min_percent: float
    max_percent: float
    ko: KOChance | None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_js(cls, data: dict[str, Any], warnings: list[str]) -> Result:
        ko = data.pop("ko")
        return cls(**data, ko=KOChance(**ko) if ko else None, warnings=warnings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
