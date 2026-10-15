"""What the clinical importers share: source times put on a sample's clock.

A source time is a wall-clock reading (no zone), an instant (an offset or a
zone was stored with it), or a calendar period known to the year, month or
day.  Each becomes inclusive bounds on the subject clock (1.1 §5.2), and
nothing is assumed on the way: a wall-clock time needs a zone to name an
instant, a relative clock needs its origin, a period spans every instant it
may cover, a reading that falls twice (a daylight-saving fall-back) is bounded
by both, and one that never occurred (the spring-forward gap) is refused.
"""

from __future__ import annotations

import datetime
import re
import zoneinfo
from collections.abc import Mapping
from typing import Any

from medh5.clinical import DAY, HOUR, Bounds, Clock, utc_us
from medh5.errors import MEDH5ValidationError

_NAIVE_EPOCH = datetime.datetime(1970, 1, 1)
_UTC = datetime.timezone.utc
_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")


def valid_id(value: str) -> bool:
    """Whether *value* is a 1.0 id (`[A-Za-z0-9_.-]{1,128}`)."""
    return _ID.fullmatch(value) is not None


def wall_us(moment: datetime.datetime) -> int:
    """A naive ``datetime`` as microseconds since 1970-01-01 on its own
    wall clock."""
    delta = moment - _NAIVE_EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def wall_datetime(us: int) -> datetime.datetime:
    """The naive ``datetime`` *us* wall-clock microseconds name."""
    return _NAIVE_EPOCH + datetime.timedelta(microseconds=us)


def zone_of(value: str | datetime.tzinfo | None) -> datetime.tzinfo | None:
    if value is None or isinstance(value, datetime.tzinfo):
        return value
    try:
        return zoneinfo.ZoneInfo(value)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        raise MEDH5ValidationError(
            f"timezone {value!r} is not an IANA time zone this Python knows "
            "(on Windows, install tzdata)"
        ) from None


def parse_moment(value: str | datetime.datetime, what: str) -> datetime.datetime:
    """An ISO 8601 date-time, or a ``datetime`` as given."""
    if isinstance(value, datetime.datetime):
        return value
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise MEDH5ValidationError(
            f"{what} {value!r} is not an ISO 8601 date-time"
        ) from None


class Timeline:
    """Puts source times on *clock*.

    ``timezone`` reads wall-clock times; ``origin`` is the instant (or the
    wall-clock reading) a relative clock counts from.  On a ``utc`` or
    ``shifted_utc`` clock a wall-clock time needs ``timezone``; on a relative
    clock with neither a zone nor an aware origin, wall-clock times are
    counted from a wall-clock origin as read (:attr:`uniform`).
    """

    def __init__(
        self,
        clock: Clock,
        *,
        timezone: str | datetime.tzinfo | None,
        origin: str | datetime.datetime | None,
        converter: str,
    ) -> None:
        if clock.reference not in ("utc", "shifted_utc", "relative"):
            raise MEDH5ValidationError(
                f"clock {clock.id!r} has reference {clock.reference!r}; "
                "expected utc, shifted_utc or relative"
            )
        self.clock = clock
        self.zone = zone_of(timezone)
        self.relative = clock.reference == "relative"
        self.uniform = False
        """Wall-clock times counted from a wall-clock origin, no zone known."""
        self._origin_us = 0
        self._origin_wall: int | None = None
        if not self.relative:
            if origin is not None:
                raise MEDH5ValidationError(
                    f"origin is for a relative clock; clock {clock.id!r} is "
                    f"{clock.reference}, so times are placed as they are"
                )
            return
        if origin is None:
            raise MEDH5ValidationError(
                f"clock {clock.id!r} is relative, and {converter} invents no "
                "origin: give origin, the instant the clock's zero is"
            )
        moment = parse_moment(origin, "origin")
        if moment.tzinfo is not None and moment.utcoffset() is not None:
            self._origin_us = utc_us(moment)
        elif self.zone is not None:
            lo, hi = self._resolve(wall_us(moment))
            if lo != hi:
                raise MEDH5ValidationError(
                    f"origin {moment.isoformat()} occurs twice in "
                    f"{self.zone}: give it with its offset"
                )
            self._origin_us = lo
        else:
            self._origin_wall = wall_us(moment)
            self.uniform = True

    def _resolve(self, wall: int) -> Bounds:
        """The instants (UTC) the wall-clock reading *wall* names in the zone."""
        assert self.zone is not None
        moment = wall_datetime(wall)
        found: set[int] = set()
        for fold in (0, 1):
            instant = utc_us(moment.replace(tzinfo=self.zone, fold=fold))
            back = (
                datetime.datetime(1970, 1, 1, tzinfo=_UTC)
                + datetime.timedelta(microseconds=instant)
            ).astimezone(self.zone)
            if back.replace(tzinfo=None) == moment:
                found.add(instant)
        if not found:
            raise MEDH5ValidationError(
                f"{moment.isoformat()} never occurred in {self.zone} (a "
                "daylight-saving gap): the source time names no instant"
            )
        return min(found), max(found)

    def _boundary(self, wall: int) -> int:
        """The instant a period boundary at wall-clock *wall* begins."""
        assert self.zone is not None
        return utc_us(wall_datetime(wall).replace(tzinfo=self.zone))

    def _shift(self, bounds: Bounds) -> Bounds:
        return bounds[0] - self._origin_us, bounds[1] - self._origin_us

    def instant(self, utc: int) -> Bounds:
        """An instant, as microseconds since the Unix epoch."""
        if self._origin_wall is not None:
            raise MEDH5ValidationError(
                "the source gives an instant and the origin is a wall-clock "
                "reading: give timezone, or the origin with its offset"
            )
        return self._shift((utc, utc))

    def wall(self, wall: int) -> Bounds:
        """A wall-clock reading, as microseconds since 1970 on that clock."""
        if self._origin_wall is not None:
            return wall - self._origin_wall, wall - self._origin_wall
        if self.zone is None:
            raise MEDH5ValidationError(
                f"{wall_datetime(wall).isoformat()} has no time zone, so it "
                f"names no instant on {self.clock.reference} clock "
                f"{self.clock.id!r}: give timezone"
            )
        return self._shift(self._resolve(wall))

    def period(self, start: int, end: int) -> Bounds:
        """Every instant of the wall-clock period ``[start, end)``.

        With no zone known, the period spans every instant it covers
        somewhere: from its start at UTC+14 to its end at UTC-12."""
        if self._origin_wall is not None:
            return start - self._origin_wall, end - 1 - self._origin_wall
        if self.zone is None:
            return self._shift((start - 14 * HOUR, end - 1 + 12 * HOUR))
        return self._shift((self._boundary(start), self._boundary(end) - 1))

    def day(self, day: datetime.date) -> Bounds:
        """Every instant of a calendar *day*."""
        start = wall_us(datetime.datetime.combine(day, datetime.time()))
        return self.period(start, start + DAY)


def option_map(
    value: Mapping[str, Any] | None, what: str, allowed: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """A mapping option as a ``dict``; keys outside *allowed* are refused."""
    found = dict(value or {})
    if allowed is not None:
        unknown = sorted(set(found) - set(allowed))
        if unknown:
            raise MEDH5ValidationError(
                f"{what} has no key {', '.join(map(repr, unknown))}; it takes "
                f"{', '.join(allowed)}"
            )
    return found


__all__: list[str] = []
