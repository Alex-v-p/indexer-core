from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from packages.rag_core.query_understanding.temporal.models import DateRange

_MONTHS = {
    name.casefold(): index
    for index in range(1, 13)
    for name in (calendar.month_name[index], calendar.month_abbr[index])
}
_MONTH_NAME_PATTERN = "|".join(sorted((re.escape(name) for name in _MONTHS), key=len, reverse=True))
_ISO_DATE_PATTERN = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DMY_DATE_PATTERN = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
_MONTH_DAY_PATTERN = re.compile(
    rf"\b({_MONTH_NAME_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
_DAY_MONTH_PATTERN = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAME_PATTERN})\s+(\d{{4}})\b",
    re.IGNORECASE,
)
_MONTH_YEAR_PATTERN = re.compile(rf"\b({_MONTH_NAME_PATTERN})\s+(\d{{4}})\b", re.IGNORECASE)
_MONTH_ONLY_PATTERN = re.compile(rf"\b({_MONTH_NAME_PATTERN})\b", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")
_RELATIVE_COUNT_PATTERN = re.compile(
    r"\b(?:last|past|previous)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ResolvedDateExpression:
    date_range: DateRange
    expression: str
    confidence: float


def resolve_date_expression(
    question: str,
    *,
    reference_datetime: datetime | None = None,
    timezone_name: str = "UTC",
) -> ResolvedDateExpression | None:
    """Resolve one explicit natural-language date expression into a UTC range."""

    normalized = " ".join(question.strip().split())
    if not normalized:
        return None

    zone = ZoneInfo(timezone_name)
    reference = reference_datetime or datetime.now(zone)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=zone)
    else:
        reference = reference.astimezone(zone)

    lower = normalized.casefold()
    relative = _resolve_relative(lower, reference)
    if relative is not None:
        return relative

    values = _extract_absolute_values(normalized, zone, reference)
    if not values:
        return None

    if "between" in lower or re.search(r"\bfrom\b.+\b(?:to|through|until)\b", lower):
        if len(values) >= 2:
            first_start, first_end, first_expression, first_confidence = values[0]
            second_start, second_end, second_expression, second_confidence = values[1]
            return ResolvedDateExpression(
                DateRange(start=first_start, end=second_end),
                f"{first_expression} to {second_expression}",
                min(first_confidence, second_confidence, 0.96),
            )

    start, end, expression, confidence = values[0]
    if re.search(r"\b(before|earlier than|prior to)\b", lower):
        return ResolvedDateExpression(DateRange(end=start), expression, min(confidence, 0.95))
    if re.search(r"\b(after|later than)\b", lower):
        return ResolvedDateExpression(DateRange(start=end), expression, min(confidence, 0.95))
    if re.search(r"\bsince\b", lower):
        return ResolvedDateExpression(DateRange(start=start), expression, min(confidence, 0.93))
    if re.search(
        r"\b(?:starting\s+from|from)\b.+\b(?:onward|onwards|forward|to\s+(?:the\s+)?present|until\s+now)\b",
        lower,
    ):
        return ResolvedDateExpression(DateRange(start=start), expression, min(confidence, 0.93))
    if re.search(r"\b(until|through)\b", lower):
        return ResolvedDateExpression(DateRange(end=end), expression, min(confidence, 0.93))
    return ResolvedDateExpression(DateRange(start=start, end=end), expression, min(confidence, 0.94))


def _resolve_relative(lower: str, reference: datetime) -> ResolvedDateExpression | None:
    today_start = datetime.combine(reference.date(), time.min, tzinfo=reference.tzinfo)
    if re.search(r"\btoday\b", lower):
        return _utc_result(today_start, today_start + timedelta(days=1), "today", 0.99)
    if re.search(r"\byesterday\b", lower):
        start = today_start - timedelta(days=1)
        return _utc_result(start, today_start, "yesterday", 0.99)
    if re.search(r"\bthis month\b", lower):
        start = today_start.replace(day=1)
        return _utc_result(start, _add_months(start, 1), "this month", 0.97)
    if re.search(r"\b(?:last|previous) month\b", lower):
        end = today_start.replace(day=1)
        return _utc_result(_add_months(end, -1), end, "last month", 0.97)
    if re.search(r"\bthis year\b", lower):
        start = today_start.replace(month=1, day=1)
        return _utc_result(start, start.replace(year=start.year + 1), "this year", 0.97)
    if re.search(r"\b(?:last|previous) year\b", lower):
        end = today_start.replace(month=1, day=1)
        return _utc_result(end.replace(year=end.year - 1), end, "last year", 0.97)

    match = _RELATIVE_COUNT_PATTERN.search(lower)
    if match is None:
        return None
    count = int(match.group(1))
    unit = match.group(2).casefold()
    end = reference
    if unit.startswith("day"):
        start = end - timedelta(days=count)
    elif unit.startswith("week"):
        start = end - timedelta(weeks=count)
    elif unit.startswith("month"):
        start = _add_months(end, -count)
    else:
        start = _add_years(end, -count)
    return _utc_result(start, end, match.group(0), 0.94)


def _extract_absolute_values(
    value: str,
    zone: ZoneInfo,
    reference: datetime,
) -> list[tuple[datetime, datetime, str, float]]:
    matches: list[tuple[int, datetime, datetime, str, float]] = []

    def add(
        match: re.Match[str],
        parsed_date: date,
        expression: str,
        confidence: float = 0.98,
    ) -> None:
        start = datetime.combine(parsed_date, time.min, tzinfo=zone)
        matches.append(
            (
                match.start(),
                start.astimezone(UTC),
                (start + timedelta(days=1)).astimezone(UTC),
                expression,
                confidence,
            ),
        )

    for match in _ISO_DATE_PATTERN.finditer(value):
        add(match, date(int(match.group(1)), int(match.group(2)), int(match.group(3))), match.group(0))
    for match in _DMY_DATE_PATTERN.finditer(value):
        add(match, date(int(match.group(3)), int(match.group(2)), int(match.group(1))), match.group(0))
    for match in _MONTH_DAY_PATTERN.finditer(value):
        add(
            match,
            date(int(match.group(3)), _MONTHS[match.group(1).casefold()], int(match.group(2))),
            match.group(0),
        )
    for match in _DAY_MONTH_PATTERN.finditer(value):
        add(
            match,
            date(int(match.group(3)), _MONTHS[match.group(2).casefold()], int(match.group(1))),
            match.group(0),
        )

    occupied = [(item[0], item[0] + len(item[3])) for item in matches]
    for match in _MONTH_YEAR_PATTERN.finditer(value):
        if _overlaps(match.start(), match.end(), occupied):
            continue
        month = _MONTHS[match.group(1).casefold()]
        year = int(match.group(2))
        start = datetime(year, month, 1, tzinfo=zone)
        end = _add_months(start, 1)
        matches.append(
            (match.start(), start.astimezone(UTC), end.astimezone(UTC), match.group(0), 0.96),
        )
        occupied.append((match.start(), match.end()))

    for match in _YEAR_PATTERN.finditer(value):
        if _overlaps(match.start(), match.end(), occupied):
            continue
        year = int(match.group(1))
        start = datetime(year, 1, 1, tzinfo=zone)
        end = datetime(year + 1, 1, 1, tzinfo=zone)
        matches.append(
            (match.start(), start.astimezone(UTC), end.astimezone(UTC), match.group(0), 0.96),
        )
        occupied.append((match.start(), match.end()))

    for match in _MONTH_ONLY_PATTERN.finditer(value):
        if _overlaps(match.start(), match.end(), occupied):
            continue
        month = _MONTHS[match.group(1).casefold()]
        year = reference.year if month <= reference.month else reference.year - 1
        start = datetime(year, month, 1, tzinfo=zone)
        end = _add_months(start, 1)
        expression = f"{calendar.month_name[month]} {year}"
        matches.append(
            (match.start(), start.astimezone(UTC), end.astimezone(UTC), expression, 0.82),
        )

    matches.sort(key=lambda item: item[0])
    return [
        (start, end, expression, confidence)
        for _, start, end, expression, confidence in matches
    ]

def _utc_result(start: datetime, end: datetime, expression: str, confidence: float) -> ResolvedDateExpression:
    return ResolvedDateExpression(
        DateRange(start=start.astimezone(UTC), end=end.astimezone(UTC)),
        expression,
        confidence,
    )


def _add_months(value: datetime, months: int) -> datetime:
    total = value.year * 12 + (value.month - 1) + months
    year, month_index = divmod(total, 12)
    day = min(value.day, calendar.monthrange(year, month_index + 1)[1])
    return value.replace(year=year, month=month_index + 1, day=day)


def _add_years(value: datetime, years: int) -> datetime:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def _overlaps(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in occupied)
