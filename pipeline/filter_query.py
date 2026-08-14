from __future__ import annotations

from dataclasses import dataclass


class FilterQueryError(ValueError):
    pass


@dataclass(frozen=True)
class FilterClause:
    field: str
    operator: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str


FIELD_ALIASES = {
    "movie/show": "title",
    "movie": "title",
    "show": "title",
    "title": "title",
    "people": "people",
    "person": "people",
    "characters": "people",
    "minsec": "minsec",
    "maxsec": "maxsec",
    "duration": "duration",
    "seconds": "duration",
    "shot": "shot",
    "shotsize": "shot",
    "camera": "camera",
    "motion": "motion",
    "mood": "mood",
    "tag": "tag",
    "files": "files",
    "downloadable": "files",
}

NUMERIC_FIELDS = {"people", "minsec", "maxsec", "duration"}
CATEGORICAL_FIELDS = {"title", "shot", "camera", "motion", "mood", "tag", "files"}
COMPARISON_OPERATORS = {"=", "!=", ">", ">=", "<", "<="}


def parse_filter_query(query: str) -> list[FilterClause]:
    tokens = _tokenize(query)
    if not tokens:
        return []

    clauses: list[FilterClause] = []
    position = 0
    while position < len(tokens):
        field_token = tokens[position]
        if field_token.kind != "value":
            raise FilterQueryError(f"Expected a field, got {field_token.value!r}")
        field = _canonical_field(field_token.value)
        position += 1

        if position >= len(tokens) or tokens[position].kind != "operator":
            raise FilterQueryError(f"Expected an operator after {field_token.value!r}")
        operator = tokens[position].value
        position += 1

        value, position = _read_value(tokens, position, field)
        values = [value]
        while position < len(tokens) and _is_keyword(tokens[position], "OR"):
            position += 1
            if (
                position + 1 < len(tokens)
                and tokens[position].kind == "value"
                and tokens[position + 1].kind == "operator"
            ):
                repeated_field = _canonical_field(tokens[position].value)
                repeated_operator = tokens[position + 1].value
                if repeated_field != field or repeated_operator != operator:
                    raise FilterQueryError("OR may only join values for the same field and operator")
                position += 2
            value, position = _read_value(tokens, position, field)
            values.append(value)

        _validate_clause(field, operator, values)
        clauses.append(FilterClause(field, operator, tuple(values)))

        if position >= len(tokens):
            break
        if not _is_keyword(tokens[position], "AND"):
            raise FilterQueryError(f"Expected AND, got {tokens[position].value!r}")
        position += 1
        if position >= len(tokens):
            raise FilterQueryError("Expected another condition after AND")

    return clauses


def _tokenize(query: str) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    while position < len(query):
        if query[position].isspace():
            position += 1
            continue

        char = query[position]
        if char in {'"', "'"}:
            quote = char
            position += 1
            value: list[str] = []
            while position < len(query):
                char = query[position]
                if char == "\\":
                    position += 1
                    if position >= len(query):
                        raise FilterQueryError("Quoted value ends with an escape character")
                    value.append(query[position])
                    position += 1
                    continue
                if char == quote:
                    position += 1
                    break
                value.append(char)
                position += 1
            else:
                raise FilterQueryError("Unclosed quoted value")
            tokens.append(_Token("value", "".join(value)))
            continue

        if char in "=<>!":
            operator = char
            position += 1
            if position < len(query) and query[position] == "=":
                operator += "="
                position += 1
            if operator not in COMPARISON_OPERATORS:
                raise FilterQueryError(f"Unsupported operator {operator!r}")
            tokens.append(_Token("operator", operator))
            continue

        start = position
        while position < len(query) and not query[position].isspace() and query[position] not in "=<>!":
            position += 1
        tokens.append(_Token("value", query[start:position]))

    return tokens


def _canonical_field(value: str) -> str:
    key = value.strip().lower().replace("_", "").replace("-", "")
    field = FIELD_ALIASES.get(key)
    if not field:
        supported = "Movie/Show, People, MINSEC, MAXSEC, Duration, Shot, Camera, Motion, Mood, Tag, Files"
        raise FilterQueryError(f"Unknown field {value!r}. Supported fields: {supported}")
    return field


def _read_value(tokens: list[_Token], position: int, field: str) -> tuple[str, int]:
    if position >= len(tokens) or tokens[position].kind != "value":
        raise FilterQueryError(f"Expected a value for {field}")
    if _is_keyword(tokens[position], "AND") or _is_keyword(tokens[position], "OR"):
        raise FilterQueryError(f"Expected a value for {field}")
    return tokens[position].value, position + 1


def _validate_clause(field: str, operator: str, values: list[str]) -> None:
    if field in CATEGORICAL_FIELDS and operator not in {"=", "!="}:
        raise FilterQueryError(f"{field} only supports = and !=")
    if operator != "=" and len(values) > 1:
        raise FilterQueryError("OR with multiple values currently requires the = operator")
    if field in NUMERIC_FIELDS and field != "people" and len(values) > 1:
        raise FilterQueryError(f"{field} accepts one numeric value per condition")
    if field == "people" and len(values) > 1 and operator != "=":
        raise FilterQueryError("People accepts multiple values only with the = operator")
    if field == "files" and len(values) > 1:
        raise FilterQueryError("Files accepts one boolean value per condition")
    if field in NUMERIC_FIELDS:
        for value in values:
            try:
                number = float(value)
            except ValueError as exc:
                raise FilterQueryError(f"{field} requires a number, got {value!r}") from exc
            if number < 0:
                raise FilterQueryError(f"{field} cannot be negative")
            if field == "people" and (not number.is_integer() or number > 3):
                raise FilterQueryError("People must be an integer from 0 to 3 (3 means group)")
    if field == "files":
        valid = {"true", "false", "yes", "no", "1", "0"}
        if any(value.lower() not in valid for value in values):
            raise FilterQueryError("Files must be true or false")


def _is_keyword(token: _Token, keyword: str) -> bool:
    return token.kind == "value" and token.value.upper() == keyword
