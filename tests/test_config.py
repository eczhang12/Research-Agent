"""Tests for configuration parsing and debug printing.

Configuration starts as strings from `.env`, so these tests verify that helpers
turn those strings into safe Python values.
"""

from config import (
    DEBUG_COLOR,
    DEBUG_COLOR_LABEL,
    RESET_COLOR,
    debug_print,
    parse_bool,
    parse_positive_int,
)


def test_parse_bool_reads_true_values():
    """Common truthy environment strings should become `True`."""

    assert parse_bool("true")
    assert parse_bool("TRUE")
    assert parse_bool("1")
    assert parse_bool("yes")
    assert parse_bool("on")


def test_parse_bool_defaults_false_for_missing_or_false_values():
    """Missing or non-truthy environment strings should become `False`."""

    assert not parse_bool(None)
    assert not parse_bool("")
    assert not parse_bool("false")
    assert not parse_bool("no")


def test_parse_positive_int_reads_valid_values():
    """A valid positive integer string should become an integer."""

    assert parse_positive_int("5", default=3) == 5


def test_parse_positive_int_uses_default_for_invalid_values():
    """Invalid iteration settings should fall back to the safe default."""

    assert parse_positive_int(None, default=3) == 3
    assert parse_positive_int("", default=3) == 3
    assert parse_positive_int("zero", default=3) == 3
    assert parse_positive_int("0", default=3) == 3
    assert parse_positive_int("-1", default=3) == 3


def test_debug_print_prints_only_when_enabled(capsys):
    """Debug output should appear only when debug mode is enabled."""

    debug_print("visible", enabled=True)
    debug_print("hidden", enabled=False)

    output = capsys.readouterr().out
    assert f"{DEBUG_COLOR_LABEL}[debug] visible{RESET_COLOR}" in output
    assert "hidden" not in output


def test_debug_print_pretty_prints_structured_values(capsys):
    """Dictionaries should be printed as readable indented JSON."""

    debug_print("payload", {"query": "python", "count": 2}, enabled=True)

    output = capsys.readouterr().out
    assert f"{DEBUG_COLOR_LABEL}[debug] payload{RESET_COLOR}" in output
    assert f'{DEBUG_COLOR}{{\n  "query": "python",\n  "count": 2\n}}{RESET_COLOR}' in output
