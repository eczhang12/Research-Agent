from config import DEBUG_COLOR, DEBUG_COLOR_LABEL, RESET_COLOR, debug_print, parse_bool


def test_parse_bool_reads_true_values():
    assert parse_bool("true")
    assert parse_bool("TRUE")
    assert parse_bool("1")
    assert parse_bool("yes")
    assert parse_bool("on")


def test_parse_bool_defaults_false_for_missing_or_false_values():
    assert not parse_bool(None)
    assert not parse_bool("")
    assert not parse_bool("false")
    assert not parse_bool("no")


def test_debug_print_prints_only_when_enabled(capsys):
    debug_print("visible", enabled=True)
    debug_print("hidden", enabled=False)

    output = capsys.readouterr().out
    assert f"{DEBUG_COLOR_LABEL}[debug] visible{RESET_COLOR}" in output
    assert "hidden" not in output


def test_debug_print_pretty_prints_structured_values(capsys):
    debug_print("payload", {"query": "python", "count": 2}, enabled=True)

    output = capsys.readouterr().out
    assert f"{DEBUG_COLOR_LABEL}[debug] payload{RESET_COLOR}" in output
    assert f'{DEBUG_COLOR}{{\n  "query": "python",\n  "count": 2\n}}{RESET_COLOR}' in output
