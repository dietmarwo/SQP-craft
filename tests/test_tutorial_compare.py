"""The comparison the tutorial workflows use reports differences instead of failing."""

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tutorials"
    / "_support"
    / "compare_results.py"
)
spec = importlib.util.spec_from_file_location("compare_results", SCRIPT)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def test_null_against_a_number_is_a_difference():
    assert compare.differences({"profit": 3.5}, {"profit": None}) == [
        "/profit: 3.5 != None"
    ]
    assert compare.differences({"profit": None}, {"profit": 3.5}) == [
        "/profit: None != 3.5"
    ]


def test_floats_never_match_booleans_or_strings():
    assert compare.differences([1.0], [True]) == ["[0]: 1.0 != True"]
    assert compare.differences([1.0], ["1.0"]) == ["[0]: 1.0 != '1.0'"]


def test_floats_agree_to_one_part_in_a_million():
    assert compare.differences({"x": 1.0}, {"x": 1.0 + 5e-7}) == []
    assert compare.differences({"x": 1e-9}, {"x": 5e-7}) == []
    assert compare.differences({"x": 1.0}, {"x": 1.00001}) == ["/x: 1.0 != 1.00001"]
    assert compare.differences({"x": 2}, {"x": 2.0}) == []
    assert compare.differences({"x": float("nan")}, {"x": float("nan")}) == []
    assert compare.differences({"x": float("nan")}, {"x": 1.0}) == ["/x: nan != 1.0"]


def test_integers_and_strings_match_exactly_and_timings_are_ignored():
    assert compare.differences({"n": 980}, {"n": 983}) == ["/n: '980' != '983'"]
    assert compare.differences({"stop": "a"}, {"stop": "b"}) == ["/stop: 'a' != 'b'"]
    assert (
        compare.differences(
            {"seconds": 1.0, "wall_time": 2}, {"seconds": 9.0, "wall_time": 3}
        )
        == []
    )
    assert compare.differences({"a": 1}, {"b": 1}) == [
        "/a: present in only one summary",
        "/b: present in only one summary",
    ]
    assert compare.differences({"a": [1, 2]}, {"a": [1]}) == ["/a: length 2 != 1"]
