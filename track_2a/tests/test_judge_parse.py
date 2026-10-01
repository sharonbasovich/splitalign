from splitalign.judge import extract_first_json, parse_judgment


def test_extract_fenced():
    obj = extract_first_json('```json\n{"a": 1}\n```')
    assert obj == {"a": 1}


def test_extract_prose_wrapped():
    obj = extract_first_json('Here is the result: {"semantic_difference": 3} done')
    assert obj["semantic_difference"] == 3


def test_extract_nested():
    obj = extract_first_json('{"x": {"y": [1, {"z": 2}]}}')
    assert obj["x"]["y"][1]["z"] == 2


def test_extract_none():
    assert extract_first_json("no json here") is None
    assert extract_first_json("{unclosed: 1") is None


def test_parse_judgment_ok():
    j = parse_judgment('{"semantic_difference": 2, "differing_spans": {"side_a": ["foo bar"], "side_b": []}}',
                       "xx foo bar yy", "zz")
    assert j is not None and j.difference == 2 and j.spans_a == ["foo bar"]


def test_parse_judgment_drops_nonverbatim_spans():
    j = parse_judgment('{"semantic_difference": 1, "differing_spans": {"side_a": ["not present"], "side_b": ["also not"]}}',
                       "real text", "other text")
    assert j.spans_a == [] and j.spans_b == []


def test_parse_judgment_rejects_garbage():
    assert parse_judgment("totally not json", "a", "b") is None
    assert parse_judgment('{"semantic_difference": 9}', "a", "b") is None
    assert parse_judgment('{"semantic_difference": "high"}', "a", "b") is None
