"""v3-tag indexed judge: schema validation, scoring, cache versioning,
similarity unknown/imputation — all offline."""
import json

from splitalign import PROMPT_VERSION_BY_KIND
from splitalign.apertus import CallLogger, MockApertusClient
from splitalign.cache import DiskCache
from splitalign.judge import Judge, parse_tag_response


def _judge(tmp_path):
    c = DiskCache(tmp_path / "cache")
    return Judge(client=MockApertusClient(0), backend="mock", cache=c,
                 logger=CallLogger(None), split="dev/val", item_id="t")


# ---- parse_tag_response: strict schema -----------------------------------

def test_tag_ok_both_sides():
    r = parse_tag_response('{"a_ids": [0, 3], "b_ids": [1]}', 10, 10)
    assert r == ([0, 3], [1])


def test_tag_empty_lists_valid():
    r = parse_tag_response('{"a_ids": [], "b_ids": []}', 5, 5)
    assert r == ([], [])


def test_tag_both_keys_required():
    assert parse_tag_response('{"a_ids": [0]}', 5, 5) is None
    assert parse_tag_response('{"b_ids": []}', 5, 5) is None
    assert parse_tag_response('{"a_ids": null, "b_ids": []}', 5, 5) is None


def test_tag_rejects_non_int_bool_float_str():
    for bad in ('{"a_ids": [true], "b_ids": []}',
                '{"a_ids": [false], "b_ids": []}',
                '{"a_ids": [1.5], "b_ids": []}',
                '{"a_ids": ["2"], "b_ids": []}',
                '{"a_ids": [null], "b_ids": []}'):
        assert parse_tag_response(bad, 5, 5) is None


def test_tag_rejects_out_of_range_and_negative():
    assert parse_tag_response('{"a_ids": [5], "b_ids": []}', 5, 5) is None
    assert parse_tag_response('{"a_ids": [-1], "b_ids": []}', 5, 5) is None
    assert parse_tag_response('{"a_ids": [], "b_ids": [3]}', 5, 3) is None


def test_tag_rejects_duplicates():
    """Duplicate ids are malformed — reject, never silently dedupe."""
    assert parse_tag_response('{"a_ids": [1, 1], "b_ids": []}', 5, 5) is None
    assert parse_tag_response('{"a_ids": [], "b_ids": [2, 2]}', 5, 5) is None


def test_tag_unparseable():
    assert parse_tag_response('not json at all', 5, 5) is None
    assert parse_tag_response('[1, 2]', 5, 5) is None


# ---- judge_tag end-to-end on the mock ------------------------------------

def test_judge_tag_prompt_carries_lang_and_indexed_arrays(tmp_path):
    j = _judge(tmp_path)
    calls = []
    orig = j.client.complete
    def spy(messages, **kw):
        calls.append(messages[-1]["content"])
        return orig(messages, **kw)
    j.client.complete = spy
    for lang in ("de", "fr", "it"):
        j.judge_tag(["alpha", "beta"], ["x", "y"], lang_a="en", lang_b=lang)
    assert '"kind": "judge_tag"' in calls[0]
    # indexed arrays: every token paired with its 0-based index
    req = json.loads(calls[0].split("```json", 1)[1].split("```", 1)[0])
    assert req["a"] == [[0, "alpha"], [1, "beta"]]
    assert req["b"] == [[0, "x"], [1, "y"]]
    # each language named in the prompt body (en labels + target name)
    name = {"de": "German", "fr": "French", "it": "Italian"}
    for i, lang in enumerate(("de", "fr", "it")):
        assert name[lang] in calls[i]


def test_judge_tag_repeated_and_special_tokens(tmp_path):
    """Repeated tokens and numeric/date/negation tokens are ordinary array
    entries — addressed by index, not text."""
    j = _judge(tmp_path)
    a = ["the", "cat", "the", "2024", "not", "keine"]
    b = ["der", "2024", "nicht"]
    r = j.judge_tag(a, b, lang_a="en", lang_b="de")
    assert r.ok and r.lang_b == "de"
    assert all(0 <= i < len(a) for i in r.a_ids)
    assert all(0 <= i < len(b) for i in r.b_ids)


def test_judge_tag_cache_keyed_by_lang_and_version(tmp_path):
    j = _judge(tmp_path)
    j.judge_tag(["a"], ["b"], lang_a="en", lang_b="de")
    j2 = _judge(tmp_path)
    r = j2.judge_tag(["a"], ["b"], lang_a="en", lang_b="de")
    assert r.cached is True
    # different language -> different prompt -> different cache key
    r2 = j2.judge_tag(["a"], ["b"], lang_a="en", lang_b="fr")
    assert r2.cached is False
    # v3 judge_tag must not collide with a v2 judge_pair entry
    assert PROMPT_VERSION_BY_KIND["judge_tag"] != \
        PROMPT_VERSION_BY_KIND["judge_pair"]


def test_judge_tag_invalid_then_repair_then_invalid(tmp_path):
    """Response invalid twice -> ok=False, counted, never valid coverage."""
    j = _judge(tmp_path)
    j.client.complete = lambda *a, **kw: type("R", (), {
        "text": "garbage {{{", "model": "m", "backend": "x",
        "prompt_tokens": 1, "completion_tokens": 1, "latency_ms": 0})()
    r = j.judge_tag(["a"], ["b"], lang_a="en", lang_b="de")
    assert r.ok is False and r.repairs == 1


# ---- similarity unknown / imputation --------------------------------------

def test_similarity_unknown_imputed_row_mean(tmp_path):
    """An unreturned cell after one repair gets the observed ROW mean and is
    counted as imputed, never silently scored 0."""
    j = _judge(tmp_path)

    class Partial:
        backend = "apertus"
        model = "m"
        calls = 0

        def complete(self, messages, **kw):
            self.calls += 1
            return type("R", (), {
                "model": "m", "backend": "a", "prompt_tokens": 1,
                "completion_tokens": 1, "latency_ms": 0,
                "text": json.dumps({"scores": [
                    {"i": 0, "j": 0, "score": 5},
                    {"i": 0, "j": 1, "score": 3},
                    {"i": 1, "j": 1, "score": 4}]})})()

    j.client = Partial()
    a, b = ["x", "y"], ["p", "q"]
    sim = j.similarity_matrix(a, b)
    # (1,0) missing after repair: row 1 observed mean = 4/5 = 0.8
    assert abs(sim[1][0] - 0.8) < 1e-6
    assert j.sim_unknown[1][0] is True
    assert j.sim_unknown_cells == 1 and j.sim_imputed_cells == 1
    assert j.client.calls == 2  # initial + exactly one repair


def test_similarity_all_missing_row_col_uses_default(tmp_path):
    j = _judge(tmp_path)

    class Empty:
        backend = "apertus"
        model = "m"

        def complete(self, messages, **kw):
            return type("R", (), {
                "model": "m", "backend": "a", "prompt_tokens": 1,
                "completion_tokens": 1, "latency_ms": 0,
                "text": json.dumps({"scores": []})})()

    j.client = Empty()
    sim = j.similarity_matrix(["x", "y"], ["p", "q"])
    # everything missing -> declared neutral prior 0.5, prior-imputed
    assert sim == [[0.5, 0.5], [0.5, 0.5]]
    assert j.sim_unknown_cells == 4 and j.sim_imputed_cells == 4
