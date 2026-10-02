"""SplitAlign — cross-lingual token-level semantic difference recognition.

Track 2A (UZH SwissGov-RSD) submission for Hack Apertus Online 2026.
Team: Waterloo Agent Lab — Sharon Basovich.
"""

__version__ = "0.2.0"
# Configuration version recorded in manifests/provenance. v2-lang: the judge
# prompt names the item's actual target language (v1 said "German" for every
# language). Cache identity is per call kind so the unchanged similarity and
# baseline prompts keep their v1 cache entries; every judge entry is re-keyed.
PROMPT_VERSION = "splitalign-prompts-v2-lang"
PROMPT_VERSION_BY_KIND = {
    "pair_similarity": "splitalign-prompts-v1",
    "judge_pair": PROMPT_VERSION,
    "judge_tag": "splitalign-prompts-v3-tag",
    "doc_baseline": "splitalign-prompts-v1",
}

# Effective method versions reported in manifests/evals (provenance only).
METHOD_VERSIONS = {
    "splitalign": PROMPT_VERSION_BY_KIND["judge_tag"],
    # the baseline IS the repair-exhausted whole-document fallback_label=5
    # policy on an unchanged v1 prompt — a reference, not a strong comparator
    "baseline": "baseline-v1-repair-exhausted",
}
