"""SplitAlign — cross-lingual token-level semantic difference recognition.

Track 2A (UZH SwissGov-RSD) submission for Hack Apertus Online 2026.
Team: Waterloo Agent Lab — Sharon Basovich.
"""

__version__ = "0.1.0"
# Configuration version recorded in manifests/provenance. v2-lang: the judge
# prompt names the item's actual target language (v1 said "German" for every
# language). Cache identity is per call kind so the unchanged similarity and
# baseline prompts keep their v1 cache entries; every judge entry is re-keyed.
PROMPT_VERSION = "splitalign-prompts-v2-lang"
PROMPT_VERSION_BY_KIND = {
    "pair_similarity": "splitalign-prompts-v1",
    "judge_pair": PROMPT_VERSION,
    "doc_baseline": "splitalign-prompts-v1",
}
