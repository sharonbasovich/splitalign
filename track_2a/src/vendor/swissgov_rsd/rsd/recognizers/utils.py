# Vendored from ZurichNLP/SwissGov-RSD @ 1807a42100e742ed03d337c54c4b9ea86995f565
# Upstream file: rsd/recognizers/utils.py (MIT License, (c) 2023 University of Zurich)
# Partial vendor: `DifferenceSample` and `tokenize` are byte-identical to upstream.
# The remaining helpers (cos_sim, pairwise_dot_score, ...) were elided because they
# require `torch` at import time; they are not used by the evaluation modules.
# See VENDORED.md for the integrity manifest.
from dataclasses import dataclass
from typing import Tuple, Optional

from tokenizers.pre_tokenizers import Whitespace


@dataclass
class DifferenceSample:
    tokens_a: Tuple[str, ...]
    tokens_b: Tuple[str, ...]
    labels_a: Tuple[float, ...]
    labels_b: Optional[Tuple[float, ...]]
    annotator_tag: Optional[int] = None


def tokenize(text: str) -> Tuple[str]:
    """
    Apply Moses-like tokenization to a string.
    """
    whitespace_tokenizer = Whitespace()
    output = whitespace_tokenizer.pre_tokenize_str(text)
    # [('This', (0, 4)), ('is', (5, 7)), ('a', (8, 9)), ('test', (10, 14)), ('.', (14, 15))]
    tokens = [str(token[0]) for token in output]
    return tuple(tokens)
