"""Subword tokenizer for the from-scratch model, trained on the documentation corpus only.

Pre-processing keeps what matters for developer text: identifiers are split at camelCase / snake_case
boundaries (``ModuleNotFoundError`` -> ``module not found error``), punctuation becomes its own token
(``cuda:0`` -> ``cuda : 0``), everything is lower-cased; then byte-pair encoding learns frequent
sub-words. Special ids: ``[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3`` (``[CLS]`` is always prepended).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

PAD, UNK, CLS, SEP = 0, 1, 2, 3
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_PUNCT = re.compile(r"([^\w\s])")
_SPACES = re.compile(r"\s+")


def normalize_for_model(text: str) -> str:
    text = _CAMEL.sub(" ", text).replace("_", " ")
    return _SPACES.sub(" ", _PUNCT.sub(r" \1 ", text)).strip().lower()


class TextTokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def train(
        cls, texts: Iterable[str], vocab_size: int = 8000, min_frequency: int = 2
    ) -> TextTokenizer:
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIALS
        )
        tok.train_from_iterator((normalize_for_model(t) for t in texts), trainer)
        return cls(tok)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Path) -> TextTokenizer:
        return cls(Tokenizer.from_file(str(path)))

    def sha256(self) -> str:
        return hashlib.sha256(self._tok.to_str().encode()).hexdigest()

    def ids(self, text: str, max_len: int) -> list[int]:
        body = self._tok.encode(normalize_for_model(text)).ids[: max_len - 1]
        return [CLS, *body]

    def encode(self, texts: Sequence[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-padded ``(ids, keep_mask)`` tensors, each ``(B, <= max_len)``."""
        rows = [self.ids(t, max_len) for t in texts]
        width = max((len(r) for r in rows), default=1)
        ids = torch.full((len(rows), width), PAD, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        return ids, ids.ne(PAD)
