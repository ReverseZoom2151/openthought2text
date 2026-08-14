"""Deterministic train-only tokenizer artifacts for constrained baselines."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .schema import NeuralTextSample


TOKENIZER_VERSION = "1.0"
TOKENIZER_KIND = "openthought2text.train_text_tokenizer"
_SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class UnknownTokenPolicy(str, Enum):
    ERROR = "error"
    UNK = "unk"


def tokenize_text(text: str) -> tuple[str, ...]:
    """Case-fold and tokenize text without learned or external dependencies."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("tokenizer input text must be a non-empty string")
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _detokenize(tokens: Iterable[str]) -> str:
    text = ""
    attach_left = {".", ",", "!", "?", ";", ":", ")", "]", "}"}
    attach_right = {"(", "[", "{"}
    for token in tokens:
        if not text:
            text = token
        elif token in attach_left:
            text += token
        elif text[-1] in attach_right:
            text += token
        else:
            text += " " + token
    return text


@dataclass(frozen=True, slots=True)
class TrainTextTokenizer:
    """Versioned vocabulary fit from an explicitly declared train partition."""

    vocabulary: tuple[str, ...]
    unknown_policy: UnknownTokenPolicy
    fit_sample_ids: tuple[str, ...]
    fit_text_checksum: str
    version: str = TOKENIZER_VERSION
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    def __post_init__(self) -> None:
        if self.version != TOKENIZER_VERSION:
            raise ValueError(f"unsupported tokenizer version: {self.version!r}")
        special_ids = (self.pad_id, self.bos_id, self.eos_id, self.unk_id)
        if special_ids != tuple(range(len(_SPECIAL_TOKENS))):
            raise ValueError("special token IDs must be pad=0, bos=1, eos=2, unk=3")
        if len(self.vocabulary) < len(_SPECIAL_TOKENS):
            raise ValueError("vocabulary is missing required special tokens")
        if self.vocabulary[: len(_SPECIAL_TOKENS)] != _SPECIAL_TOKENS:
            raise ValueError("vocabulary special tokens are missing or reordered")
        if len(self.vocabulary) != len(set(self.vocabulary)):
            raise ValueError("vocabulary tokens must be unique")
        if any(not isinstance(token, str) or not token for token in self.vocabulary):
            raise ValueError("vocabulary tokens must be non-empty strings")
        if not isinstance(self.unknown_policy, UnknownTokenPolicy):
            raise ValueError("unknown_policy must be an UnknownTokenPolicy")
        if not self.fit_sample_ids or len(set(self.fit_sample_ids)) != len(self.fit_sample_ids):
            raise ValueError("tokenizer requires unique train fit_sample_ids")
        if not re.fullmatch(r"[0-9a-f]{64}", self.fit_text_checksum):
            raise ValueError("fit_text_checksum must be a lowercase SHA-256 digest")

    @property
    def special_ids(self) -> dict[str, int]:
        return {
            "pad": self.pad_id,
            "bos": self.bos_id,
            "eos": self.eos_id,
            "unk": self.unk_id,
        }

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.vocabulary)}

    @property
    def checksum(self) -> str:
        return _canonical_hash(self.to_dict(include_checksum=False))

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> tuple[int, ...]:
        """Encode text using the declared unknown-token policy."""
        mapping = self.token_to_id
        encoded: list[int] = []
        if add_bos:
            encoded.append(self.bos_id)
        for token in tokenize_text(text):
            token_id = mapping.get(token)
            if token_id is None:
                if self.unknown_policy == UnknownTokenPolicy.ERROR:
                    raise ValueError(f"unknown token under error policy: {token!r}")
                token_id = self.unk_id
            encoded.append(token_id)
        if add_eos:
            encoded.append(self.eos_id)
        return tuple(encoded)

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        """Decode validated IDs; special IDs are skipped unless explicitly requested."""
        tokens: list[str] = []
        special = set(self.special_ids.values())
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError("token IDs must be integers")
            if token_id < 0 or token_id >= len(self.vocabulary):
                raise ValueError(f"token ID is outside vocabulary: {token_id}")
            if skip_special_tokens and token_id in special:
                continue
            tokens.append(self.vocabulary[token_id])
        return _detokenize(tokens)

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": TOKENIZER_KIND,
            "version": self.version,
            "vocabulary": list(self.vocabulary),
            "unknown_policy": self.unknown_policy.value,
            "special_ids": self.special_ids,
            "fit_sample_ids": list(self.fit_sample_ids),
            "fit_text_checksum": self.fit_text_checksum,
        }
        if include_checksum:
            data["checksum"] = self.checksum
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainTextTokenizer":
        if data.get("kind") != TOKENIZER_KIND:
            raise ValueError("not an OpenThought2Text train text tokenizer artifact")
        special_ids = data.get("special_ids")
        if not isinstance(special_ids, Mapping):
            raise ValueError("tokenizer artifact is missing special_ids")
        try:
            tokenizer = cls(
                vocabulary=tuple(data["vocabulary"]),
                unknown_policy=UnknownTokenPolicy(data["unknown_policy"]),
                fit_sample_ids=tuple(str(sample_id) for sample_id in data["fit_sample_ids"]),
                fit_text_checksum=str(data["fit_text_checksum"]),
                version=str(data.get("version", TOKENIZER_VERSION)),
                pad_id=int(special_ids["pad"]),
                bos_id=int(special_ids["bos"]),
                eos_id=int(special_ids["eos"]),
                unk_id=int(special_ids["unk"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid train text tokenizer artifact") from error
        expected = data.get("checksum")
        if expected is not None and expected != tokenizer.checksum:
            raise ValueError("tokenizer artifact checksum does not match its contents")
        return tokenizer


def fit_train_text_tokenizer(
    samples: Iterable[NeuralTextSample],
    *,
    unknown_policy: UnknownTokenPolicy | str | None = None,
) -> TrainTextTokenizer:
    """Fit a vocabulary from all and only samples explicitly declared ``train``."""
    if unknown_policy is None:
        raise ValueError("unknown_policy must be explicitly declared as 'error' or 'unk'")
    try:
        policy = UnknownTokenPolicy(unknown_policy)
    except ValueError as error:
        raise ValueError("unknown_policy must be 'error' or 'unk'") from error
    rows = tuple(samples)
    if not rows:
        raise ValueError("cannot fit a tokenizer without samples")
    non_train = [sample.sample_id for sample in rows if sample.split != "train"]
    if non_train:
        raise ValueError("tokenizer fit received non-train samples: " + ", ".join(sorted(non_train)))
    missing_target = [sample.sample_id for sample in rows if sample.target is None]
    if missing_target:
        raise ValueError("tokenizer fit needs target text for: " + ", ".join(sorted(missing_target)))
    sample_ids = [sample.sample_id for sample in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("tokenizer fit sample IDs must be unique")
    counts: Counter[str] = Counter()
    fingerprint_rows: list[dict[str, str]] = []
    for sample in rows:
        assert sample.target is not None
        counts.update(tokenize_text(sample.target.text))
        fingerprint_rows.append({"sample_id": sample.sample_id, "text": sample.target.fingerprint})
    ordered_tokens = tuple(sorted(counts, key=lambda token: (-counts[token], token)))
    return TrainTextTokenizer(
        vocabulary=_SPECIAL_TOKENS + ordered_tokens,
        unknown_policy=policy,
        fit_sample_ids=tuple(sorted(sample_ids)),
        fit_text_checksum=_canonical_hash({"train_targets": sorted(fingerprint_rows, key=lambda row: row["sample_id"])}),
    )


def write_train_text_tokenizer(path: str | Path, tokenizer: TrainTextTokenizer) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(tokenizer.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_train_text_tokenizer(path: str | Path) -> TrainTextTokenizer:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid train text tokenizer artifact: {source}") from error
    if not isinstance(data, dict):
        raise ValueError("train text tokenizer artifact must be a JSON object")
    return TrainTextTokenizer.from_dict(data)
