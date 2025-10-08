from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Any, Hashable, Sequence

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

if TYPE_CHECKING:  # Avoid runtime import to break circular deps with env/containers
    from gfn.containers.trajectories import Trajectories


class Chunker(ABC):
    """Abstract base class for chunkers that propose new vocab tokens.

    Chunkers operate on trajectories and environment context and return
    a sequence of token keys (any Hashable) to be added to the env vocab.
    """

    @abstractmethod
    def propose_tokens(
        self,
        env: "Any",
        trajectories: Trajectories,
        n_tokens_to_add: int,
        remove_old: bool,
    ) -> Sequence[Hashable]:
        raise NotImplementedError


class UniformChunker(Chunker):
    """Proposes random bigrams of current non-exit tokens as tuples of ints."""

    def propose_tokens(
        self,
        env: "Any",
        trajectories: Trajectories,
        n_tokens_to_add: int,
        remove_old: bool,
    ) -> Sequence[Hashable]:
        # Build non-exit pool from current vocab ids.
        non_exit_ids = [i for i in range(env.n_actions) if i != env.exit_token_id]
        seen = set(env.vocab)
        out: set[Hashable] = set()
        while len(out) < n_tokens_to_add and len(out) < 10_000:
            a, b = random.choice(non_exit_ids), random.choice(non_exit_ids)
            candidate = (a, b)
            if candidate not in seen:
                out.add(candidate)
        return list(out)


class _StringMapping:
    """Utility to map env keys to strings suitable for tokenizers."""

    def __init__(self, delimiter: str = "") -> None:
        self.delimiter = delimiter

    def key_to_str(self, key: Hashable) -> str:
        if isinstance(key, tuple):
            return self.delimiter.join(str(x) for x in key)
        return str(key)


class BPEChunker(Chunker):
    def __init__(self, unk_token: str = "[UNK]", delimiter: str = "") -> None:
        self.unk_token = unk_token
        self.mapper = _StringMapping(delimiter=delimiter)

    def propose_tokens(
        self,
        env: "Any",
        trajectories: Trajectories,
        n_tokens_to_add: int,
        remove_old: bool,
        min_frequency: int = 5,
    ) -> Sequence[Hashable]:
        # Build corpus strings from trajectories via env tokenizer
        corpus = env.trajectories_to_token_strings(trajectories)

        # Build initial vocab from current env keys mapped to strings
        vocab_dict = {self.mapper.key_to_str(k): i for i, k in enumerate(env.vocab)}
        tokenizer = Tokenizer(BPE(vocab_dict, [], unk_token=self.unk_token))

        target_vocab_size = len(env.vocab) - 1 + n_tokens_to_add
        trainer = BpeTrainer(
            vocab_size=target_vocab_size,  # type: ignore
            special_tokens=[self.unk_token],  # type: ignore
            min_frequency=min_frequency,  # type: ignore
        )
        tokenizer.train_from_iterator(corpus, trainer=trainer)

        # Take the most common new tokens.
        base_vocab = set(vocab_dict.keys())
        encodings = tokenizer.encode_batch(corpus)
        counts = Counter()
        for enc in encodings:
            for tok in enc.tokens:
                if tok not in base_vocab and tok != self.unk_token and len(tok) > 0:
                    counts[tok] += 1

        top_new = [tok for tok, _ in counts.most_common(n_tokens_to_add)]
        return top_new


class WordPieceChunker(Chunker):
    def __init__(self, unk_token: str = "[UNK]", delimiter: str = "") -> None:
        self.unk_token = unk_token
        self.mapper = _StringMapping(delimiter=delimiter)

    def propose_tokens(
        self,
        env: "Any",
        trajectories: Trajectories,
        n_tokens_to_add: int,
        remove_old: bool,
        min_frequency: int = 5,
    ) -> Sequence[Hashable]:
        corpus = env.trajectories_to_token_strings(trajectories)
        vocab_dict = {self.mapper.key_to_str(k): i for i, k in enumerate(env.vocab)}
        tokenizer = Tokenizer(
            WordPiece(
                vocab=vocab_dict,
                unk_token=self.unk_token,
                max_input_chars_per_word=100,
            )
        )
        target_vocab_size = len(env.vocab) - 1 + n_tokens_to_add
        trainer = WordPieceTrainer(
            vocab_size=target_vocab_size,
            continuing_subword_prefix="##",  # Defined prefix (removed later).
            special_tokens=[self.unk_token],
            min_frequency=min_frequency,
        )
        tokenizer.train_from_iterator(corpus, trainer=trainer)

        # Take the most common new tokens.
        base_vocab = set(vocab_dict.keys())
        encodings = tokenizer.encode_batch(corpus)
        counts = Counter()
        for enc in encodings:
            for tok in enc.tokens:
                if tok not in base_vocab and tok != self.unk_token and len(tok) > 0:
                    counts[tok.lstrip("##")] += 1  # Remove prefix if present.

        top_new = [tok for tok, _ in counts.most_common(n_tokens_to_add)]
        return top_new
