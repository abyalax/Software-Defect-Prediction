"""Build the model input "file path + diff" within the token limit of a model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from software_defects.config import DiffInputConfig
from software_defects.data.defectors.diff_truncator import HunkAwareTruncator, TruncationResult

PATH_SEPARATOR = "\n"
PATH_ELLIPSIS = ".../"


@dataclass(frozen=True)
class DiffInput:
    """Model input of one file change plus what the preprocessing did to it.

    Attributes:
        text: The model input: the path on the first line, then the diff.
        truncated: Whether any part of the diff was dropped or clipped (including the pre-cut).
        pre_cut: Whether the diff was longer than the pre-cut limit and was cut before tokenization.
        fallback: Whether the text still exceeded the limit after all refits (the tokenizer then hard-truncates).
        path_shortened: Whether the path itself had to be shortened to fit ``path_max_tokens``.
        full_tokens: Token count of the text before token-level truncation (a lower bound when ``pre_cut``).
        hunks_total: Number of hunks of the (pre-cut) diff.
        hunks_kept: Number of hunks whose header is in the text.
        changed_total: Number of added/removed lines of the (pre-cut) diff.
        changed_kept: Number of added/removed lines in the text.
    """

    text: str
    truncated: bool
    pre_cut: bool
    fallback: bool
    path_shortened: bool
    full_tokens: int
    hunks_total: int
    hunks_kept: int
    changed_total: int
    changed_kept: int


class DiffInputBuilder:
    """Assemble ``path + diff`` inputs that fit ``max_length`` tokens.

    The path is always the first line and is never cut by the diff truncation. The diff is first pre-cut by
    characters when it is extremely long, then shrunk hunk-aware to the token budget that is left after
    the path and the model's special tokens.
    """

    def __init__(
        self,
        count_tokens: Callable[[list[str]], list[int]],
        special_tokens: int,
        max_length: int,
        config: DiffInputConfig | None = None,
    ) -> None:
        """Initialize the builder.

        Args:
            count_tokens: Token count of each text, without special tokens.
            special_tokens: Number of special tokens the model adds to a single sequence.
            max_length: Maximum sequence length of the model, special tokens included.
            config: Input settings; defaults to ``DiffInputConfig()``.
        """
        self._count_tokens = count_tokens
        self._special_tokens = special_tokens
        self._max_length = max_length
        self._config = config or DiffInputConfig()
        self._token_truncator = HunkAwareTruncator(count_tokens, header_share=self._config.header_share)
        self._char_truncator = HunkAwareTruncator(
            lambda lines: [len(line) for line in lines], header_share=self._config.header_share
        )

    def build(self, paths: Sequence[str], diffs: Sequence[str]) -> list[DiffInput]:
        """Build one input per (path, diff) pair.

        Args:
            paths: File path of each change.
            diffs: Diff of each change.

        Returns:
            One ``DiffInput`` per pair, in the same order.
        """
        if len(paths) != len(diffs):
            raise ValueError("paths and diffs must have the same length")

        path_texts = self._prepare_paths(paths)
        cut_diffs = [self._pre_cut(str(diff or "")) for diff in diffs]
        texts = [self._join(path_texts[str(path)][0], cut.text) for path, cut in zip(paths, cut_diffs, strict=True)]
        full_tokens = self._count_tokens(texts)

        results: list[DiffInput] = []
        for path, cut, text, tokens in zip(paths, cut_diffs, texts, full_tokens, strict=True):
            path_text, path_shortened = path_texts[str(path)]
            pre_cut = cut.truncated
            if tokens + self._special_tokens <= self._max_length:
                results.append(self._result(text, cut, cut, tokens, pre_cut, False, path_shortened, pre_cut))
                continue
            fitted, fallback = self._fit(path_text, cut.text)
            results.append(
                self._result(
                    self._join(path_text, fitted.text), cut, fitted, tokens, pre_cut, fallback, path_shortened, True
                )
            )
        return results

    def _join(self, path_text: str, diff: str) -> str:
        return f"{path_text}{PATH_SEPARATOR}{diff}" if self._config.include_path else diff

    def _prepare_paths(self, paths: Sequence[str]) -> dict[str, tuple[str, bool]]:
        """Map each unique path to its (possibly shortened) text and a shortened flag."""
        unique = sorted({str(path) for path in paths})
        if not self._config.include_path:
            return {path: ("", False) for path in unique}
        counts = self._count_tokens(unique)
        prepared: dict[str, tuple[str, bool]] = {}
        for path, count in zip(unique, counts, strict=True):
            prepared[path] = (path, False) if count <= self._config.path_max_tokens else (self._shorten(path), True)
        return prepared

    def _shorten(self, path: str) -> str:
        """Drop leading directories until the path fits ``path_max_tokens`` (the filename is always kept)."""
        parts = path.split("/")
        for start in range(1, len(parts)):
            candidate = PATH_ELLIPSIS + "/".join(parts[start:])
            if self._count_tokens([candidate])[0] <= self._config.path_max_tokens:
                return candidate
        return PATH_ELLIPSIS + parts[-1]

    def _pre_cut(self, diff: str) -> TruncationResult:
        """Cut extremely long diffs by characters, hunk-aware, so that tokenization stays cheap."""
        limit = self._config.pre_cut_chars
        if len(diff) <= limit:
            hunks, changed = HunkAwareTruncator.count_structure(diff)
            return TruncationResult(diff, False, hunks, hunks, changed, changed)
        return self._char_truncator.truncate(diff, limit)

    def _fit(self, path_text: str, diff: str) -> tuple[TruncationResult, bool]:
        """Shrink ``diff`` until ``path + diff`` fits the limit; the bool tells when it could not."""
        overhead = self._special_tokens
        if self._config.include_path:
            overhead += self._count_tokens([path_text])[0] + 1
        budget = self._max_length - overhead
        result = self._token_truncator.truncate(diff, budget)
        for _ in range(self._config.max_refits):
            total = self._count_tokens([self._join(path_text, result.text)])[0] + self._special_tokens
            if total <= self._max_length:
                return result, False
            budget -= total - self._max_length
            result = self._token_truncator.truncate(diff, budget)
        total = self._count_tokens([self._join(path_text, result.text)])[0] + self._special_tokens
        return result, total > self._max_length

    @staticmethod
    def _result(
        text: str,
        cut: TruncationResult,
        fitted: TruncationResult,
        full_tokens: int,
        pre_cut: bool,
        fallback: bool,
        path_shortened: bool,
        truncated: bool,
    ) -> DiffInput:
        # Totals describe the diff as received; kept counts come from the last truncation stage
        return DiffInput(
            text=text,
            truncated=truncated,
            pre_cut=pre_cut,
            fallback=fallback,
            path_shortened=path_shortened,
            full_tokens=full_tokens,
            hunks_total=cut.hunks_total,
            hunks_kept=fitted.hunks_kept,
            changed_total=cut.changed_total,
            changed_kept=fitted.changed_kept,
        )
