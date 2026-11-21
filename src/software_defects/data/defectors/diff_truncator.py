"""Hunk-aware truncation of unified diffs under a cost budget."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

HUNK_MARKER = "@@"
MIN_CLIP_COST = 4


@dataclass(frozen=True)
class TruncationResult:
    """Outcome of truncating one diff.

    Attributes:
        text: Diff text that fits the budget (hunk order and line order are preserved).
        truncated: Whether any line or hunk was dropped or clipped.
        hunks_total: Number of hunks found in the diff.
        hunks_kept: Number of hunks whose header is kept.
        changed_total: Number of added/removed lines in the diff.
        changed_kept: Number of added/removed lines kept (clipped lines count as kept).
    """

    text: str
    truncated: bool
    hunks_total: int
    hunks_kept: int
    changed_total: int
    changed_kept: int


@dataclass
class _Hunk:
    """One hunk: an optional header line and the line indices of its body."""

    header: int | None
    body: list[int]


class HunkAwareTruncator:
    """Shrink a diff to a budget while keeping its structure.

    Policy, in order of priority:

    1. Every kept hunk keeps its ``@@`` header. Headers may use at most ``header_share`` of the budget;
       hunks beyond that are dropped from the end.
    2. The remaining budget goes to added/removed lines. It is split between hunks with a max-min fair
       share, so a small hunk keeps all its changes and a large hunk cannot starve the others.
    3. Budget that is still left is spent on context lines nearest to a kept change, then on changed
       lines that did not fit their hunk's share.

    The cost of a line comes from ``count_cost`` (tokens for the model, characters for the pre-cut), plus
    ``newline_cost`` for the line break.
    """

    def __init__(
        self,
        count_cost: Callable[[list[str]], list[int]],
        newline_cost: int = 1,
        header_share: float = 0.5,
    ) -> None:
        """Initialize the truncator.

        Args:
            count_cost: Returns the cost of each line (without its newline).
            newline_cost: Extra cost charged per kept line for the line break.
            header_share: Maximum share of the budget that hunk headers may use.
        """
        if not 0 < header_share <= 1:
            raise ValueError("header_share must be in (0, 1]")
        self._count_cost = count_cost
        self._newline_cost = newline_cost
        self._header_share = header_share

    def truncate(self, diff: str, budget: int) -> TruncationResult:
        """Truncate a diff so that its total cost is at most ``budget``.

        Args:
            diff: Unified diff text of one file.
            budget: Maximum total cost of the returned text.

        Returns:
            The truncated text and statistics about what was kept.
        """
        lines = diff.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        hunks = self._parse_hunks(lines)
        changed_total = sum(1 for line in lines if self._is_changed(line))
        hunks_total = sum(1 for hunk in hunks if hunk.header is not None)

        costs = [cost + self._newline_cost for cost in self._count_cost(lines)] if lines else []
        if sum(costs) <= budget:
            return TruncationResult(diff, False, hunks_total, hunks_total, changed_total, changed_total)

        kept, clipped = self._select(lines, costs, hunks, max(budget, 0))
        text_lines = [clipped.get(index, lines[index]) for index in sorted(kept)]
        hunks_kept = sum(1 for hunk in hunks if hunk.header is not None and hunk.header in kept)
        changed_kept = sum(1 for index in kept if self._is_changed(lines[index]))
        return TruncationResult("\n".join(text_lines), True, hunks_total, hunks_kept, changed_total, changed_kept)

    @classmethod
    def count_structure(cls, diff: str) -> tuple[int, int]:
        """Count the hunks and the added/removed lines of a diff without any tokenization.

        Args:
            diff: Unified diff text of one file.

        Returns:
            Tuple of (number of hunks, number of changed lines).
        """
        lines = diff.split("\n")
        hunks = sum(1 for line in lines if line.startswith(HUNK_MARKER))
        changed = sum(1 for line in lines if cls._is_changed(line))
        return hunks, changed

    @staticmethod
    def _is_changed(line: str) -> bool:
        return line.startswith(("+", "-"))

    @staticmethod
    def _parse_hunks(lines: list[str]) -> list[_Hunk]:
        """Group line indices into hunks; text before the first ``@@`` forms a header-less hunk."""
        hunks: list[_Hunk] = []
        current: _Hunk | None = None
        for index, line in enumerate(lines):
            if line.startswith(HUNK_MARKER):
                current = _Hunk(header=index, body=[])
                hunks.append(current)
                continue
            if current is None:
                current = _Hunk(header=None, body=[])
                hunks.append(current)
            current.body.append(index)
        return hunks

    def _select(
        self,
        lines: list[str],
        costs: list[int],
        hunks: list[_Hunk],
        budget: int,
    ) -> tuple[set[int], dict[int, str]]:
        """Choose the line indices to keep; returns them with replacement text for clipped lines."""
        kept: set[int] = set()
        clipped: dict[int, str] = {}

        admitted = self._admit_hunks(hunks, costs, budget)
        kept.update(hunk.header for hunk in admitted if hunk.header is not None)
        remaining = budget - sum(costs[index] for index in kept)

        allocation = self._fair_share(admitted, costs, lines, remaining)
        for hunk, share in zip(admitted, allocation, strict=True):
            used = self._take_changed(hunk, lines, costs, share, kept, clipped)
            remaining -= used

        remaining = self._fill_changed(admitted, lines, costs, remaining, kept)
        self._fill_context(admitted, lines, costs, remaining, kept)
        return kept, clipped

    def _admit_hunks(self, hunks: list[_Hunk], costs: list[int], budget: int) -> list[_Hunk]:
        """Keep hunks in order while their headers fit the header share of the budget."""
        header_budget = int(budget * self._header_share)
        admitted: list[_Hunk] = []
        used = 0
        for hunk in hunks:
            header_cost = costs[hunk.header] if hunk.header is not None else 0
            if used + header_cost > header_budget and admitted:
                break
            if used + header_cost > budget:
                break
            used += header_cost
            admitted.append(hunk)
        return admitted

    def _fair_share(self, hunks: list[_Hunk], costs: list[int], lines: list[str], remaining: int) -> list[int]:
        """Max-min fair split of ``remaining`` over the changed-line demand of each hunk."""
        demand = [sum(costs[i] for i in hunk.body if self._is_changed(lines[i])) for hunk in hunks]
        order = sorted(range(len(hunks)), key=lambda i: demand[i])
        share = [0] * len(hunks)
        left = max(remaining, 0)
        for position, index in enumerate(order):
            fair = left // (len(order) - position)
            share[index] = min(demand[index], fair)
            left -= share[index]
        return share

    def _take_changed(
        self,
        hunk: _Hunk,
        lines: list[str],
        costs: list[int],
        share: int,
        kept: set[int],
        clipped: dict[int, str],
    ) -> int:
        """Keep changed lines of one hunk in order while they fit ``share``; returns the cost used."""
        used = 0
        changed = [i for i in hunk.body if self._is_changed(lines[i])]
        for index in changed:
            if used + costs[index] <= share:
                kept.add(index)
                used += costs[index]
        if changed and not any(i in kept for i in changed) and share >= MIN_CLIP_COST:
            first = changed[0]
            clipped[first] = self._clip(lines[first], costs[first], share)
            kept.add(first)
            used += share
        return used

    def _clip(self, line: str, cost: int, allowed: int) -> str:
        """Cut a line to roughly ``allowed`` cost, proportionally by characters (conservative)."""
        text_cost = max(cost - self._newline_cost, 1)
        usable = max(allowed - self._newline_cost, 1)
        keep_chars = max(int(len(line) * usable / text_cost * 0.9), 1)
        return line[:keep_chars]

    def _fill_changed(
        self,
        hunks: list[_Hunk],
        lines: list[str],
        costs: list[int],
        remaining: int,
        kept: set[int],
    ) -> int:
        """Spend leftover budget on changed lines that missed their hunk's share."""
        for hunk in hunks:
            for index in hunk.body:
                if self._is_changed(lines[index]) and index not in kept and costs[index] <= remaining:
                    kept.add(index)
                    remaining -= costs[index]
        return remaining

    def _fill_context(
        self,
        hunks: list[_Hunk],
        lines: list[str],
        costs: list[int],
        remaining: int,
        kept: set[int],
    ) -> None:
        """Spend leftover budget on context lines, nearest to a kept change first."""
        candidates: list[tuple[int, int]] = []
        for hunk in hunks:
            anchors = [i for i in hunk.body if i in kept and self._is_changed(lines[i])]
            for index in hunk.body:
                if self._is_changed(lines[index]) or index in kept:
                    continue
                distance = min((abs(index - anchor) for anchor in anchors), default=len(lines))
                candidates.append((distance, index))
        for _, index in sorted(candidates):
            if costs[index] <= remaining:
                kept.add(index)
                remaining -= costs[index]
