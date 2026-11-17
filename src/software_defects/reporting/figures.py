
"""Reusable utilities for creating and exporting matplotlib figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DEFAULT_PALETTE = {
    "data": "#dbe9f6",
    "model": "#fde8c8",
    "eval": "#dcefd9",
    "accent": "#f6d5d5",
    "edge": "#44546a",
    "text": "#1f2937",
}


def _pyplot() -> ModuleType:
    """Import matplotlib lazily using the non-interactive Agg backend."""
    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except ImportError as error:
        raise ImportError(
            "Matplotlib is required. Install it using `uv add matplotlib`."
        ) from error

    return plt


class FigureBuilder:
    """Reusable utility for creating and exporting matplotlib figures."""

    def __init__(
        self,
        output_dir: Path | str,
        palette: Mapping[str, str] | None = None,
        style: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initialize the figure builder.

        Args:
            output_dir: Directory where generated figures are stored.
            palette: Optional color palette for drawing primitives.
            style: Optional matplotlib rcParams configuration.
        """
        self.output_dir = Path(output_dir)
        self.palette = dict(palette or DEFAULT_PALETTE)

        self.style = {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            **(style or {}),
        }

    def save(
        self,
        fig: Figure,
        name: str,
        formats: Sequence[str] = ("png", "svg"),
        dpi: int = 200,
        close: bool = True,
    ) -> dict[str, Path]:
        """
        Export a matplotlib figure into one or more formats.

        Args:
            fig: Matplotlib figure instance.
            name: Output filename without extension.
            formats: Supported output formats.
            dpi: Resolution for raster formats.
            close: Whether to close the figure after exporting.

        Returns:
            Dictionary mapping file formats to their output paths.
        """
        if not name or Path(name).name != name:
            raise ValueError("name must be a non-empty filename without a path.")

        supported_formats = {"png", "svg", "pdf", "jpg", "jpeg"}

        invalid_formats = set(formats) - supported_formats
        if invalid_formats:
            raise ValueError(
                f"Unsupported formats: {sorted(invalid_formats)}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        _pyplot()
        output_paths: dict[str, Path] = {}

        try:
            with plt.rc_context(self.style):
                for fmt in formats:
                    path = self.output_dir / f"{name}.{fmt}"

                    fig.savefig(
                        path,
                        format=fmt,
                        dpi=dpi,
                        bbox_inches="tight",
                    )

                    output_paths[fmt] = path

        finally:
            if close:
                plt.close(fig)

        return output_paths

    def draw_box(
        self,
        ax: Axes,
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        color: str | None = None,
        fontsize: float = 8,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Draw a rounded rectangle with centered text."""
        from matplotlib.patches import FancyBboxPatch

        edge_color = self.palette.get("edge", "#44546a")
        text_color = self.palette.get("text", "#1f2937")

        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fc=color or "white",
                ec=edge_color,
                lw=1.0,
                **kwargs,
            )
        )

        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
        )

    def draw_arrow(
        self,
        ax: Axes,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Draw a straight directional arrow."""
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color or self.palette.get("edge", "#44546a"),
                "lw": 1.1,
                **kwargs,
            },
        )

    def draw_elbow(
        self,
        ax: Axes,
        start: tuple[float, float],
        end: tuple[float, float],
        mid_y: float,
        color: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Draw an orthogonal connector with a directional endpoint."""
        edge_color = color or self.palette.get("edge", "#44546a")

        x1, y1 = start
        x2, y2 = end

        ax.plot(
            [x1, x1, x2],
            [y1, mid_y, mid_y],
            color=edge_color,
            lw=1.1,
            solid_capstyle="round",
            **kwargs,
        )

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x2, mid_y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": edge_color,
                "lw": 1.1,
            },
        )
