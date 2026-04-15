"""
Plot the 2D Fermi-Hubbard model.

This file is linked to ``Hubbard.py`` and uses the same shell-based geometry:
- shell 1: nearest neighbors
- shell 2: diagonal next-nearest neighbors
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


Bond = tuple[int, int]


def _load_hubbard_module():
    here = Path(__file__).resolve()
    for candidate in ("Hubbard.py", "Hubbard_staging.py"):
        path = here.with_name(candidate)
        if path.exists() and path != here:
            spec = importlib.util.spec_from_file_location("hubbard_linked", path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        "Could not locate Hubbard.py or Hubbard_staging.py "
        f"next to {here}."
    )


_HUB = _load_hubbard_module()
hubbard_index = _HUB.hubbard_index
hubbard_geometry = _HUB.hubbard_geometry


def _spatial_positions(
    Lx: int,
    Ly: int,
    *,
    x_spacing: float = 1.8,
    y_spacing: float = 1.5,
) -> dict[int, tuple[float, float]]:
    return {
        hubbard_index(x, y, Ly): (x * x_spacing, -y * y_spacing)
        for x in range(Lx)
        for y in range(Ly)
    }


def _shift_positions(
    positions: dict[int, tuple[float, float]],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> dict[int, tuple[float, float]]:
    return {site: (x + dx, y + dy) for site, (x, y) in positions.items()}


def _draw_bonds(
    ax: Any,
    positions: dict[int, tuple[float, float]],
    bonds: list[Bond],
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    wrap_curvature: float = 0.22,
) -> None:
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    x_span = max(xs) - min(xs) if xs else 0.0
    y_span = max(ys) - min(ys) if ys else 0.0
    wrap_x = 0.75 * x_span
    wrap_y = 0.75 * y_span

    for i, j in bonds:
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        is_wrap = (x_span > 0.0 and abs(x1 - x2) > wrap_x) or (
            y_span > 0.0 and abs(y1 - y2) > wrap_y
        )

        if is_wrap:
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x2, y2),
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "linestyle": ":",
                    "linewidth": linewidth,
                    "alpha": alpha,
                    "connectionstyle": f"arc3,rad={wrap_curvature}",
                },
            )
        else:
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )


def plot_hubbard_layout(
    *,
    Lx: int = 4,
    Ly: int = 4,
    pbc_x: bool = False,
    pbc_y: bool = False,
    t1: float = 1.0,
    t2: float = 0.0,
    U: float = 4.0,
    show_labels: bool = True,
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
    show: bool = True,
):
    """
    Plot the 2D Fermi-Hubbard model.

    Left panel: spatial lattice geometry.
    Right panel: spin-up and spin-down orbital layers linked by onsite U.
    """
    geometry = hubbard_geometry(Lx=Lx, Ly=Ly, pbc_x=pbc_x, pbc_y=pbc_y)
    positions = _spatial_positions(Lx, Ly)

    if figsize is None:
        figsize = (max(11.0, 2.2 * Lx + 3.0), max(5.2, 1.7 * Ly + 2.8))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.0, 1.1]},
    )
    ax_left, ax_right = axes

    shell_1 = geometry["shell_1"]
    shell_2 = geometry["shell_2"]

    left_specs = [
        ("shell 1", shell_1, "black", "-", 2.2, t1 != 0.0),
        ("shell 2", shell_2, "tab:green", "--", 2.0, t2 != 0.0),
    ]
    for _, bonds, color, linestyle, linewidth, enabled in left_specs:
        if enabled and bonds:
            _draw_bonds(
                ax_left,
                positions,
                bonds,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )

    xs = [positions[i][0] for i in sorted(positions)]
    ys = [positions[i][1] for i in sorted(positions)]
    ax_left.scatter(xs, ys, s=90, c="white", edgecolors="black", linewidths=1.8, zorder=3)

    if show_labels:
        for x in range(Lx):
            for y in range(Ly):
                site = hubbard_index(x, y, Ly)
                px, py = positions[site]
                ax_left.text(
                    px,
                    py + 0.18,
                    f"({x},{y})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="dimgray",
                )

    left_handles = [
        Line2D([0], [0], color=color, lw=linewidth, ls=linestyle, label=label)
        for label, _, color, linestyle, linewidth, enabled in left_specs
        if enabled
    ]
    if left_handles:
        ax_left.legend(handles=left_handles, loc="upper center", ncol=2, frameon=False)

    ax_left.set_title("Spatial Geometry")
    ax_left.set_aspect("equal")
    ax_left.axis("off")

    dn_positions = _shift_positions(positions, dx=-0.10, dy=-0.22)
    up_positions = _shift_positions(positions, dx=0.34, dy=0.30)

    right_specs = [
        ("up t1", up_positions, shell_1, "tab:blue", "-", 2.1, t1),
        ("up t2", up_positions, shell_2, "tab:blue", "--", 1.8, t2),
        ("dn t1", dn_positions, shell_1, "tab:red", "-", 2.1, t1),
        ("dn t2", dn_positions, shell_2, "tab:red", "--", 1.8, t2),
    ]
    for _, pos_map, bonds, color, linestyle, linewidth, coupling in right_specs:
        if coupling != 0.0 and bonds:
            _draw_bonds(
                ax_right,
                pos_map,
                bonds,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )

    if U != 0.0:
        for site in sorted(positions):
            x1, y1 = dn_positions[site]
            x2, y2 = up_positions[site]
            ax_right.plot(
                [x1, x2],
                [y1, y2],
                color="black",
                linestyle=":",
                linewidth=1.5,
                alpha=0.85,
                zorder=2,
            )

    ax_right.scatter(
        [up_positions[i][0] for i in sorted(up_positions)],
        [up_positions[i][1] for i in sorted(up_positions)],
        s=82,
        c="white",
        edgecolors="tab:blue",
        linewidths=1.8,
        zorder=3,
    )
    ax_right.scatter(
        [dn_positions[i][0] for i in sorted(dn_positions)],
        [dn_positions[i][1] for i in sorted(dn_positions)],
        s=82,
        c="white",
        edgecolors="tab:red",
        linewidths=1.8,
        zorder=3,
    )

    if show_labels:
        for site, (x, y) in positions.items():
            ax_right.text(
                x + 0.12,
                y + 0.04,
                f"{site}",
                ha="center",
                va="center",
                fontsize=7,
                color="dimgray",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9},
            )

    right_handles = [
        Line2D([0], [0], color="tab:blue", lw=2.1, ls="-", label="spin-up t1"),
        Line2D([0], [0], color="tab:blue", lw=1.8, ls="--", label="spin-up t2"),
        Line2D([0], [0], color="tab:red", lw=2.1, ls="-", label="spin-down t1"),
        Line2D([0], [0], color="tab:red", lw=1.8, ls="--", label="spin-down t2"),
        Line2D([0], [0], color="black", lw=1.5, ls=":", label="onsite U"),
    ]
    right_handles = [
        handle
        for handle, enabled in zip(
            right_handles,
            [t1 != 0.0, t2 != 0.0, t1 != 0.0, t2 != 0.0, U != 0.0],
        )
        if enabled
    ]
    if right_handles:
        ax_right.legend(handles=right_handles, loc="upper center", ncol=3, frameon=False)

    ax_right.text(
        min(pos[0] for pos in up_positions.values()) - 0.35,
        max(pos[1] for pos in up_positions.values()) + 0.12,
        "spin up",
        color="tab:blue",
        fontsize=9,
        fontweight="bold",
    )
    ax_right.text(
        min(pos[0] for pos in dn_positions.values()) - 0.35,
        min(pos[1] for pos in dn_positions.values()) - 0.34,
        "spin down",
        color="tab:red",
        fontsize=9,
        fontweight="bold",
    )

    ax_right.set_title("Spin-Layer View")
    ax_right.set_aspect("equal")
    ax_right.axis("off")

    margin_x = 1.0
    margin_y = 1.0
    ax_left.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax_left.set_ylim(min(ys) - margin_y, max(ys) + margin_y)
    right_xs = [pos[0] for pos in up_positions.values()] + [pos[0] for pos in dn_positions.values()]
    right_ys = [pos[1] for pos in up_positions.values()] + [pos[1] for pos in dn_positions.values()]
    ax_right.set_xlim(min(right_xs) - margin_x, max(right_xs) + margin_x)
    ax_right.set_ylim(min(right_ys) - margin_y, max(right_ys) + margin_y)

    fig.suptitle(
        "2D Fermi-Hubbard Model"
        f"\nLx={Lx}, Ly={Ly}, pbc_x={pbc_x}, pbc_y={pbc_y}"
        f"\nt1={t1}, t2={t2}, U={U}",
        fontsize=12,
    )
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes


def quick_plot(*, save: str | None = None, show: bool = True):
    """Fast default entry for the Hubbard model."""
    return plot_hubbard_layout(save=save, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the 2D Fermi-Hubbard model.")
    parser.add_argument("--Lx", type=int, default=4, help="Lattice size along x. Default: 4.")
    parser.add_argument("--Ly", type=int, default=4, help="Lattice size along y. Default: 4.")
    parser.add_argument("--pbc-x", action="store_true", help="Use periodic boundary conditions along x.")
    parser.add_argument("--pbc-y", action="store_true", help="Use periodic boundary conditions along y.")
    parser.add_argument("--t1", type=float, default=1.0, help="Nearest-neighbor hopping.")
    parser.add_argument("--t2", type=float, default=0.0, help="Diagonal next-nearest-neighbor hopping.")
    parser.add_argument("--U", type=float, default=4.0, help="Onsite Hubbard interaction.")
    parser.add_argument("--hide-labels", action="store_true", help="Hide site labels.")
    parser.add_argument("--save", type=str, default=None, help="Optional output image path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window.")
    args = parser.parse_args()

    plot_hubbard_layout(
        Lx=args.Lx,
        Ly=args.Ly,
        pbc_x=args.pbc_x,
        pbc_y=args.pbc_y,
        t1=args.t1,
        t2=args.t2,
        U=args.U,
        show_labels=not args.hide_labels,
        save=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
