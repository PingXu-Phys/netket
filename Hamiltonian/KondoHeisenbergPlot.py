"""
Plot the generalized 2D Kondo-Heisenberg ladder.

This file is linked to ``KondoHeisenberg.py`` and uses the same shell-based
geometry:
- shell 1: nearest neighbors
- shell 2: plaquette diagonals
- shell 3: axial third neighbors
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


Bond = tuple[int, int]


def _load_kondo_heisenberg_module():
    here = Path(__file__).resolve()
    for candidate in ("KondoHeisenberg.py", "KondoHeisenberg_staging.py"):
        path = here.with_name(candidate)
        if path.exists() and path != here:
            spec = importlib.util.spec_from_file_location("kondo_heisenberg_linked", path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        "Could not locate KondoHeisenberg.py or KondoHeisenberg_staging.py "
        f"next to {here}."
    )


_KH = _load_kondo_heisenberg_module()
ladder_index = _KH.ladder_index
kondo_heisenberg_geometry = _KH.kondo_heisenberg_geometry


def _spatial_positions(
    Lx: int,
    n_legs: int,
    *,
    x_spacing: float = 1.8,
    y_spacing: float = 1.5,
) -> dict[int, tuple[float, float]]:
    return {
        ladder_index(x, leg, n_legs): (x * x_spacing, -leg * y_spacing)
        for x in range(Lx)
        for leg in range(n_legs)
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


def plot_kondo_heisenberg_layout(
    *,
    Lx: int = 4,
    n_legs: int = 2,
    pbc_x: bool = False,
    pbc_y: bool = False,
    t1: float = 1.0,
    t2: float = 0.0,
    t3: float = 0.0,
    J_K: float = 1.0,
    J1: float = 1.0,
    J2: float = 0.0,
    J3: float = 0.0,
    show_labels: bool = True,
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
    show: bool = True,
):
    """
    Plot the generalized 2D Kondo-Heisenberg ladder.

    Shell convention:
    - ``t1`` / ``J1``: nearest neighbors
    - ``t2`` / ``J2``: plaquette diagonals
    - ``t3`` / ``J3``: axial third neighbors
    """
    geometry = kondo_heisenberg_geometry(Lx=Lx, n_legs=n_legs, pbc_x=pbc_x, pbc_y=pbc_y)
    positions = _spatial_positions(Lx, n_legs)

    if figsize is None:
        figsize = (max(12.0, 2.2 * Lx + 3.0), max(5.0, 1.6 * n_legs + 2.5))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    ax_left, ax_right = axes

    shell_1 = geometry["shell_1"]
    shell_2 = geometry["shell_2"]
    shell_3 = geometry["shell_3"]

    left_specs = [
        ("shell 1", shell_1, "black", "-", 2.2, t1 != 0.0 or J1 != 0.0),
        ("shell 2", shell_2, "tab:green", "--", 2.0, t2 != 0.0 or J2 != 0.0),
        ("shell 3", shell_3, "tab:orange", "-.", 1.9, t3 != 0.0 or J3 != 0.0),
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
            for leg in range(n_legs):
                site = ladder_index(x, leg, n_legs)
                px, py = positions[site]
                ax_left.text(
                    px,
                    py + 0.18,
                    f"({x},{leg})",
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
        ax_left.legend(handles=left_handles, loc="upper center", ncol=3, frameon=False)

    ax_left.set_title("Spatial Geometry")
    ax_left.set_aspect("equal")
    ax_left.axis("off")

    # Pseudo-3D projection: the conduction layer is lifted and shifted so the
    # onsite J_K bonds are slanted instead of being hidden under the sites.
    s_positions = _shift_positions(positions, dx=-0.10, dy=-0.18)
    c_positions = _shift_positions(positions, dx=0.38, dy=0.34)

    right_specs = [
        ("J1", s_positions, shell_1, "tab:red", "-", 2.1, J1),
        ("J2", s_positions, shell_2, "tab:red", "--", 1.8, J2),
        ("J3", s_positions, shell_3, "tab:red", "-.", 1.8, J3),
        ("t1", c_positions, shell_1, "tab:blue", "-", 2.1, t1),
        ("t2", c_positions, shell_2, "tab:blue", "--", 1.8, t2),
        ("t3", c_positions, shell_3, "tab:blue", "-.", 1.8, t3),
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

    if J_K != 0.0:
        for site in sorted(positions):
            x1, y1 = s_positions[site]
            x2, y2 = c_positions[site]
            ax_right.plot(
                [x1, x2],
                [y1, y2],
                color="black",
                linestyle=":",
                linewidth=1.6,
                alpha=0.85,
                zorder=2,
            )

    ax_right.scatter(
        [c_positions[i][0] for i in sorted(c_positions)],
        [c_positions[i][1] for i in sorted(c_positions)],
        s=82,
        c="white",
        edgecolors="tab:blue",
        linewidths=1.8,
        zorder=3,
    )
    ax_right.scatter(
        [s_positions[i][0] for i in sorted(s_positions)],
        [s_positions[i][1] for i in sorted(s_positions)],
        s=82,
        c="white",
        edgecolors="tab:red",
        linewidths=1.8,
        zorder=3,
    )

    if show_labels:
        for site, (x, y) in positions.items():
            ax_right.text(
                x + 0.14,
                y + 0.06,
                f"{site}",
                ha="center",
                va="center",
                fontsize=7,
                color="dimgray",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9},
            )

    right_handles = []
    if t1 != 0.0:
        right_handles.append(Line2D([0], [0], color="tab:blue", lw=2.1, ls="-", label="electron t1"))
    if t2 != 0.0:
        right_handles.append(Line2D([0], [0], color="tab:blue", lw=1.8, ls="--", label="electron t2"))
    if t3 != 0.0:
        right_handles.append(Line2D([0], [0], color="tab:blue", lw=1.8, ls="-.", label="electron t3"))
    if J1 != 0.0:
        right_handles.append(Line2D([0], [0], color="tab:red", lw=2.1, ls="-", label="spin J1"))
    if J2 != 0.0:
        right_handles.append(Line2D([0], [0], color="tab:red", lw=1.8, ls="--", label="spin J2"))
    if J3 != 0.0:
        right_handles.append(Line2D([0], [0], color="tab:red", lw=1.8, ls="-.", label="spin J3"))
    if J_K != 0.0:
        right_handles.append(Line2D([0], [0], color="black", lw=1.4, ls=":", label="onsite J_K"))
    if right_handles:
        ax_right.legend(handles=right_handles, loc="upper center", ncol=2, frameon=False)

    ax_right.text(
        min(pos[0] for pos in c_positions.values()) - 0.4,
        max(pos[1] for pos in c_positions.values()) + 0.15,
        "c layer",
        color="tab:blue",
        fontsize=9,
        fontweight="bold",
    )
    ax_right.text(
        min(pos[0] for pos in s_positions.values()) - 0.4,
        min(pos[1] for pos in s_positions.values()) - 0.35,
        "S layer",
        color="tab:red",
        fontsize=9,
        fontweight="bold",
    )

    ax_right.set_title("Layered Sector View")
    ax_right.set_aspect("equal")
    ax_right.axis("off")

    margin_x = 1.0
    margin_y = 1.0
    ax_left.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax_left.set_ylim(min(ys) - margin_y, max(ys) + margin_y)
    right_xs = [pos[0] for pos in c_positions.values()] + [pos[0] for pos in s_positions.values()]
    right_ys = [pos[1] for pos in c_positions.values()] + [pos[1] for pos in s_positions.values()]
    ax_right.set_xlim(min(right_xs) - margin_x, max(right_xs) + margin_x)
    ax_right.set_ylim(min(right_ys) - margin_y, max(right_ys) + margin_y)

    fig.suptitle(
        "Kondo-Heisenberg Ladder"
        f"\nLx={Lx}, n_legs={n_legs}, pbc_x={pbc_x}, pbc_y={pbc_y}"
        f"\nt=({t1}, {t2}, {t3}), J_K={J_K}, J=({J1}, {J2}, {J3})",
        fontsize=12,
    )
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes


def quick_plot(*, save: str | None = None, show: bool = True):
    """Fast default entry: direct run produces a 4x2 open ladder."""
    return plot_kondo_heisenberg_layout(save=save, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the generalized 2D Kondo-Heisenberg ladder.")
    parser.add_argument("--Lx", type=int, default=4, help="Length along x. Default: 4.")
    parser.add_argument("--n-legs", type=int, default=2, help="Number of spatial ladder legs. Default: 2.")
    parser.add_argument("--pbc-x", action="store_true", help="Use periodic boundary conditions along x.")
    parser.add_argument("--pbc-y", action="store_true", help="Use periodic boundary conditions along the leg direction.")
    parser.add_argument("--t1", type=float, default=1.0, help="Electron hopping on shell 1.")
    parser.add_argument("--t2", type=float, default=0.0, help="Electron hopping on shell 2.")
    parser.add_argument("--t3", type=float, default=0.0, help="Electron hopping on shell 3.")
    parser.add_argument("--J-K", dest="J_K", type=float, default=1.0, help="Onsite Kondo exchange.")
    parser.add_argument("--J1", type=float, default=1.0, help="Local-spin exchange on shell 1.")
    parser.add_argument("--J2", type=float, default=1.0, help="Local-spin exchange on shell 2.")
    parser.add_argument("--J3", type=float, default=0.0, help="Local-spin exchange on shell 3.")
    parser.add_argument("--hide-labels", action="store_true", help="Hide site labels.")
    parser.add_argument("--save", type=str, default=None, help="Optional output image path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window.")
    args = parser.parse_args()

    plot_kondo_heisenberg_layout(
        Lx=args.Lx,
        n_legs=args.n_legs,
        pbc_x=args.pbc_x,
        pbc_y=args.pbc_y,
        t1=args.t1,
        t2=args.t2,
        t3=args.t3,
        J_K=args.J_K,
        J1=args.J1,
        J2=args.J2,
        J3=args.J3,
        show_labels=not args.hide_labels,
        save=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
