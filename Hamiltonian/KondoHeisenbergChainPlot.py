"""
Plot the dedicated 1D Kondo-Heisenberg chain.

This file depends only on KondoHeisenbergChain.py.
It is intentionally separate from the multileg plotter.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _load_chain_module():
    here = Path(__file__).resolve()
    for candidate in ("KondoHeisenbergChain.py", "KondoHeisenbergChain_staging.py"):
        path = here.with_name(candidate)
        if path.exists() and path != here:
            spec = importlib.util.spec_from_file_location("kondo_heisenberg_chain_linked", path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        "Could not locate KondoHeisenbergChain.py or KondoHeisenbergChain_staging.py "
        f"next to {here}."
    )


_CHAIN = _load_chain_module()
kondo_heisenberg_chain_geometry = _CHAIN.kondo_heisenberg_chain_geometry


def _draw_bonds(
    ax: Any,
    positions: dict[int, tuple[float, float]],
    bonds: list[tuple[int, int]],
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    wrap_curvature: float = 0.25,
) -> None:
    for i, j in bonds:
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        is_wrap = abs(x1 - x2) > 2.0

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


def plot_kondo_heisenberg_chain(
    *,
    Lx: int = 8,
    pbc: bool = False,
    t: float = 1.0,
    J_K: float = 1.0,
    J1: float = 1.0,
    J2: float = 0.0,
    show_labels: bool = True,
    figsize: tuple[float, float] = (12.0, 4.8),
    save: str | None = None,
    show: bool = True,
):
    """
    Plot the 1D Kondo-Heisenberg chain.

    Upper line: conduction chain.
    Lower line: local-spin chain.
    Vertical dotted bonds: onsite Kondo exchange.
    """
    geometry = kondo_heisenberg_chain_geometry(Lx=Lx, pbc=pbc)
    nn = geometry["nn"]
    nnn = geometry["nnn"]

    x_spacing = 1.8
    y_c = 0.95
    y_s = -0.95
    c_positions = {site: (site * x_spacing, y_c) for site in range(Lx)}
    s_positions = {site: (site * x_spacing, y_s) for site in range(Lx)}

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if t != 0.0:
        _draw_bonds(ax, c_positions, nn, color="tab:blue", linewidth=2.5)
    if J1 != 0.0:
        _draw_bonds(ax, s_positions, nn, color="tab:red", linewidth=2.5)
    if J2 != 0.0:
        _draw_bonds(ax, s_positions, nnn, color="tab:orange", linestyle="--", linewidth=2.0)
    if J_K != 0.0:
        for site in range(Lx):
            x = c_positions[site][0]
            ax.plot([x, x], [y_s, y_c], color="black", linestyle=":", linewidth=1.5, alpha=0.9)

    ax.scatter(
        [c_positions[i][0] for i in range(Lx)],
        [c_positions[i][1] for i in range(Lx)],
        s=95,
        c="white",
        edgecolors="tab:blue",
        linewidths=1.8,
        zorder=3,
    )
    ax.scatter(
        [s_positions[i][0] for i in range(Lx)],
        [s_positions[i][1] for i in range(Lx)],
        s=95,
        c="white",
        edgecolors="tab:red",
        linewidths=1.8,
        zorder=3,
    )

    if show_labels:
        for site in range(Lx):
            x = c_positions[site][0]
            ax.text(x, y_c + 0.22, f"{site}", ha="center", va="bottom", fontsize=9)

    left_x = -0.8 * x_spacing
    ax.text(left_x, y_c, "conduction\nchain", ha="right", va="center", fontsize=10, color="tab:blue")
    ax.text(left_x, y_s, "local-spin\nchain", ha="right", va="center", fontsize=10, color="tab:red")

    ax.set_title("1D Kondo-Heisenberg Chain")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(left_x - 0.3, (Lx - 1) * x_spacing + 0.8)
    ax.set_ylim(y_s - 0.8, y_c + 0.8)

    handles = []
    if t != 0.0:
        handles.append(Line2D([0], [0], color="tab:blue", lw=2.5, label="electron hopping t"))
    if J1 != 0.0:
        handles.append(Line2D([0], [0], color="tab:red", lw=2.5, label="local-spin J1"))
    if J2 != 0.0:
        handles.append(Line2D([0], [0], color="tab:orange", lw=2.0, ls="--", label="local-spin J2"))
    if J_K != 0.0:
        handles.append(Line2D([0], [0], color="black", lw=1.5, ls=":", label="onsite Kondo J_K"))
    if handles:
        ax.legend(handles=handles, loc="upper center", ncol=4, frameon=False)

    fig.suptitle(
        f"Kondo-Heisenberg Chain\nLx={Lx}, pbc={pbc}, t={t}, J_K={J_K}, J1={J1}, J2={J2}",
        fontsize=12,
    )
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def quick_plot(
    *,
    save: str | None = None,
    show: bool = True,
):
    """Fast default entry for the chain."""
    return plot_kondo_heisenberg_chain(save=save, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the dedicated 1D Kondo-Heisenberg chain.")
    parser.add_argument("--Lx", type=int, default=8, help="Chain length. Default: 8.")
    parser.add_argument("--pbc", action="store_true", help="Use periodic boundary conditions along the chain.")
    parser.add_argument("--t", type=float, default=1.0, help="Electron hopping along the chain.")
    parser.add_argument("--J-K", dest="J_K", type=float, default=1.0, help="Onsite Kondo exchange.")
    parser.add_argument("--J1", type=float, default=1.0, help="Local-spin nearest-neighbor exchange.")
    parser.add_argument("--J2", type=float, default=0.0, help="Local-spin next-nearest-neighbor exchange.")
    parser.add_argument("--hide-labels", action="store_true", help="Hide site indices.")
    parser.add_argument("--save", type=str, default=None, help="Optional output image path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window.")
    args = parser.parse_args()

    plot_kondo_heisenberg_chain(
        Lx=args.Lx,
        pbc=args.pbc,
        t=args.t,
        J_K=args.J_K,
        J1=args.J1,
        J2=args.J2,
        show_labels=not args.hide_labels,
        save=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
