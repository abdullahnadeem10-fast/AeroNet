import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from src.grid_model import Grid, Zone

ZONE_COLORS = {
    Zone.RESIDENTIAL: "#DEB887",
    Zone.COMMERCIAL: "#5B8DEE",
    Zone.HOSPITAL: "#E85D75",
    Zone.SCHOOL: "#F2C94C",
    Zone.INDUSTRIAL: "#5C6670",
    Zone.OPEN_FIELD: "#59A14F",
}

ROUTE_COLORS = ["#e63946", "#1d3557", "#2a9d8f"]  # red, navy, teal - high contrast
ROUTE_LINESTYLES = ["-", "--", "-."]
ROUTE_LINEWIDTHS = [4.0, 3.2, 2.6]


def _zone_legend_patches():
    return [mpatches.Patch(color=ZONE_COLORS[z], label=z.value) for z in Zone]


def _draw_zone_imshow(ax, grid: Grid, title: str, *, show_row_col_labels: bool = False):
    """Colored zone grid only (no legend)."""
    rgb = [[to_rgb(ZONE_COLORS[cell.zone]) for cell in row] for row in grid.grid]
    ax.imshow(rgb, origin="upper", aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([i - 0.5 for i in range(grid.cols + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(grid.rows + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    if show_row_col_labels:
        ax.set_xticks(range(grid.cols))
        ax.set_yticks(range(grid.rows))
        ax.tick_params(which="major", labelsize=8)
        ax.set_xlabel("col")
        ax.set_ylabel("row")
    else:
        ax.tick_params(
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )


def _draw_zone_layer(ax, grid: Grid, title: str):
    _draw_zone_imshow(ax, grid, title, show_row_col_labels=False)
    zone_leg = ax.legend(
        handles=_zone_legend_patches(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    ax.add_artist(zone_leg)


def _hatch_no_fly(ax, grid: Grid):
    """Mark no-fly cells with a diagonal hatch + border."""
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.grid[r][c].no_fly:
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        fill=False,
                        linewidth=1.8,
                        edgecolor="#ffffff",
                        hatch="////",
                        zorder=5,
                    )
                )


def _show_figure(fig, show: bool):
    if not show:
        return
    try:
        __IPYTHON__  # noqa: F821
        plt.show()
    except NameError:
        if matplotlib.is_interactive():
            plt.show()
        else:
            plt.close(fig)


def plot_zone_layout(
    grid: Grid,
    figsize=None,
    ax=None,
    title="Zone layout",
    show: bool = True,
    mark_no_fly: bool = False,
):
    created_fig = ax is None
    if created_fig:
        if figsize is None:
            figsize = (
                max(6.0, grid.cols * 0.55),
                max(5.0, grid.rows * 0.55),
            )
        _, ax = plt.subplots(figsize=figsize)

    _draw_zone_layer(ax, grid, title)
    if mark_no_fly:
        _hatch_no_fly(ax, grid)

    if created_fig:
        fig = ax.figure
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        _show_figure(fig, show)

    return ax


def plot_delivery_routes(
    grid: Grid,
    segment_paths: list[tuple[str, list[tuple[int, int]]]],
    *,
    hub: tuple[int, int] | None = None,
    pickup: tuple[int, int] | None = None,
    dropoff: tuple[int, int] | None = None,
    figsize=None,
    title: str = "A* delivery visualization",
    show: bool = True,
):
    """
    Two panels: (a) zones + no-fly hatch + zone legend, (b) same + A* polylines.
    Overlapping segments get small perpendicular offsets so all three colors show.
    """
    n_seg = max(1, len(segment_paths))
    if figsize is None:
        figsize = (max(11.0, grid.cols * 1.05 * 2), max(5.2, grid.rows * 0.72))

    fig, (ax_map, ax_route) = plt.subplots(1, 2, figsize=figsize)

    _draw_zone_imshow(
        ax_map,
        grid,
        "(a) Zone map  (//// = no-fly)",
        show_row_col_labels=True,
    )
    _hatch_no_fly(ax_map, grid)

    _draw_zone_imshow(
        ax_route,
        grid,
        "(b) A* paths  (offsets only when segments share cells)",
        show_row_col_labels=True,
    )
    _hatch_no_fly(ax_route, grid)

    route_handles: list = []
    for i, (label, path) in enumerate(segment_paths):
        if not path or len(path) < 1:
            continue
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        # Separate stacked segments in screen Y so colors don't hide each other
        dy = (i - (n_seg - 1) / 2.0) * 0.11
        ys_adj = [y + dy for y in ys]
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        ls = ROUTE_LINESTYLES[i % len(ROUTE_LINESTYLES)]
        lw = ROUTE_LINEWIDTHS[i % len(ROUTE_LINEWIDTHS)]
        (line,) = ax_route.plot(
            xs,
            ys_adj,
            color=color,
            linestyle=ls,
            linewidth=lw,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
            zorder=8,
            alpha=0.95,
        )
        route_handles.append(line)

    stops_plotted = []
    if hub is not None:
        ax_route.plot(
            hub[1],
            hub[0],
            "s",
            markersize=13,
            markerfacecolor="#ffd166",
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=15,
            label="hub",
        )
        stops_plotted.append("hub")
    if pickup is not None:
        ax_route.plot(
            pickup[1],
            pickup[0],
            "^",
            markersize=12,
            markerfacecolor="#9b5de5",
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=15,
            label="pickup",
        )
        stops_plotted.append("pickup")
    if dropoff is not None:
        ax_route.plot(
            dropoff[1],
            dropoff[0],
            "D",
            markersize=11,
            markerfacecolor="#00bbf9",
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=15,
            label="drop-off",
        )
        stops_plotted.append("drop-off")

    if route_handles or stops_plotted:
        ax_route.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
            framealpha=0.95,
            title="Route",
        )

    zone_handles = _zone_legend_patches()
    fig.legend(
        handles=zone_handles,
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=True,
        title="Cell = zone type",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16, top=0.9, right=0.78, wspace=0.35)

    _show_figure(fig, show)
    return ax_route


def plot_demand_heatmap(
    grid: Grid,
    figsize=None,
    title: str = "Delivery demand heatmap",
    show: bool = True,
):
    """Heatmap of cell demand values across the grid."""
    data = [[cell.demand for cell in row] for row in grid.grid]
    if figsize is None:
        figsize = (max(5.0, grid.cols * 0.6), max(4.5, grid.rows * 0.55))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="YlOrRd", origin="upper", aspect="equal", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Demand", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(grid.cols))
    ax.set_yticks(range(grid.rows))
    ax.tick_params(which="major", labelsize=8)
    ax.set_xlabel("col", fontsize=9)
    ax.set_ylabel("row", fontsize=9)
    for r in range(grid.rows):
        for c in range(grid.cols):
            val = data[r][c]
            color = "white" if val > 35 else "black"
            ax.text(c, r, str(val), ha="center", va="center", fontsize=7, color=color)
    fig.tight_layout()
    _show_figure(fig, show)
    return ax


def plot_disruption_reroute(
    grid: Grid,
    original_segments: list[tuple[str, list[tuple[int, int]]]],
    rerouted_segments: list[tuple[str, list[tuple[int, int]]]],
    *,
    hub: tuple[int, int] | None = None,
    pickup: tuple[int, int] | None = None,
    dropoff: tuple[int, int] | None = None,
    disrupted_cell: tuple[int, int] | None = None,
    title: str = "Disruption & re-routing",
    show: bool = True,
):
    """
    Side-by-side comparison: (a) original remaining route, (b) rerouted path.
    The disrupted cell is marked with a red X on both panels.
    """
    figsize = (max(12.0, grid.cols * 1.1 * 2), max(5.5, grid.rows * 0.75))
    fig, (ax_orig, ax_new) = plt.subplots(1, 2, figsize=figsize)

    panel_data = [
        (ax_orig, original_segments, "(a) Original remaining route"),
        (ax_new,  rerouted_segments, "(b) Rerouted path"),
    ]
    for ax, segs, panel_title in panel_data:
        _draw_zone_imshow(ax, grid, panel_title, show_row_col_labels=True)
        _hatch_no_fly(ax, grid)

        n = max(1, len(segs))
        for i, (label, path) in enumerate(segs):
            if not path:
                continue
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            dy = (i - (n - 1) / 2.0) * 0.11
            color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
            ax.plot(
                xs,
                [y + dy for y in ys],
                color=color,
                linestyle=ROUTE_LINESTYLES[i % len(ROUTE_LINESTYLES)],
                linewidth=ROUTE_LINEWIDTHS[i % len(ROUTE_LINEWIDTHS)],
                marker="o",
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=label,
                zorder=8,
                alpha=0.9,
            )

        if disrupted_cell is not None:
            dr, dc = disrupted_cell
            ax.plot(
                dc, dr, "rx",
                markersize=16, markeredgewidth=3,
                zorder=20, label="disruption",
            )

        for coord, marker, mcolor, lbl in [
            (hub,     "s", "#ffd166", "hub"),
            (pickup,  "^", "#9b5de5", "pickup"),
            (dropoff, "D", "#00bbf9", "drop-off"),
        ]:
            if coord is not None:
                ax.plot(
                    coord[1], coord[0], marker,
                    markersize=11,
                    markerfacecolor=mcolor,
                    markeredgecolor="black",
                    markeredgewidth=1.4,
                    zorder=15,
                    label=lbl,
                )

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            framealpha=0.9,
            title="Legend",
        )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, top=0.9, right=0.82, wspace=0.4)
    _show_figure(fig, show)
    return ax_new


__all__ = [
    "plot_zone_layout",
    "plot_delivery_routes",
    "plot_demand_heatmap",
    "plot_disruption_reroute",
]
