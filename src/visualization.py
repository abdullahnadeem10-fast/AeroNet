import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

from src.grid_model import Grid, Zone

ZONE_COLORS = {
    Zone.RESIDENTIAL: "#DEB887",
    Zone.COMMERCIAL: "#5B8DEE",
    Zone.HOSPITAL: "#E85D75",
    Zone.SCHOOL: "#F2C94C",
    Zone.INDUSTRIAL: "#5C6670",
    Zone.OPEN_FIELD: "#59A14F",
}


def plot_zone_layout(grid: Grid, figsize=None, ax=None, title="Zone layout"):
    rgb = [[to_rgb(ZONE_COLORS[cell.zone]) for cell in row] for row in grid.grid]
    created_fig = ax is None
    if created_fig:
        if figsize is None:
            figsize = (
                max(6.0, grid.cols * 0.55),
                max(5.0, grid.rows * 0.55),
            )
        _, ax = plt.subplots(figsize=figsize)

    ax.imshow(rgb, origin="upper", aspect="equal", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([i - 0.5 for i in range(grid.cols + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(grid.rows + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    handles = [
        mpatches.Patch(color=ZONE_COLORS[z], label=z.value) for z in Zone
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        fontsize=9,
        frameon=False,
    )

    if created_fig:
        fig = ax.figure
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        try:
            __IPYTHON__  # noqa: F821
            plt.show()
        except NameError:
            if matplotlib.is_interactive():
                plt.show()
            else:
                plt.close(fig)

    return ax
