from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.grid_model import Grid, Zone
from src.layout_validator import print_validation_report, repair_layout
from src.visualization import plot_zone_layout


def run_demo() -> Grid:
    grid = Grid(
        10,
        10,
        zone_limits={
            Zone.RESIDENTIAL: 10,
            Zone.COMMERCIAL: 10,
            Zone.HOSPITAL: 3,
            Zone.SCHOOL: 5,
            Zone.INDUSTRIAL: 8,
        },
    )
    grid.populate_grid()
    print_validation_report(grid)
    repair_layout(grid)
    print_validation_report(grid)
    plot_zone_layout(grid)
    return grid


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
