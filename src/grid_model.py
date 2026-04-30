from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class Cell:
    row: int
    col: int
    zone: str = "residential"
    density: int = 1
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_sample_grid(rows: int = 10, cols: int = 10) -> List[List[Cell]]:
    grid: List[List[Cell]] = []
    for r in range(rows):
        row: List[Cell] = []
        for c in range(cols):
            # default
            zone = "residential"
            density = 1
            is_hub = False
            is_charging = False
            is_medical_pickup = False
            no_fly = False
            demand = 0

            # industrial district (top-right corner)
            if r in (0, 1, 2) and c in (7, 8, 9):
                zone = "industrial"
                density = 0

            # commercial corridor rows 4-5 (slightly higher demand)
            if r in (4, 5):
                zone = "commercial"
                density = 2
                demand = 2

            # hospital + medical pickup
            if (r, c) == (2, 2):
                zone = "hospital"
                is_medical_pickup = True
                density = 2
                demand = 3

            # hubs
            if (r, c) in ((0, 0), (9, 9)):
                is_hub = True
                zone = "hub"
                density = 2

            # charging pads near hubs
            if (r, c) in ((0, 1), (9, 8)):
                is_charging = True
                zone = "charging"

            # a deliberate no-fly cell for Phase 1 visualization
            if (r, c) == (5, 5):
                no_fly = True
                zone = "no_fly"

            cell = Cell(
                row=r,
                col=c,
                zone=zone,
                density=density,
                is_hub=is_hub,
                is_charging=is_charging,
                is_medical_pickup=is_medical_pickup,
                no_fly=no_fly,
                demand=demand,
            )
            row.append(cell)
        grid.append(row)
    return grid


def grid_to_serializable(grid: List[List[Cell]]) -> Dict[str, object]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    cells = []
    for row in grid:
        for cell in row:
            cells.append(cell.to_dict())
    return {"rows": rows, "cols": cols, "cells": cells}


# module-level sample grid
SAMPLE_GRID = create_sample_grid()


def get_sample_grid_serializable() -> Dict[str, object]:
    return grid_to_serializable(SAMPLE_GRID)
