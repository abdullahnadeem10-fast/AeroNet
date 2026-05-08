import random
from enum import Enum


class Zone(Enum):
    RESIDENTIAL = "Residential"
    COMMERCIAL = "Commercial"
    HOSPITAL = "Hospital"
    SCHOOL = "School"
    INDUSTRIAL = "Industrial"
    OPEN_FIELD = "Open Field"


class Cell:
    def __init__(
        self,
        row,
        col,
        zone,
        density=0,
        is_hub=False,
        is_charging=False,
        is_medical_pickup=False,
        demand=0,
        no_fly=False,
    ):
        self.row = row
        self.col = col
        self.zone = zone
        self.density = density
        self.is_hub = is_hub
        self.is_charging = is_charging
        self.is_medical_pickup = is_medical_pickup
        self.demand = demand
        self.no_fly = no_fly

    def __str__(self):
        return f"({self.row}, {self.col}) {self.zone.value}"


class Grid:
    def __init__(self, rows, cols, zone_limits=None):
        self.rows = rows
        self.cols = cols
        self.grid = []
        self.zone_limits = zone_limits

    def populate_grid(self):
        self.grid = []
        total = self.rows * self.cols

        def make_cell(row, col, zone):
            return Cell(
                row=row,
                col=col,
                zone=zone,
                density=random.randint(0, 100),
                is_hub=random.choice([True, False]),
                is_charging=random.choice([True, False]),
                is_medical_pickup=random.choice([True, False]),
                demand=random.randint(0, 50),
                no_fly=False,
            )

        if self.zone_limits is None:
            for row in range(self.rows):
                current_row = []
                for col in range(self.cols):
                    zone = random.choice(list(Zone))
                    current_row.append(make_cell(row, col, zone))
                self.grid.append(current_row)
            return

        caps = {}
        for z, n in self.zone_limits.items():
            if z == Zone.OPEN_FIELD:
                continue
            if n < 0:
                raise ValueError(
                    f"Zone limit for {z} must be non-negative, got {n}"
                )
            caps[z] = int(n)

        pool = []
        for z, cap in caps.items():
            pool.extend([z] * cap)

        if len(pool) > total:
            raise ValueError(
                f"Sum of zone limits ({len(pool)}) exceeds grid size ({total}). "
                "Lower the limits or enlarge the grid."
            )

        pool.extend([Zone.OPEN_FIELD] * (total - len(pool)))
        random.shuffle(pool)

        i = 0
        for row in range(self.rows):
            current_row = []
            for col in range(self.cols):
                current_row.append(make_cell(row, col, pool[i]))
                i += 1
            self.grid.append(current_row)

    def get_neighbors(self, row, col):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr = row + dr
            nc = col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append(self.grid[nr][nc])
        return neighbors

    def manhattan(self, a, b):
        return abs(a.row - b.row) + abs(a.col - b.col)
