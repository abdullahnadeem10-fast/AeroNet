import random
import re

from src.grid_model import Grid, Zone

_R1_INDUSTRY = re.compile(r"Industrial cell \((\d+), (\d+)\) adjacent to")
_R2_RESIDENTIAL = re.compile(r"Residential cell \((\d+), (\d+)\)")
_R3_HUB = re.compile(r"Hub at \((\d+), (\d+)\)")


def check_industrial_safety(grid: Grid) -> list[str]:
    errors = []
    for row in grid.grid:
        for cell in row:
            if cell.zone == Zone.INDUSTRIAL:
                for n in grid.get_neighbors(cell.row, cell.col):
                    if n.zone in (Zone.SCHOOL, Zone.HOSPITAL):
                        errors.append(
                            f"R1 FAILED: Industrial cell "
                            f"({cell.row}, {cell.col}) adjacent to "
                            f"{n.zone.value} at ({n.row}, {n.col})"
                        )
    return errors


def check_residential_coverage(grid: Grid) -> list[str]:
    errors = []
    hubs = [cell for row in grid.grid for cell in row if cell.is_hub]
    for row in grid.grid:
        for cell in row:
            if cell.zone == Zone.RESIDENTIAL:
                if not any(grid.manhattan(cell, hub) <= 3 for hub in hubs):
                    errors.append(
                        f"R2 FAILED: Residential cell "
                        f"({cell.row}, {cell.col}) "
                        f"is farther than 3 cells from every hub"
                    )
    return errors


def check_hub_charging(grid: Grid) -> list[str]:
    errors = []
    charging_cells = [cell for row in grid.grid for cell in row if cell.is_charging]
    for row in grid.grid:
        for cell in row:
            if cell.is_hub:
                if not any(grid.manhattan(cell, ch) <= 2 for ch in charging_cells):
                    errors.append(
                        f"R3 FAILED: Hub at "
                        f"({cell.row}, {cell.col}) "
                        f"has no charging pad within 2 cells"
                    )
    return errors


def check_medical_access(grid: Grid) -> list[str]:
    hospitals = [cell for row in grid.grid for cell in row if cell.zone == Zone.HOSPITAL]
    pickups = [cell for row in grid.grid for cell in row if cell.is_medical_pickup]
    for hospital in hospitals:
        for pickup in pickups:
            if grid.manhattan(hospital, pickup) <= 1:
                return []
    return [
        "R4 FAILED: No hospital has a medical pickup point within 1 cell"
    ]


def validate_layout(grid: Grid) -> list[str]:
    errors = []
    errors.extend(check_industrial_safety(grid))
    errors.extend(check_residential_coverage(grid))
    errors.extend(check_hub_charging(grid))
    errors.extend(check_medical_access(grid))
    return errors


def fix_layout_from_errors(grid: Grid, errors: list[str]) -> bool:
    changed = False
    for err in errors:
        if err.startswith("R1 FAILED"):
            m = _R1_INDUSTRY.search(err)
            if not m:
                continue
            r, c = int(m.group(1)), int(m.group(2))
            cell = grid.grid[r][c]
            if cell.zone == Zone.INDUSTRIAL:
                cell.zone = random.choice([Zone.OPEN_FIELD, Zone.COMMERCIAL])
                changed = True

        elif err.startswith("R2 FAILED"):
            m = _R2_RESIDENTIAL.search(err)
            if not m:
                continue
            r, c = int(m.group(1)), int(m.group(2))
            placed = False
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    if abs(dr) + abs(dc) > 3:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                        grid.grid[nr][nc].is_hub = True
                        placed = True
                        break
                if placed:
                    break
            if placed:
                changed = True

        elif err.startswith("R3 FAILED"):
            m = _R3_HUB.search(err)
            if not m:
                continue
            r, c = int(m.group(1)), int(m.group(2))
            grid.grid[r][c].is_charging = True
            changed = True

        elif err.startswith("R4 FAILED"):
            fixed_r4 = False
            for row in grid.grid:
                for cell in row:
                    if cell.zone == Zone.HOSPITAL:
                        cell.is_medical_pickup = True
                        changed = True
                        fixed_r4 = True
                        break
                if fixed_r4:
                    break
            if not fixed_r4:
                c0 = grid.grid[0][0]
                c0.zone = Zone.HOSPITAL
                c0.is_medical_pickup = True
                changed = True

    return changed


def repair_layout(grid: Grid, max_iterations: int = 200) -> bool:
    for _ in range(max_iterations):
        errors = validate_layout(grid)
        if not errors:
            return True
        if not fix_layout_from_errors(grid, errors):
            break
    return len(validate_layout(grid)) == 0


def print_validation_report(grid: Grid) -> None:
    errors = validate_layout(grid)

    by_rule = {
        "R1": [e for e in errors if e.startswith("R1")],
        "R2": [e for e in errors if e.startswith("R2")],
        "R3": [e for e in errors if e.startswith("R3")],
        "R4": [e for e in errors if e.startswith("R4")],
    }
    rule_meta = [
        ("R1", "Industrial not adjacent to School/Hospital"),
        ("R2", "Residential within 3 cells of a hub"),
        ("R3", "Hub within 2 cells of a charging pad"),
        ("R4", "Hospital has medical pickup within 1 cell"),
    ]

    print("\n==============================")
    print(" AERONET LAYOUT VALIDATION ")
    print("==============================\n")
    for rule_id, desc in rule_meta:
        rule_errs = by_rule[rule_id]
        if rule_errs:
            print(f"  {rule_id} FAILED - {desc}")
            for e in rule_errs:
                print(f"    {e}")
        else:
            print(f"  {rule_id} PASSED - {desc}")
    print()
    if not errors:
        print("Layout validity = TRUE - all CSP constraints satisfied.")
    else:
        print(f"Layout validity = FALSE  ({len(errors)} violation(s) total)")
