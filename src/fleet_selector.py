"""
Fleet selection under a fixed budget (genetic algorithm).
Drone specs: Light $1000, 2 kg, 12 cells | Heavy $1800, 5 kg, 20 cells.
Fitness: score = 0.75 * coverage% - 0.25 * budget_used%
"""

from __future__ import annotations

from pathlib import Path
import random
import sys
from dataclasses import dataclass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.grid_model import Grid

LIGHT_COST = 1000
HEAVY_COST = 1800
LIGHT_PAYLOAD_KG = 2
HEAVY_PAYLOAD_KG = 5
LIGHT_RANGE_CELLS = 12
HEAVY_RANGE_CELLS = 20


def total_grid_demand(grid: Grid) -> int:
    return sum(cell.demand for row in grid.grid for cell in row)


def fleet_cost(light: int, heavy: int) -> int:
    return light * LIGHT_COST + heavy * HEAVY_COST


def fleet_payload_kg(light: int, heavy: int) -> int:
    return light * LIGHT_PAYLOAD_KG + heavy * HEAVY_PAYLOAD_KG


def max_affordable_payload_kg(budget: int) -> int:
    """Best total kg (2L + 5H) you can buy under this budget."""
    best = 0
    for light in range(budget // LIGHT_COST + 1):
        rem = budget - light * LIGHT_COST
        for heavy in range(rem // HEAVY_COST + 1):
            best = max(best, fleet_payload_kg(light, heavy))
    return best


def coverage_pct(light: int, heavy: int, grid: Grid, budget: int) -> float:
    """
    What share of the *reachable* demand this fleet covers: we compare payload
    to min(city demand, best payload money can buy), capped at 100%.
    """
    demand = total_grid_demand(grid)
    affordable = max_affordable_payload_kg(budget)
    if demand <= 0:
        return 100.0
    target = min(demand, affordable)
    if target <= 0:
        return 100.0
    payload = fleet_payload_kg(light, heavy)
    return min(100.0, 100.0 * payload / target)


def clip_to_budget(light: int, heavy: int, budget: int) -> list[int]:
    """Keep counts non-negative and cost <= budget."""
    light = max(0, light)
    heavy = max(0, heavy)
    while fleet_cost(light, heavy) > budget:
        if heavy > 0:
            heavy -= 1
        elif light > 0:
            light -= 1
        else:
            break
    return [light, heavy]


def random_chromosome(budget: int) -> list[int]:
    max_light = budget // LIGHT_COST
    light = random.randint(0, max_light)
    remaining = budget - light * LIGHT_COST
    heavy = random.randint(0, remaining // HEAVY_COST)
    return [light, heavy]


def fitness(chrom: list[int], grid: Grid, budget: int) -> float:
    light, heavy = chrom
    if fleet_cost(light, heavy) > budget:
        return -1e9
    if light == 0 and heavy == 0:
        return -1e9

    cov = coverage_pct(light, heavy, grid, budget)
    budget_used_pct = 100.0 * fleet_cost(light, heavy) / budget
    return 0.75 * cov - 0.25 * budget_used_pct


def crossover(a: list[int], b: list[int], budget: int) -> list[int]:
    light = (a[0] + b[0]) // 2
    heavy = (a[1] + b[1]) // 2
    return clip_to_budget(light, heavy, budget)


def mutate(chrom: list[int], budget: int) -> list[int]:
    light, heavy = chrom
    if random.random() < 0.5:
        light += random.choice([-1, 1])
    else:
        heavy += random.choice([-1, 1])
    return clip_to_budget(light, heavy, budget)


def tournament_pick(pop: list[list[int]], grid: Grid, budget: int, k: int = 3) -> list[int]:
    pick = random.sample(pop, min(k, len(pop)))
    return max(pick, key=lambda c: fitness(c, grid, budget))


@dataclass
class FleetResult:
    light_count: int
    heavy_count: int
    total_cost: int
    payload_kg: int
    coverage_pct: float
    budget_used_pct: float
    score: float


def select_fleet_ga(
    grid: Grid,
    budget: int,
    pop_size: int = 24,
    generations: int = 40,
    mutation_rate: float = 0.25,
    seed: int | None = None,
) -> FleetResult:
    """
    Small GA: chromosome = [light_count, heavy_count].
    """
    if budget < LIGHT_COST:
        raise ValueError("budget must cover at least one light drone ($1000)")
    if seed is not None:
        random.seed(seed)

    population = [random_chromosome(budget) for _ in range(pop_size)]

    for _ in range(generations):
        population.sort(key=lambda c: fitness(c, grid, budget), reverse=True)
        best = population[0][:]
        next_pop = [best]
        while len(next_pop) < pop_size:
            p1 = tournament_pick(population, grid, budget)
            p2 = tournament_pick(population, grid, budget)
            child = crossover(p1, p2, budget)
            if random.random() < mutation_rate:
                child = mutate(child, budget)
            next_pop.append(child)
        population = next_pop

    population.sort(key=lambda c: fitness(c, grid, budget), reverse=True)
    best_light, best_heavy = population[0]
    cost = fleet_cost(best_light, best_heavy)
    payload = fleet_payload_kg(best_light, best_heavy)
    cov = coverage_pct(best_light, best_heavy, grid, budget)
    bu = 100.0 * cost / budget
    sc = fitness([best_light, best_heavy], grid, budget)

    return FleetResult(
        light_count=best_light,
        heavy_count=best_heavy,
        total_cost=cost,
        payload_kg=payload,
        coverage_pct=cov,
        budget_used_pct=bu,
        score=sc,
    )


def format_fleet_summary(r: FleetResult) -> str:
    lines = [
        "Fleet (GA)",
        f"  Light: {r.light_count}  Heavy: {r.heavy_count}",
        f"  Cost: ${r.total_cost}  (budget used: {r.budget_used_pct:.1f}%)",
        f"  Payload: {r.payload_kg} kg  |  Coverage (vs reachable demand): {r.coverage_pct:.1f}%",
        f"  Score: {r.score:.2f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from src.grid_model import Zone

    g = Grid(10, 10, zone_limits={Zone.RESIDENTIAL: 20, Zone.COMMERCIAL: 20})
    g.populate_grid()
    res = select_fleet_ga(g, budget=10_000, seed=42)
    print(format_fleet_summary(res))
