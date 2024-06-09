"""
See main script at the bottom (to run the code).
"""

from test_data import TEST_DATA

import pulp
from pulp.apis import PULP_CBC_CMD
from pulp import lpSum

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    NonNegativeIntegers,
    SolverFactory,
    sum_product,
    ConstraintList,
    minimize,
)
from pyomo.opt import SolverStatus, TerminationCondition

from collections import Counter
from random import shuffle
from sys import platform
import os


def _get_project_root_abs_path():
    return os.path.dirname(os.path.realpath(__file__))


SOLVER_TIMEOUT_SECONDS = 30
GAP_TOLERANCE = 0.03
if platform == "linux" or platform == "linux2":
    venv_executable_loc = (
        "/venv/lib/python3.11/site-packages/pulp/solverdir/cbc/linux/64/cbc"
    )
else:
    # Consider as Windows
    venv_executable_loc = r"\venv\Lib\site-packages\pulp\solverdir\cbc\win\64\cbc.exe"
CBC_EXECUTABLE_PATH = str(_get_project_root_abs_path()) + venv_executable_loc


def get_var_index_from_var_name(lp_variable) -> str:
    return lp_variable.name.split("x")[1]


def calculate_solution_pulp(
    combos_per_bar: dict[int, list[tuple[int, ...]]],
    parts: list[int],
    bars_quantities: dict[int, int],
    log_solver_logs,  # To print out all the logs during the LP nesting process
    timeout_seconds,
    already_selected_patterns={},
):
    lp = pulp.LpProblem("amel_solution", pulp.LpMinimize)

    all_combos = [i for v in combos_per_bar.values() for i in v]
    combos_unique_sorted = sorted(set(all_combos), key=lambda c: (-sum(c), len(c)))
    parts_qtys = Counter(parts)
    all_bars_sum = sum(b for b in bars_quantities)

    combos_2_total_available_bars = {}
    for b, cs in combos_per_bar.items():
        for c in cs:
            if c not in combos_2_total_available_bars:
                combos_2_total_available_bars[c] = 0
            if bars_quantities[b] == 0:
                combos_2_total_available_bars[c] = -1
            if combos_2_total_available_bars[c] != -1:
                combos_2_total_available_bars[c] += 1

    # one combo is one variable
    x = [
        pulp.LpVariable(f"x{i}", lowBound=0, cat="Integer")
        for i in range(len(combos_unique_sorted))
    ]

    # add objective function (only one function can be, that is why we sum it up to an expression.)
    lp += all_bars_sum * lpSum(x)

    # add constraints
    lp += all_bars_sum * lpSum(x) >= sum(
        parts
    )  # negative waste means solution len is larger than bars

    combo_2_parts_count = {c: Counter(c) for c in combos_unique_sorted}
    nr_of_combos = len(combos_unique_sorted)
    for p, q in parts_qtys.items():
        lp += (
            lpSum(
                x[i] * combo_2_parts_count[combos_unique_sorted[i]][p]
                for i in range(nr_of_combos)
            )
            == q
        )

    # If bar quantities are provided than set that constraint as well. If not
    # the algorithm will work assuming infinite amount for undefined bar.
    for i, c in enumerate(combos_unique_sorted):
        if combos_2_total_available_bars[c] > 0:
            lp += x[i] <= combos_2_total_available_bars[c]

    lp.solve(
        PULP_CBC_CMD(
            msg=log_solver_logs, timeLimit=timeout_seconds, gapRel=GAP_TOLERANCE
        )
    )  # Turn off printing out all the detailed messages
    if log_solver_logs:
        print("STATUS:", pulp.LpStatus[lp.status])

    selected_patterns = [
        (combos_unique_sorted[int(get_var_index_from_var_name(v))], int(v.varValue))
        for v in lp.variables()
        if v.name != "__dummy" and v.varValue > 0
    ]

    selected_patterns = sorted(selected_patterns, key=lambda x: -sum(x[0]))

    bars_quantities = dict(sorted(bars_quantities.items()))
    # Start from largest combo length and lowest bar length to find first best fits.
    bars_combos = {}
    empty_bar_key = ""
    for c, n in selected_patterns:
        combo_len = sum(c)
        if empty_bar_key:
            del bars_quantities[empty_bar_key]
            empty_bar_key = ""
        for b, q in bars_quantities.items():
            if b >= combo_len:
                if b not in bars_combos:
                    bars_combos[b] = {}

                if q == 0:
                    bars_combos[b][c] = n
                    break
                elif q == n:
                    bars_combos[b][c] = n
                    bars_quantities[b] -= n
                    empty_bar_key = b
                    break
                elif q > n:
                    bars_combos[b][c] = n
                    bars_quantities[b] -= n
                    break
                elif q < n:
                    bars_combos[b][c] = q
                    n -= q
                    bars_quantities[b] = 0
                    empty_bar_key = b
                    break

                else:
                    raise Exception("Unexpected if else case.")

    return lp.status, {**already_selected_patterns, **bars_combos}


PROBLEMATIC_STATUSES = (SolverStatus.aborted, SolverStatus.warning)


def calculate_solution_pyomo(
    combos_per_bar: dict[int, list[tuple[int, ...]]],
    parts: list[int],
    bars_quantities: dict[int, int],
    log_solver_logs,  # Adjust this for Pyomo logging if necessary
    timeout_seconds,
    already_selected_patterns={},
):
    """
    This uses `pyomo` library BUT `pulp's` CBC solver, because it does not have it's own solver.
    """

    model = ConcreteModel()

    all_combos = [i for v in combos_per_bar.values() for i in v]
    combos_unique_sorted = sorted(set(all_combos), key=lambda c: (-sum(c), len(c)))
    parts_qtys = Counter(parts)
    all_bars_sum = sum(bars_quantities.keys())

    # Variables
    model.x = Var(range(len(combos_unique_sorted)), domain=NonNegativeIntegers)

    # Objective function
    model.objective = Objective(expr=sum(model.x[i] for i in model.x), sense=minimize)

    # Constraints
    model.constraints = ConstraintList()

    # Constraint for total bar lengths
    # (negative waste means solution len is larger than bars)
    model.constraints.add(
        (all_bars_sum * sum(model.x[i] for i in model.x)) >= sum(parts)
    )

    # Constraints for parts quantities
    combo_2_parts_count = {c: Counter(c) for c in combos_unique_sorted}
    nr_of_combos = len(combos_unique_sorted)
    for p, q in parts_qtys.items():
        model.constraints.add(
            sum(
                model.x[i] * combo_2_parts_count[combos_unique_sorted[i]].get(p, 0)
                for i in range(nr_of_combos)
            )
            == q
        )

    # Constraint for total available bars per combo
    # If bar quantities are provided than set that constraint as well. If not
    # the algorithm will work assuming infinite amount for undefined bar.
    combos_2_total_available_bars = {c: 0 for c in all_combos}
    for b, cs in combos_per_bar.items():
        for c in cs:
            if bars_quantities[b] == 0:
                combos_2_total_available_bars[c] = -1
            elif combos_2_total_available_bars[c] != -1:
                combos_2_total_available_bars[c] += 1

    for i, c in enumerate(combos_unique_sorted):
        if combos_2_total_available_bars[c] > 0:
            model.constraints.add(model.x[i] <= combos_2_total_available_bars[c])

    # Solve the model
    SOLVER_NAME_PYOMO = "cbc"
    solver = SolverFactory(SOLVER_NAME_PYOMO, executable=CBC_EXECUTABLE_PATH)
    if "cbc" in SOLVER_NAME_PYOMO:
        solver.options["seconds"] = timeout_seconds
    else:
        raise ValueError(f"Unkwnon solver name provided '{SOLVER_NAME_PYOMO}'.")
    if log_solver_logs:
        solver.options["log"] = 1  # Adjust as necessary for Pyomo/CBC
    results = solver.solve(model, tee=log_solver_logs)

    if results.Solver.Status in PROBLEMATIC_STATUSES:
        raise Exception(f"Timeout limit reached. Status: {results.Solver.Status}")

    # Loading solution into results object
    model.solutions.load_from(results)

    if log_solver_logs:
        print("SolverStatus:", results.solver.status)
        print("Termination condition:", results.solver.termination_condition)
    if results.solver.status == SolverStatus.ok:
        status = 1
    else:
        status = -1

    selected_patterns = [
        (combos_unique_sorted[i], model.x[i].value)
        for i in model.x
        if model.x[i].value > 0
    ]
    selected_patterns = sorted(selected_patterns, key=lambda x: -sum(x[0]))

    bars_quantities = dict(sorted(bars_quantities.items()))
    # Start from largest combo length and lowest bar length to find first best fits.
    bars_combos = {}
    empty_bar_key = ""
    for c, n in selected_patterns:
        combo_len = sum(c)
        if empty_bar_key:
            del bars_quantities[empty_bar_key]
            empty_bar_key = ""
        for b, q in bars_quantities.items():
            if b >= combo_len:
                # Initialize empty bar counter
                if b not in bars_combos:
                    bars_combos[b] = {}

                if q == 0:
                    bars_combos[b][c] = n
                    break
                elif q == n:
                    bars_combos[b][c] = n
                    bars_quantities[b] -= n
                    empty_bar_key = b
                    break
                elif q > n:
                    bars_combos[b][c] = n
                    bars_quantities[b] -= n
                    break
                elif q < n:
                    bars_combos[b][c] = q
                    n -= q
                    bars_quantities[b] = 0
                    empty_bar_key = b
                    break

                else:
                    raise Exception("Unexpected if else case.")

    return status, {**already_selected_patterns, **bars_combos}


def calculate_solution(
    method,
    combos_per_bar: dict[int, list[tuple[int, ...]]],
    parts: list[int],
    bars_quantities: dict[int, int],
    log_solver_logs=False,  # To print out all the logs during the LP nesting process
    timeout_seconds=300,
    already_selected_patterns={},
):
    args = (
        combos_per_bar,
        parts,
        bars_quantities,
        log_solver_logs,
        timeout_seconds,
        already_selected_patterns,
    )
    if method == 1:
        return calculate_solution_pulp(*args)
    elif method == 2:
        return calculate_solution_pyomo(*args)
    raise Exception("Wrong method number selected.")


def _get_parts_list(parts_quantities: dict):
    return [l for l, q in parts_quantities.items() for _ in range(q)]


METHOD_PULP = 1
METHOD_PYOMO = 2

if __name__ == "__main__":

    TEST_CASE = 2  # NOTE: change this for different cases (add into data.py own cases)

    _td = TEST_DATA[TEST_CASE]
    status, results = calculate_solution(
        method=METHOD_PULP,
        combos_per_bar=_td["possible_combos_per_bar"],
        parts=_get_parts_list(_td["parts_quantities"]),
        bars_quantities=_td["bars_quantities"],
    )

    # Print out results:
    print("Status:", status)
    print("Combinations per bar:")
    for bar, combos in results.items():
        print("bar:", bar)
        for combo, repetitions in combos.items():
            print("     ", combo, "x", repetitions)
