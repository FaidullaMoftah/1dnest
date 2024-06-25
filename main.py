import ast
import math
from ortools.sat.python import cp_model
from pyvpsolver.solvers import mvpsolver
from sortedcontainers import SortedList
import test_data

data = test_data.TEST_DATA


class problem:

    def __init__(self, p):

        self.bars = [(key, value) for key, value in p['bars_quantities'].items()]
        self.parts = [(key, value) for key, value in p['parts_quantities'].items()]
        keys = [key for key, value in p['parts_quantities'].items()]

        self.inv_parts, self.in_keys = {}, {}

        for ind, bars in enumerate(self.bars):
            self.inv_parts[bars[0]] = ind

        for ind, part in enumerate(self.parts):
            self.inv_parts[part[0]] = ind

        self.bars_list = []
        self.parts_list = []

        # CP_SOLVER
        self.CP_SOLVER_TIME_LIMIT = 150
        self.CP_SOLVER_GAP_LIMIT = 0.001
        self.CP_NUM_CORES = 8
        self.CP_LOGGING = 0

        for pt in self.parts:
            for _ in range(pt[1]):
                self.parts_list.append(pt[0])
        # for bars with infinite quantity
        # put the highest number you might need
        for b in self.bars:
            times = b[1]
            if b[1] == 0:
                times = self.find_count(b[0])
            for _ in range(times):
                self.bars_list.append(b[0])

        # solution
        self.best_cost = 1e20
        self.best_solution = None
    def find_count(self, _x):
        _list = self.parts_list.copy()
        _list = sorted(_list)
        n = 1
        left = _x
        for it in _list:
            if it > _x:
                continue
            elif it < left:
                left -= it
            else:
                left = _x - it
                n += 1
        return n

    def largest_in_smallest(self):
        bl = [(cap, id_) for id_, cap in zip(range(len(self.bars_list)), self.bars_list)]
        solution = [-1] * len(self.parts_list)
        sl = SortedList(bl)
        pl = sorted(self.parts_list, reverse=True)
        for part in range(len(pl)):
            index = sl.bisect_left((pl[part], -1))
            if index < len(sl):
                w, i = sl[index]
                sl.pop(index)
                sl.add((w - pl[part], i))
                solution[part] = i
        return solution

    # Minimizing this is equivalent to minimizing sum of sizes of unused bins
    def loss(self, solution):
        loss = 0
        done = [0] * len(self.bars_list)
        for i in range(len(solution)):
            if solution[i] == -1:
                loss = math.inf
            done[solution[i]] = 1
        for i in range(len(self.bars_list)):
            loss += done[i] * self.bars_list[i]
        return loss

    def polynomial_hash(self, arr, base=31, mod=1_000_000_007):
        arr = sorted(arr)
        hash_value = 0
        for i, element in enumerate(arr):
            hash_value = (hash_value * base + element) % mod
        return hash_value

    def score(self, solution):
        print(f"Used total of {self.loss(solution)} , Wasting {self.loss(solution) - sum(self.parts_list)}."
              f"\nThis amounts to {(self.loss(solution) - sum(self.parts_list)) / self.loss(solution) * 100}% of materials.")
    # This function finds any feasible solution, it is usually very bad. if you want to minimize use
    # cp_sat solver or VPSolver instead, its only merit is to check whether a problem is solvable, fast.
    def parse_sol(self, solution):
        d = {}
        for i in range(len(solution)):
            if solution[i] not in d:
                d[solution[i]] = []
            d[solution[i]].append(self.parts_list[i])
        inverse = {}
        for key, val in d.items():
            inverse[self.polynomial_hash(val)] = val
        sol = {}
        sizes = {}
        for key, val in self.bars:
            sizes[key] = {}
            sol[key] = {}
        for key, value in d.items():
            value = sorted(value)
            h = self.polynomial_hash(value)
            size = self.bars_list[key]
            if size not in sol:
                sol[size] = {}
            if h not in sol[size]:
                sol[size][h] = 0
            sol[size][h] += 1

        final = {}
        #final = {size -> (total size, [(count, [pattern]))}
        for key, val in self.bars:
            final[key] = [0, []]
        for key, val in self.bars:
            cnt = 0
            for key2, val2 in sol[key].items():
                cnt += val2
                final[key][1].append((val2, inverse[key2]))
            final[key][0] = cnt
        final = [(key, value) for key, value in final.items()]
        ans = ""

        for i in range(len(final)):
            ans += f"Bar number {i} : {final[i][1][0]} Patterns:\n"
            for j in range(0, len(final[i][1][1])):
                cnt, pat = final[i][1][1][j][0], final[i][1][1][j][1]
                app = list(map(lambda x:x, pat))
                ans += f"{cnt}x {str(app)}\n"
        print(ans)

    def or_multiple_knapsack_sat(self):
        d = {}

        d['c_bins'] = len(self.bars_list)
        d['c_parts'] = len(self.parts_list)
        d['bins'] = self.bars_list
        d['parts'] = self.parts_list

        model = cp_model.CpModel()

        x = {}
        for i in range(d['c_parts']):
            for j in range(d['c_bins']):
                x[i, j] = model.new_bool_var(f"x_{i}_{j}")

        for i in range(d["c_parts"]):
            model.add_at_most_one(x[i, b] for b in range(d["c_bins"]))

        for b in range(d["c_bins"]):
            model.add(
                sum(x[i, b] * d["parts"][i] for i in range(d["c_parts"]))
                <= d["bins"][b]
            )

        objective = []
        for i in range(d["c_parts"]):
            for b in range(d["c_bins"]):
                objective.append(cp_model.LinearExpr.Term(x[i, b], 1))
        model.maximize(cp_model.LinearExpr.sum(objective))
        solver = cp_model.CpSolver()
        status = solver.solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = [-1] * d['c_parts']

            for bar in range(len(d['bins'])):
                for item in range(len(d['parts'])):
                    if solver.value(x[item, bar]) > 0:
                        solution[item] = bar
            return solution
        else:
            raise Exception('No feasible solution found')

    def solver_sat(self):
        d = {}

        # The main weakness of this formulation is using total_count(bars) * total_count(items) variables,
        # which makes convergence very slow
        d['c_bins'] = len(self.bars_list)
        d['c_parts'] = len(self.parts_list)
        d['bins'] = self.bars_list
        d['parts'] = self.parts_list

        model = cp_model.CpModel()
        # min sum y_i a_i (y_i is binary, whether a bar is taken by any part)
        # st:   positivity constraints on x and y
        #       sum_i x_ij = 1 (all items are taken)
        #       sum_i x_ij w_i <= a_j (no bin is overfilled)
        #       yj >= max_i x_ij (big one!) (yj is the correct indicator variable or whether bin j is occupied)

        x, y = {}, {}
        for i in range(d['c_parts']):
            for j in range(d['c_bins']):
                x[i, j] = model.new_bool_var(f"x_{i}_{j}")
        for j in range(d['c_bins']):
            y[j] = model.new_bool_var(f"y_{j}")

        for i in range(d["c_parts"]):
            model.add_exactly_one(x[i, b] for b in range(d["c_bins"]))

        for b in range(d["c_bins"]):
            model.add(
                sum(x[i, b] * d["parts"][i] for i in range(d["c_parts"]))
                <= d["bins"][b]
            )

        for j in range(d["c_bins"]):
            model.add_max_equality(y[j], [x[i, j] for i in range(d['c_parts'])])

        objective = []
        for j in range(d["c_bins"]):
            objective.append(cp_model.LinearExpr.Term(y[j], d['bins'][j]))
        model.minimize(cp_model.LinearExpr.sum(objective))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.CP_SOLVER_TIME_LIMIT
        solver.parameters.relative_gap_limit = self.CP_SOLVER_GAP_LIMIT
        solver.parameters.num_workers = self.CP_NUM_CORES

        solver.parameters.log_search_progress = self.CP_LOGGING
        solver.log_callback = print

        status = solver.solve(model)

        print(solver.status_name(status))
        solution = [-1] * d['c_parts']

        for bar in range(len(d['bins'])):
            for item in range(len(d['parts'])):
                if solver.value(x[item, bar]) > 0:
                    solution[item] = bar
        print("Objective:", objective)
        return self.parse_sol(solution)

    def VP(self):
        Ws = [[key] for key, val in self.bars]
        Cs = [key for key, val in self.bars]
        Qs = [-1 if val <= 0 else val for key, val in self.bars]
        ws = [[[part]] for part, count in self.parts]
        b = [count for part, count in self.parts]

        sol = mvpsolver.solve(Ws, Cs, Qs, ws, b, script="vpsolver_glpk.sh", verbose=False)
        obj, lst_sol = sol
        if obj is not None:
            print("Objective:", obj)
        print("Solution:")
        for i, sol in enumerate(lst_sol):
            cnt = sum(m for m, p in sol)
            print("Bins of type {0}: {1} {2}".format(
                Ws[i][0], cnt, ["bins", "bin"][cnt == 1]
            ))
            for mult, patt in sol:
                print("{0} x [{1}]".format(
                    mult, ", ".join(
                        ["{0}".format(ws[it][0][0]) for it, opt in patt]
                    )
                ))
        return sol
tc = test_data.TEST_DATA[5]
#problem(tc).solver_sat()
problem(tc).VP()
