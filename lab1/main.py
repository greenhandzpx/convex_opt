import cvxpy as cp
import numpy as np
import time

SOLVER_NAME = ["ECOS", "SCS", "OSQP", "CVXOPT"]
SOLVERS = [cp.ECOS, cp.SCS, cp.OSQP, cp.CVXOPT]


def solve_and_timing(x, prob, loop=1000):
    for solver, name in zip(SOLVERS, SOLVER_NAME):
        print("====== Use solver {} ======".format(name))
        start = time.perf_counter()
        try:
            for _ in range(loop):
                result = prob.solve(solver=solver)
        except Exception as e:
            print("{} failed to solve this problem, error: {}".format(name, e))
            continue
        end = time.perf_counter()
        print("The optimal value: {}".format(result))
        print("The optimal solution: {}".format(x.value))
        print("Total time: {}".format((end - start) / loop))
    print("\n")

def linear_program():
    A = np.array([[2, 1], [1, 3]])
    b = np.array([1, 1])
    c = np.array([-1, -1])
    x = cp.Variable(2)
    prob = cp.Problem(cp.Maximize(c.T @ x),
                      [A @ x >= b, 
                       x >= 0])
    print("linear_program:")
    solve_and_timing(x, prob)

def quadratic_program():
    x = cp.Variable(2)
    Q = np.array([[2, 0], [0, 2]])
    c = np.array([-2, -6])
    A = np.array([[1, 1], [-1, 2], [2, 1]])
    b = np.array([2, 2, 3])
    objective = cp.Minimize(cp.quad_form(x, Q) / 2 + c @ x)
    constraints = [A @ x <= b]
    prob = cp.Problem(objective, constraints)
    print("quadratic_program:")
    solve_and_timing(x, prob)

def qcqp():
    x = cp.Variable(2)
    objective = cp.Minimize(cp.sum_squares(x) + x @ np.array([2, 4]))
    constraints = [cp.sum_squares(x) <= 1, cp.sum(x) <= 0]
    prob = cp.Problem(objective, constraints)
    print("qcqp:")
    solve_and_timing(x, prob)


if __name__ == "__main__":
    linear_program()
    quadratic_program()
    qcqp()