import numpy as np
from scipy.optimize import minimize


class Benchamrk:
    def __init__(self, n_samples, n_features):
        self.n_samples = n_samples
        self.n_features = n_features
        # np.random.seed(1)
        self.X = np.random.randn(n_samples, n_features)
        # 生成稀疏系数
        beta_true = np.zeros(n_features)
        non_zero_indices = np.random.choice(n_features, size=10, replace=False)
        beta_true[non_zero_indices] = np.random.randn(10)
        # 生成目标变量
        self.y = self.X.dot(beta_true) + 0.01 * np.random.randn(n_samples)

    def mse(self, beta):
        err = self.X @ beta - self.y 
        # print("err shape", err.shape)
        return 0.5 * np.linalg.norm(err, ord=2)**2 / self.n_samples

    def sparsity(self, beta, lambda1, lambda2):
        return lambda1 * np.linalg.norm(beta, ord=1) + lambda2 * np.linalg.norm(beta, ord=2)**2

    def opt_func(self, beta, lambda1, lambda2):
        # beta = x[:n_features]
        # lambda1 = x[n_features]
        # lambda2 = x[n_features + 1]
        return self.mse(beta) + self.sparsity(beta, lambda1, lambda2)
        
    def mse_grad(self, beta):
        return 1 / self.n_samples * self.X.T @ (self.X @ beta - self.y)
    def sparsity_grad(self, beta, lambda1, lambda2):
        grad1 = lambda1 * np.sign(beta)
        grad2 = 2 * lambda2 * beta
        return grad1 + grad2
    def opt_grad(self, beta, lambda1, lambda2):
        # beta = x[:n_features]
        # lambda1 = x[n_features]
        # lambda2 = x[n_features + 1]
        return self.mse_grad(beta) + self.sparsity_grad(beta, lambda1, lambda2)

    def print_result(self, method_name, beta, lambda1, lambda2, n_iters, converged):
        print("========================{}: ========================".format(method_name))
        print("N samples: {}, N features: {}".format(self.n_samples, self.n_features))
        print("(Initial) Learning rate: {}, lambda1: {}, lambda2: {}".format(self.learning_rate, self.lambda1, self.lambda2))
        print("Converged: {}, n_iters: {}".format(converged, n_iters))
        print("Final beta: {}".format(beta))
        print("Objective function minimum result: {}".format(self.opt_func(beta, lambda1, lambda2)))
        print("MSE minimum result: {}".format(self.mse(beta)))

    def lagrange_dual(self):
        # beta = np.zeros(n_features)
        
        def inner_func(beta, v, k):
            abs_beta = np.abs(beta) 
            limit = self.n_features - k + 1 if self.n_features > k - 1 else 0
            indices_to_zero = abs_beta.argsort()[:int(limit)]
            beta[indices_to_zero] = 0
            # print("beta norm", np.linalg.norm(beta, ord=0))
            return self.mse(beta) + v * (np.linalg.norm(beta, ord=0) - k)

        # def outer_func(x, k):
        def outer_func(v, k):
            beta = np.ones(self.n_features)
            beta /= 1000
            result = minimize(inner_func, beta, (v[0], k))
            # print("inner succ", result.success)
            # print("inner msg", result.message)
            return -result.fun
        
        # Constraint for v
        upper_bound = 5e+6
        constraint1 = {'type': 'ineq', 'fun': lambda v: v[0] - 1e-4}
        constraint2 = {'type': 'ineq', 'fun': lambda v: -v[0] + upper_bound}

        # max L(beta, v)
        k = self.n_features / 3
        v = np.zeros(1)
        v[0] = upper_bound / 10 
        result = minimize(outer_func, v, (k), constraints=[constraint1, constraint2])
        # result = minimize(outer_func, v, (k), constraints=[constraint1])

        print("========================Lagrange dual: ========================")
        v = result.x
        # print("succ", result.success)
        # print("msg", result.message)
        # beta = np.zeros(n_features)
        beta = np.ones(self.n_features)
        beta /= 1000
        inner_res = minimize(inner_func, beta, (v[0], k))
        # print("succ", inner_res.success)
        # print("msg", inner_res.message)
        beta = inner_res.x
        print("Final beta: {}".format(beta))
        print("Final beta's Zero norm: {}".format(np.linalg.norm(beta, ord=0)))
        print("Final v: {}".format(v))
        print("MSE minimum result: {}".format(self.mse(beta)))

    def gradient_descent(self, lambda1=1e-5, lambda2=1e-5, learning_rate=0.01, n_iters=5000, err=1e-5):
        n_features = self.n_features
        # beta = np.ones(n_features)
        beta = np.zeros(n_features)
        converged = False
        for i in range(n_iters):
            grad = self.opt_grad(beta, lambda1, lambda2)
            old_loss = self.opt_func(beta, lambda1, lambda2)  
            # print("gd: res ", opt_func(beta, lambda1, lambda2))
            beta = beta - learning_rate * grad
            loss_diff = abs(old_loss - self.opt_func(beta, lambda1, lambda2))
            if loss_diff < err:
                converged = True
                n_iters = i + 1
                # print("Converged after {} iters".format(i + 1))
                break

        self.print_result("Gradient descent", beta, lambda1, lambda2, n_iters, converged)

    def steepest_descent(self, lambda1=1e-5, lambda2=1e-5, learning_rate=0.01, n_iters=1000, err=1e-5):
        n_features = self.n_features
        # beta = np.ones(n_features)
        beta = np.zeros(n_features)

        converged = False
        def optimize_learning_rate(learning_rate, grad):
            fn = lambda learing_rate: self.opt_func(beta - learing_rate * grad, lambda1, lambda2)
            constraint = {'type': 'ineq', 'fun': lambda lr: lr - 1e-10}
            result = minimize(fn, learning_rate, constraints=[constraint])
            # print("opt lr suc: {}".format(result.success))
            return result.x
        
        for i in range(n_iters):
            grad = self.opt_grad(beta, lambda1, lambda2)
            old_loss = self.opt_func(beta, lambda1, lambda2)  
            learning_rate = optimize_learning_rate(learning_rate, grad)
            beta = beta - learning_rate * grad
            loss_diff = abs(old_loss - self.opt_func(beta, lambda1, lambda2))
            if loss_diff < err:
                converged = True
                n_iters = i + 1
                # print("Converged after {} iters".format(i + 1))
                break

        self.print_result("Steepest descent", beta, lambda1, lambda2, n_iters, converged)

    def newton(self, lambda1=1e-5, lambda2=1e-5, n_iters=5000, err=1e-5):
        n_features = self.n_features
        n_samples = self.n_samples
        # beta = np.ones(n_features)
        beta = np.zeros(n_features)
        converged = False
        def hessian(beta, lambda1, lambda2):
            # lambda2 = x[n_features + 1]
            return 1 / n_samples * self.X.T @ self.X + 2 * lambda2 * np.diag(np.ones(n_features))

        for i in range(n_iters):
            grad = self.opt_grad(beta, lambda1, lambda2)
            old_loss = self.opt_func(beta, lambda1, lambda2)  
            beta = beta - np.linalg.inv(hessian(beta, lambda1, lambda2)) @ grad
            # x[:n_features] = x[:n_features] - np.linalg.inv(hessian(x)) @ grad
            loss_diff = abs(old_loss - self.opt_func(beta, lambda1, lambda2))
            if loss_diff < err:
                converged = True
                n_iters = i + 1
                # print("Converged after {} iters".format(i + 1))
                break

        self.print_result("Newton", beta, lambda1, lambda2, n_iters, converged)


    def coordinate_descent(self, lambda1=1e-5, lambda2=1e-5, n_iters=5000, err=1e-5):
        n_features = self.n_features
        # beta = np.ones(n_features)
        beta = np.zeros(n_features)
        converged = False
        def func(beta_i, i):
            # print("shape", beta_i.shape)
            beta[i] = beta_i[0]
            return self.opt_func(beta, lambda1, lambda2)

        for i in range(n_iters):
            old_loss = self.opt_func(beta, lambda1, lambda2)  
            for i in range(len(beta)):
                result = minimize(func, beta[i], i) 
                beta[i] = result.x[0]
            loss_diff = abs(old_loss - self.opt_func(beta, lambda1, lambda2))
            if loss_diff < err:
                converged = True
                n_iters = i + 1
                break

        self.print_result("Coordinate descent", beta, lambda1, lambda2, n_iters, converged)

    def benchmark(self, learning_rate, lambda1, lambda2):
        n_iters = 2000
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lagrange_dual()
        self.gradient_descent(lambda1=lambda1, lambda2=lambda2, learning_rate=learning_rate, n_iters=n_iters)
        self.steepest_descent(lambda1=lambda1, lambda2=lambda2, learning_rate=learning_rate, n_iters=n_iters)
        self.newton(lambda1=lambda1, lambda2=lambda2, n_iters=n_iters)
        self.coordinate_descent(lambda1=lambda1, lambda2=lambda2, n_iters=n_iters)


if __name__ == "__main__":

    n_samples_arr = [300, 500, 1000]

    n_features_arr = [50, 80, 100]

    learning_rates = [0.1, 0.01, 0.001]

    lambdas = [1e-5, 1e-3, 1e-1]

    for ns, nf in zip(n_samples_arr, n_features_arr):
        bench = Benchamrk(ns, nf)
        for lr, lambda1 in zip(learning_rates, lambdas):
            bench.benchmark(lr, lambda1, lambda1)