import numpy as np

alpha = 2
K = 20
T = 10000
eta = 1 / np.sqrt(T)

with open('Loss_matrix.txt', 'r') as file:
    lines = file.readlines()

mat = [list(map(float, line.split())) for line in lines]
C = np.array(mat)
# print(C)

def binary_search(f, low, high):
    # print("low {}, high {}".format(low, high))
    while low < high:
        mid = (low + high) / 2
        # print("mid {}, f {}".format(mid, f(mid)))
        val = f(mid)
        if val < 1e-10 and val > -1e-10:
            # print("low {}, high {}, ret {}".format(low, high, mid))
            return mid
        if val > 0:
            high = mid
        elif val < 0:
            low = mid
    # print("low {}, high {}, ret {}, f {}".format(low, high, mid, f(mid)))
    return mid

def lagrangian_multiplier(c_tilde: np.ndarray, alpha, eta, p_t: np.ndarray) -> np.ndarray:
    '''
        We can solve the problem using KKT.
        Orig feasiblity: 
        p_1 + p_2 + ... + p_K = 1
        p_i >= 0
        Dual feasibilty: 
        lambda_i >= 0
        v >= 0
        Complementary Slackness:
        lambda_i * p_i = 0
        Stability:
        L.grad(p_i) = 0
        Since p_i cannot be zero in Bregman Divergence (p_t_i will be the denominator),
        we can easily conclude that lambda_i will all be zero, then only `v` and `p_i` will be the variable.
    '''

    def calculate_p_i(p_t_i, alpha, eta, v, c_t_i):
        # print("v2 {}".format(v))
        '''
            Using Stability condition to calculate p_i .
            Note that since alpha is 2 in this problem, v is positively correlated with p_i.
        '''
        return (p_t_i ** (1 / alpha - 1) - eta * (v - c_t_i)) ** (alpha / (1 - alpha))

    def orig_feasibilty(c_tilde, alpha, eta, p_t, v):
        # print("v1 {}, len {}".format(v, len(p_t)))
        '''
            p_1 + p_2 + ... + p_K = 1
        '''
        tmp_p = np.array(
            [calculate_p_i(p_t[i], alpha, eta, v, c_tilde[i]) for i in range(len(p_t))]
        )
        # for i in range(len(p_t)):
        #     print("tmp_p_i {}: {}".format(i, tmp_p[i]))
        return tmp_p.sum() - 1

    bs_fn = lambda v: orig_feasibilty(c_tilde, alpha, eta, p_t, v)
    v_lower_bound, v_upper_bound = 0, c_tilde.max()
    # Since bs_fn is Monotonically incremental with `v`, we can use binary search to find zero point .
    v = binary_search(bs_fn, v_lower_bound, v_upper_bound)

    return np.array([calculate_p_i(p_t[i], alpha, eta, v, c_tilde[i]) for i in range(len(p_t))])


def tsallis_inf_plus(T, alpha, eta, K, C):
    def sample_from_p(p: np.ndarray):
        return np.random.choice(len(p), 1, p=p)

    choose_action_cost = 0
    p_t = np.full(K, 1 / K)
    for t in range(T):
        I_t = sample_from_p(p_t)

        c_t_I_t = C[t][I_t]
        choose_action_cost += c_t_I_t
        c_tilde = np.zeros(K)
        if p_t[I_t] >= eta:
            c_tilde[I_t] = c_t_I_t / p_t[I_t]
        else:
            c_tilde[I_t] = c_t_I_t / (p_t[I_t] + eta)
        # print("c_tilde {}".format(c_tilde))
        p_t = lagrangian_multiplier(c_tilde, alpha, eta, p_t) 
    pos, action = p_t[0], 0
    for i in range(len(p_t)):
        if p_t[i] > pos:
            pos = p_t[i]
            action = i
    print("Final p_t: {}".format(p_t))
    print("Action with the greatest possbility: {}, possbility: {}".format(action, pos))
    print("Chosen actions' total cost: {}".format(choose_action_cost))

if __name__ == "__main__":
    # print("C shape {}".format(C.shape))
    action_real_costs = np.zeros(K)
    for c_t in C:
        for i in range(len(c_t)):
            action_real_costs[i] += c_t[i]
    print("Actions' real cost: {}".format(action_real_costs))
    minimum, mini_action = action_real_costs[0], 0
    for i in range(len(action_real_costs)):
        if action_real_costs[i] < minimum:
            minimum = action_real_costs[i]
            mini_action = i
    print("Best action with minimum cost: {}".format(mini_action))

    tsallis_inf_plus(T, alpha, eta, K, C)
