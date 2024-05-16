import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 0.02
RHO = 0.4
T = 10
GROUPS = [
    [0],
    [1],
    [2],
    [3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13],
    [14, 15],
    [16],
    [17],
    [18],
]
NUM_ITERS = 10000
FSTAR = 49.9649387126726

X_TRAIN_PATH = "./X_train.csv"
Y_TRAIN_PATH = "./Y_train.csv"


def load_data(X_train_path, y_train_path):
    X_train = np.loadtxt(X_train_path, delimiter=",")
    X_train = np.hstack([np.ones(X_train.shape[0])[:, np.newaxis], X_train])
    y_train = np.loadtxt(y_train_path, delimiter=",")
    return X_train, y_train


def objective(X, y, beta, reg, w, groups):
    """Compute the objective.

    Note: We do not regularize the bias term.

    Params:
        X: the training data matrix.
        y: the training targets.
        beta: the model parameters.
        reg: the regularization strength.
        w: the weights for each group.
        groups: the list of groups, where each group is a list of feature indices.

    Returns:
        objective: the full training objective
        error: the least-squares error (i.e. no regularization)
    """

    error = (0.5 * X.shape[0]) * np.linalg.norm(X @ beta - y) ** 2
    reg_term = reg * w.T @ np.array([np.linalg.norm(beta[groups[j]])for j in range(len(groups))])

    return error + reg_term, error


def gradient(X, y, beta):
    """Compute the gradient of the least-squares loss.

    Params:
        X: the training data matrix.
        y: the training targets.
        beta: the model parameters.

    Returns:
        g: the gradient of the least squares error.
    """
    return X.T @ (X @ beta - y) / X.shape[0]


def prox(v, groups, eta, w):
    """Evaluate the proximal operator of the group penalty.

    Note: we do not regularize the bias term.

    Params:
        v: the vector at which to evaluate the proximal operator.
        groups: the list of groups, where each group is a list of feature indices.
        eta: the parameter controlling the strength of the proximal term.
        w: the weights for each group.

    Returns:
        prox(v): the result of the proximal operator.
    """
    prox = np.zeros_like(v)
    for g_idx, j in enumerate(groups):
        
        coef = (1 - (eta * w[g_idx]) / np.linalg.norm(v[j]))
        prox[j] = np.max([0, coef]) * v[j]
    return prox


def run_admm(
    X,
    y,
    reg,
    rho,
    num_iters,
    fstar,
    groups,
):
    """Solve the group lasso problem using ADMM.
    Params:
        X: the training data matrix.
        y: the training targets.
        reg: the regularization strength.
        rho: the penalty strength for ADMM.
        num_iters: the number of iterations to run.
        fstar: the optimal value of the problem.
        groups: the list of groups, where each group is a list of feature indices.

    Returns:
        gaps: list of sub-optimalities f^k - f^*.
    """
    gaps = np.empty(num_iters)

    # compute the group weights.
    w = np.array([np.sqrt(len(g)) for g in groups])

    # initialize the primal and dual parameters:
    n = X.shape[0]
    m = X.shape[1]
    alpha = np.ones(m)
    beta = np.ones(m)
    u = np.ones(m)
    prefac_X = np.linalg.inv(X.T @ X / n + rho * np.eye(m))

    # run ADMM
    for iteration in range(num_iters):
        # primal updates
        beta = prefac_X @ ((X.T @ y / n) + rho * (alpha + u))
        alpha_copy = alpha.copy()
        for g_idx, j in enumerate(groups):
            alpha_sub = alpha_copy
            alpha_sub[j] = np.zeros(len(j))
            r_j = beta - u - alpha_sub
            
            coef = (1 - (reg * w[g_idx] / rho) / np.linalg.norm(r_j))
            alpha[j] = np.max([0, coef]) * r_j[j]

        # dual update
        u += alpha - beta

        # save sub-optimality
        f, _ = objective(X, y, beta, reg, w, groups)
        gaps[iteration] = f - fstar

    print("ADMM results:")
    for j in range(len(groups)):
        print("Group {}: {}".format(j, beta[groups[j]]))

    return gaps


def try_pgd_update(beta, t, g, groups, reg, w):
    """Evaluate PGD update with step-size t.
    Params:
        beta: the model parameters.
        t: the step-size to try.
        g: the gradient.
        groups: the list of groups, where each group is a list of feature indices.
        reg: the regularization strength.
        w: the weights for each group.

    Returns:
        beta_next: the result of one proximal-gradient step.
    """
    return prox(beta - t * g, groups, t * reg, w)


def ls_cond(f_diff, beta_diff, g, t):
    """Check if the proximal-gradient update satisfies the line-search
    condition.

    Params:
        f_diff: the difference in objectives, f_next - f.
        beta_diff: the difference in iterates.
        g: the gradient.
        t: the step-size.

    Returns:
        sat: boolean indicating whether or not the condition is satisfied.
    """
    lhs = f_diff
    rhs = g @ beta_diff + (0.5 / t) * np.linalg.norm(beta_diff) ** 2

    if lhs <= rhs:
        return True
    elif t < 1e8:
        return True
    else:
        return False


def run_pgd(
    X,
    y,
    reg,
    t,
    fstar,
    num_iters,
    groups,
    gamma=0.8,
    use_ls=False,
    use_acc=False,
    r_fista=False,
    d_fista=False,
    v_fista=False,
    kappa=None,
):
    """Solve the group lasso problem using (accelerated) proximal gradient
    descent.

    Params:
        X: the training data matrix.
        y: the training targets.
        reg: the regularization strength.
        t: the initial step-size.
        fstar: the optimal value of the problem.
        num_iters: the number of iterations to run.
        groups: the list of groups, where each group is a list of feature indices.
        gamma: the back-tracking constant.
        use_ls: whether or not to use a line-search.
        use_acc: whether or not to use acceleration.
        r_fista: whether or not to use scheduled restarts.
        d_fista: whether or not to use dynamic restarts.
        v_fista: whether or not to strongly convex fista.
        kappa: condition number of the problem.

    Returns:
        gaps: list of sub-optimalities f^k - f^*.
    """
    gaps = np.empty(num_iters)

    # compute the group weights.
    w = np.array([np.sqrt(len(g)) for g in groups])

    # initialize the model parameters.
    beta = np.random.randn(X.shape[1])
    beta[0] = 0

    # initialize extrapolation parameters.
    v = beta

    # secondary sequence for extrapolation step-size.
    q = 1.0

    # compute initial objective
    f, e = objective(X, y, beta, reg, w, groups)

    # least squares error evaluated at ve.
    ve = e

    # save starting step-size
    t0 = t

    if r_fista or v_fista:
        assert kappa is not None and kappa > 0
        restart_n = np.sqrt(8*kappa) - 1

    for iteration in range(num_iters):
        g = gradient(X, y, v)
        beta_plus = try_pgd_update(v, t, g, groups, reg, w)
        f_plus, e_plus = objective(X, y, beta_plus, reg, w, groups)

        while use_ls and not ls_cond(e_plus - ve, beta_plus - v, g, t):
            # perform backtracking line-search
            t = gamma * t



        if use_acc:
            # update these conditions for triggering a restart.
            if (d_fista and r_fista) or (r_fista):

                # evaluate restart condition
                if iteration % restart_n == 0:
                   restart_cond = True
                elif d_fista and (beta_plus-beta).T @ (v - beta_plus) >= 0:
                    restart_cond = True
                else:
                    restart_cond = False

                # check restart cond and restart if necessary.
                if restart_cond:
                    q_plus = 1
                    v = beta_plus
                    t = t0

            else:
                if v_fista:
                    # handle the strongly convex case.
                    v  = beta_plus + (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1) * (beta_plus - beta)

                else:
                    # handle the convex case
                    pass

                q_plus = 1 + 0.5 * np.sqrt(1 + 4 * q ** 2)

                # update extrapolation parameter.
                v = beta_plus + ((q-1) / q_plus) * (beta_plus - beta)

                # update objective at extrapolation parameter
                # necessary for line-search
                _, ve = objective(X, y, v, reg, w, groups)
                q = q_plus
        else:
            # default to proximal gradient step from last iterate.
            v = beta_plus
            ve = e_plus

        # update counters.
        beta = beta_plus
        f = f_plus
        e = e_plus

        # try to increase the step-size
        if use_ls:
            t = t / gamma

        gaps[iteration] = f - fstar

    print("Proximal Gradient results")
    for j in range(len(groups)):
        print("--- Group {}: {}".format(j, beta[groups[j]]))

    return gaps

if __name__ == "__main__":
    X_train, y_train = load_data(X_TRAIN_PATH, Y_TRAIN_PATH)

    # run ADMM
    admm_gaps = run_admm(X_train, y_train, LAMBDA, RHO, NUM_ITERS, FSTAR, GROUPS)

    # run PGD
        # plain
    pgd_gaps_e = run_pgd(X_train, y_train, LAMBDA, 0.5, FSTAR, NUM_ITERS, GROUPS)
        # with line search
    # pgd_gaps_g = run_pgd(X_train, y_train, LAMBDA, 10, FSTAR, NUM_ITERS, GROUPS, use_ls=True, gamma=0.8)
    #     # with acc - FISTA
    # eigens = np.linalg.eigvals(X_train.T @ X_train)
    # KAPPA = np.max(eigens) / np.min(eigens)
    # pgd_gaps_h = run_pgd(X_train, y_train, LAMBDA, 10, FSTAR, NUM_ITERS, GROUPS, use_ls=True, use_acc=True, gamma=0.8)
    #     # with acc - RFISTA
    # pgd_gaps_j = run_pgd(X_train, y_train, LAMBDA, 10, FSTAR, NUM_ITERS, GROUPS, use_acc=True, gamma=0.8, r_fista=True, kappa=KAPPA)
    #     # with acc - MFISTA
    # pgd_gaps_k = None
    #     # with acc - DFISTA
    # pgd_gaps_l = run_pgd(X_train, y_train, LAMBDA, 10, FSTAR, NUM_ITERS, GROUPS, use_ls=True, use_acc=True, gamma=0.8, r_fista=True, d_fista=True, kappa=KAPPA)
    #     # with acc - VFISTA
    # pgd_gaps_m = run_pgd(X_train, y_train, LAMBDA, 10, FSTAR, NUM_ITERS, GROUPS, use_ls=True, use_acc=True, gamma=0.8, r_fista=True, d_fista=True, v_fista=True, kappa=KAPPA)

    # plot the results
    plt.plot(admm_gaps, label="ADMM")
    plt.plot(pgd_gaps_e, label="PGD")
    # plt.plot(pgd_gaps_g, label="PGD-LS")
    # plt.plot(pgd_gaps_h, label="PGD-LS-ACC")
    # plt.plot(pgd_gaps_j, label="PGD-LS-RFISTA")
    # plt.plot(pgd_gaps_l, label="PGD-LS-DFISTA")
    # plt.plot(pgd_gaps_m, label="PGD-LS-VFISTA")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Sub-optimality")
    plt.legend()
    plt.show()