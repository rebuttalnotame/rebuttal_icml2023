import numpy as np
from numpy.linalg import pinv
import util_interactions
from lamda_path import run_selection_path
from queue import Queue
import time
import sys


def stepsize_Inclusion(XA, Xj, xA_te, xj_te, y, y_te, beta, nu, lmd):
    tol = 1e-10
    W = y - XA.dot(beta)  # error
    W_te = y_te - xA_te.T.dot(beta)  # test error
    s_j = (Xj.T.dot(W) + xj_te.T.dot(W_te)) / lmd
    gamma_j = xj_te - (Xj.T.dot(XA) + xj_te.dot(xA_te.T)).dot(nu)

    if gamma_j != 0:
        d = (lmd * (np.sign(gamma_j) - s_j)) / gamma_j
        d = d.item()
        if 0 < d < tol:
            d = np.inf
    else:
        d = np.inf

    return d


def stepsize_Deletion(beta, nu):
    tol = 1e-10
    indx_nz = np.where(beta != 0)[0].tolist()  # Search over only active features
    beta_nz = beta[indx_nz]
    nu_nz = nu[indx_nz]

    step_sizes = []
    for i in range(len(indx_nz)):
        if nu_nz[i] != 0:
            tmp = - beta_nz[i] / nu_nz[i]
            if tmp > 0:  # "step-size" must be positive !
                step_sizes.append(tmp)
            else:
                step_sizes.append(np.inf)
        else:
            step_sizes.append(np.inf)

    if len(step_sizes) > 0:
        indx_min = np.argmin(step_sizes)
        d2 = step_sizes[indx_min]
        j_out = indx_nz[indx_min]
        if 0 < d2 < tol:
            d2 = np.inf
            j_out = None
    else:
        d2 = np.inf
        j_out = None
    return d2, j_out


def bound(XA, Xj, xA_te, xj_te, y, y_te, d_opt, beta, nu):
    W = y - XA.dot(beta)
    indx_pos_W = np.where(W > 0)[0]
    indx_neg_W = np.where(W < 0)[0]

    b_rho_pos = Xj[indx_pos_W].T.dot(np.abs(W[indx_pos_W]))
    b_rho_neg = Xj[indx_neg_W].T.dot(np.abs(W[indx_neg_W]))
    b_rho = max(b_rho_pos, b_rho_neg)

    theta = XA.dot(nu)
    indx_pos_theta = np.where(theta > 0)[0]
    indx_neg_theta = np.where(theta < 0)[0]

    b_theta_pos = Xj[indx_pos_theta].T.dot(np.abs(theta[indx_pos_theta]))
    b_theta_neg = Xj[indx_neg_theta].T.dot(np.abs(theta[indx_neg_theta]))
    b_theta = max(b_theta_pos, b_theta_neg)

    W_te = y_te - xA_te.T.dot(beta)  # test error
    kappa_te = np.abs(xj_te * W_te)

    eta_te = np.abs(xj_te * xA_te.T.dot(nu))

    max_bound = b_rho + kappa_te + d_opt * (xj_te + b_theta + eta_te)

    return max_bound


def check_Bound_Main_Search(XA, Xj, xA_te, xj_te, y, y_te, d_opt, beta, nu, alpha):
    W = y - XA.dot(beta)
    W_te = y_te - xA_te.T.dot(beta)  # test error
    V = XA.dot(nu)
    V_te = xA_te.T.dot(nu)

    rho0 = XA[:, 0].T.dot(W) - alpha * beta[0]
    kappa0 = xA_te[0] * W_te
    theta0 = XA[:, 0].T.dot(V) + alpha * nu[0]
    eta0 = xA_te[0] * V_te

    min_bound = np.abs(rho0) - np.abs(kappa0) - d_opt * (xA_te[0] + np.abs(theta0) + np.abs(eta0))
    max_bound = bound(XA, Xj, xA_te, xj_te, y, y_te, d_opt, beta, nu)

    check_bound_flag = False
    if max_bound > min_bound:  # i.e. if feasible solution exists
        check_bound_flag = True

    return check_bound_flag


def compute_nu(X, x_te, A, alpha=0):
    XA = util_interactions.construct_XA(X, A)
    xA_te = util_interactions.construct_XA(x_te.T, A).T
    cond = alpha * np.identity(len(A))
    C_inv = pinv(XA.T.dot(XA) + xA_te.dot(xA_te.T) + cond)
    nu = C_inv.dot(xA_te)
    return nu.flatten()


def call_lamda_path(X, y, lmd, alpha, max_depth, verbose=0):
    lmd_list, beta_list, aset_list, _, _, _ = run_selection_path(X, y, lmd_tgt=lmd, alpha=alpha,
                                                                 max_depth=max_depth, verbose=verbose)
    if len(aset_list) != 0:
        A = aset_list[-1].copy()
        beta = beta_list[-1].copy()
        return A, beta
    else:
        print("Debug: call_lamda_path!,    A is empty...... \n")


def update_list(list_tau, list_aset, list_beta, list_nu, tau_t, A, beta, nu):
    list_tau.append(tau_t)
    list_aset.append(A.copy())
    list_beta.append(beta.copy())
    list_nu.append(nu.copy())
    return list_tau, list_aset, list_beta, list_nu


class node(object):
    def __init__(self, level, pattern, depth):
        self.level = level
        self.pattern = pattern
        self.depth = depth


def explore(X, y, x_te, y_te, beta, nu, A, lmd, alpha, max_depth=None, verbose=0):
    p = X.shape[1]
    Q = Queue(maxsize=0)

    if max_depth is None:
        max_depth = p
    else:
        max_depth = max_depth
    root_pattern = [0] * p

    XA = util_interactions.construct_XA(X, A)
    xA_te = util_interactions.construct_XA(x_te.T, A).T
    d_opt = np.Inf
    j_opt = None
    root = node(-1, root_pattern, 0)
    Q.put(root)
    i = 0

    start_time = time.time()  # Note start time
    while not Q.empty():
        parent = Q.get_nowait()
        level = parent.level
        pattern = parent.pattern
        depth = parent.depth

        while level < p - 1 and depth < max_depth:
            level += 1
            pattern_child = pattern[:level] + [1] + pattern[level + 1:]
            j = np.where(np.array(pattern_child) == 1)[0].tolist()
            Xj = np.prod(X[:, j], 1)[:, np.newaxis]
            xj_te = np.prod(x_te.T[:, j], 1)[:, np.newaxis]

            if np.sum(np.vstack((Xj, xj_te))) == 0:
                continue

            if j not in A:
                dt = stepsize_Inclusion(XA, Xj, xA_te, xj_te, y, y_te, beta, nu, lmd)
                if dt > 0 and dt < d_opt:
                    d_opt = dt
                    j_opt = j

            if d_opt == np.inf:
                check_bound_flag = True
            else:
                check_bound_flag = check_Bound_Main_Search(XA, Xj, xA_te, xj_te, y, y_te, d_opt, beta, nu, alpha)

            if check_bound_flag:
                child = node(level, pattern_child, depth + 1)
                Q.put_nowait(child)
                if verbose >= 2:
                    print("pattern : {},  i = {}".format(pattern_child, i))
                i += 1

    ts = time.time() - start_time  # Note time difference
    return d_opt, j_opt, i, ts


def run_tau_path(X, y, x_te, lmd, alpha=0.0, max_depth=None, y_min=-20, y_max=20, verbose=0):

    X_aug, y_aug = np.vstack((X, x_te.T)), np.hstack((y, y_min))

    # Initialization
    list_lmd_init, list_beta_init, list_aset_init, _, _, _ = run_selection_path(X_aug, y_aug, lmd_tgt=lmd, alpha=alpha,
                                                                                max_depth=max_depth, verbose=verbose)

    if len(list_aset_init) == 0:
        return [], [], [], []

    A = list_aset_init[-1]
    beta = list_beta_init[-1]
    tau_t = y_min

    # Compute direction vector "nu"
    nu = compute_nu(X, x_te, A, alpha)

    # Update lists (Initial update)
    list_tau = [tau_t]
    list_aset = [A.copy()]
    list_beta = [beta.copy()]
    list_nu = [nu.copy()]

    node_counts = []
    tss = []
    node_traversed, ts_avg = 0, 0

    while tau_t < y_max:
        d2, j_out = stepsize_Deletion(beta, nu)  # Step-size deletion
        d1, j_in, counts, ts = explore(X, y, x_te, tau_t, beta, nu, A, lmd, alpha, max_depth, verbose)  # Step-size
        # inclusion

        node_counts.append(counts)
        tss.append(ts)

        ts_day = 86400
        if np.sum(tss) >= ts_day:
            print("The 'tau-path' did not finish in one day !!! \n")
            sys.exit()

        if d1 <= 0:
            d1 = np.inf
            j_in = None
        if d2 <= 0:
            d2 = np.inf
            j_out = None

        if d1 == np.Inf and d2 == np.Inf:
            if tau_t != y_max:
                tau_t = y_max  # capped at "y_max"
                y_aug = np.hstack((y, tau_t))
                A, beta = call_lamda_path(X_aug, y_aug, lmd, alpha, max_depth, verbose)
                list_tau, list_aset, list_beta, list_nu = update_list(
                    list_tau, list_aset, list_beta, list_nu, tau_t, A, beta, nu)  # "nu" does not change at this
                # point. # That's the whole point

            break

        d = min(d1, d2)
        tol = 1e-11
        if d < tol:
            return [], [], [], [], [], []
        else:
            # UPDATE
            beta = beta + d * nu  # Update 'beta'
            tau_t = tau_t + d  # Update "tau_t"

        if d == d1:  # Inclusion
            A.append(j_in)  # Update active set
            beta = np.append(beta, np.zeros(1))
            if verbose >= 2:
                print("Pattern Included: {}, zk = {}, A = {},  d1 = {}, d2 = {}".format(j_in, tau_t, A, d1, d2))
        else:  # Deletion
            pattern_del = A[j_out]
            del A[j_out]  # Update A
            beta = np.delete(beta, j_out)  # Update beta
            if verbose >= 2:
                print("Pattern Deleted: {}, zk = {}, A = {},  d1 = {}, d2 = {}".format(pattern_del, tau_t, A, d1, d2))

        if len(A) != 0:  # Update all lists
            if tau_t > y_max:
                tau_t = y_max  # capped at "y_max"
                y_aug = np.hstack((y, tau_t))
                A, beta = call_lamda_path(X_aug, y_aug, lmd, alpha, max_depth)  # call "lamda-path"

            # Compute direction vector "nu"
            nu = compute_nu(X, x_te, A, alpha)
            list_tau, list_aset, list_beta, list_nu = update_list(list_tau, list_aset, list_beta, list_nu, tau_t, A,
                                                                  beta, nu)
        else:
            if verbose >= 2:
                print("A is empty ! Path breaks here...., recomputing for the next break point....\n")

            list_tau, list_aset, list_beta, list_nu = update_list(list_tau, list_aset, list_beta, list_nu, tau_t, A,
                                                                  beta, nu)

            repeat = 0
            while tau_t < y_max:
                repeat += 1
                tau_t = tau_t + 1e-4
                yt = np.hstack((y, tau_t))
                lmd_list, beta_list, aset_list, _, _, _ = run_selection_path(X_aug, yt, lmd_tgt=lmd, alpha=alpha,
                                                                             max_depth=max_depth, verbose=0)
                if len(aset_list) != 0:
                    A = aset_list[-1].copy()
                    beta = beta_list[-1].copy()

                    # Compute direction vector "nu"
                    nu = compute_nu(X, x_te, A, alpha)

                    # Update lists (Reinitialization)
                    list_tau, list_aset, list_beta, list_nu = update_list(list_tau, list_aset, list_beta, list_nu,
                                                                          tau_t, A,
                                                                          beta, nu)
                    break
            else:
                break

    if verbose == -2:
        node_traversed = np.mean(node_counts)
        ts_avg = np.mean(tss)

    return list_tau, list_beta, list_aset, list_nu, node_traversed, ts_avg