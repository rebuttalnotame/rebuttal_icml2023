import numpy as np
from sklearn.model_selection import train_test_split
from tau_path import run_tau_path
from lamda_path import run_selection_path
import util_interactions
import intervals
import pdb


def CP(X, y, s_beta, tau1, tau2, lmd, sig_level=0.1, alpha=0.001):
    n_samples, n_features = X.shape
    cond = alpha * np.identity(X.shape[1])
    H = X.T.dot(X) + cond
    H_inv = np.linalg.pinv(H)
    C = np.eye(n_samples) - X.dot(H_inv.dot(X.T))

    A = C.dot(list(y) + [0]) + lmd * X.dot(H_inv.dot(s_beta))
    B = C[:, -1]
    conf_pred, hat_y, p_values = compute_p_values(A, B, tau1, tau2, sig_level)
    return conf_pred, hat_y, p_values


def compute_p_values(A, B, tau1, tau2, sig_level):
    n_samples = A.shape[0]
    negative_B = np.where(B < 0)[0]
    A[negative_B] *= -1
    B[negative_B] *= -1
    S, U, V = [], [], []

    for i in range(n_samples):

        if B[i] != B[-1]:
            tmp_u_i = (A[i] - A[-1]) / (B[-1] - B[i])
            tmp_v_i = -(A[i] + A[-1]) / (B[-1] + B[i])
            u_i, v_i = np.sort([tmp_u_i, tmp_v_i])
            U += [u_i]
            V += [v_i]

        elif B[i] != 0:
            tmp_uv = -0.5 * (A[i] + A[-1]) / B[i]

            U += [tmp_uv]
            V += [tmp_uv]

        if B[-1] > B[i]:
            S += [intervals.closed(U[-1], V[-1])]

        elif B[-1] < B[i]:
            intvl_u = intervals.openclosed(-np.inf, U[-1])
            intvl_v = intervals.closedopen(V[-1], np.inf)
            S += [intvl_u.union(intvl_v)]

        elif B[-1] == B[i] and B[i] > 0 and A[-1] < A[i]:
            S += [intervals.closedopen(U[-1], np.inf)]

        elif B[-1] == B[i] and B[i] > 0 and A[-1] > A[i]:
            S += [intervals.openclosed(-np.inf, U[-1])]

        elif B[-1] == B[i] and B[i] == 0 and abs(A[-1]) <= abs(A[i]):
            S += [intervals.open(-np.inf, np.inf)]

        elif B[-1] == B[i] and B[i] == 0 and abs(A[-1]) > abs(A[i]):
            S += [intervals.empty()]

        elif B[-1] == B[i] and A[-1] == A[i]:
            S += [intervals.open(-np.inf, np.inf)]

        else:
            print("boom !!!")

    hat_y = np.sort([-np.inf] + U + V + [np.inf])
    size = hat_y.shape[0]
    conf_pred = intervals.empty()
    p_values = np.zeros(size)

    for i in range(size - 1):

        n_pvalue_i = 0.
        intvl_i = intervals.closed(hat_y[i], hat_y[i + 1])

        for j in range(n_samples):
            n_pvalue_i += intvl_i in S[j]
            # print("S[{}]: {}\n".format(j, S[j]))
        # pdb.set_trace()

        p_values[i] = n_pvalue_i / n_samples

        if p_values[i] > sig_level:
            conf_pred = conf_pred.union(intvl_i)

    # pdb.set_trace()
    tau_intv = intervals.closed(tau1, tau2)
    conf_pred = conf_pred.intersection(tau_intv)

    return conf_pred, hat_y, p_values


def union_of_intervals(z_interval):
    new_z_interval = []
    for each_interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_z_interval[-1][1]
            if abs(sub) < 0.01:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)

    return new_z_interval[0]


def pred(X, y, x_te, lmd, alpha, max_depth):
    list_lmd, list_beta, list_aset, _, _, _ = run_selection_path(X, y, lmd_tgt=lmd, alpha=alpha, max_depth=max_depth,
                                                                 verbose=0)
    if len(list_aset) != 0:
        A = list_aset[-1]
        beta = list_beta[-1]
        # print("A: {}, beta: {}\n".format(A, beta))
        # pdb.set_trace()
        xA_te = util_interactions.construct_XA(x_te.T, A)
        y_pred = xA_te.dot(beta).item()
        return y_pred

    else:
        print("Debug: Inside 'pred' (compute_CP): Active set is empty !\n")
        pdb.set_trace()
        return None


def cp_homotopy(X_tr, y_tr, x_te, lamda, alpha, max_depth, sig_level, verbose=0):
    y_pred = pred(X_tr, y_tr, x_te, lamda, alpha, max_depth)
    y_min, y_max = np.min(y_tr), np.max(y_tr)
    y_delta = 0.5 * (y_max - y_min)
    tau_min = y_min - y_delta
    tau_max = y_max + y_delta

    list_tau, list_beta, list_aset, list_nu, nc, ts = run_tau_path(X_tr, y_tr, x_te, lamda, alpha, max_depth,
                                                                   y_min=tau_min, y_max=tau_max, verbose=verbose)

    X = np.vstack((X_tr, x_te.T))

    CP_set = []
    i = 0

    while i < len(list_tau) - 1:
        tau1, tau2 = list_tau[i], list_tau[i + 1]

        aset, beta, nu = list_aset[i], list_beta[i], list_nu[i]
        XA = util_interactions.construct_XA(X, aset)
        y = np.hstack((y_tr, tau1))

        v = XA.T.dot(y - XA.dot(beta)) / lamda
        s_beta = np.sign(v)

        conf_pred, hat_y, p_values = CP(XA, y_tr, s_beta, tau1, tau2, lamda, sig_level=sig_level)

        if not conf_pred.is_empty():
            CP_set += [[conf_pred.lower, conf_pred.upper]]

        i += 1

    CP_set_new = union_of_intervals(CP_set)

    return CP_set_new, y_pred, nc, ts


def cp_split(X_tr, y_tr, x_te, lmd, alpha, max_depth, sig_level):

    list_lmd, list_beta, list_aset = [], [], []
    itr, n_itr = 0, 20
    while itr < n_itr:
        itr += 1
        X_train, X_valid, y_train, y_valid = train_test_split(X_tr, y_tr, test_size=0.5)

        # Fit with half the data
        list_lmd, list_beta, list_aset, _, _, _ = run_selection_path(X_train, y_train, lmd_tgt=lmd, alpha=alpha,
                                                                     max_depth=max_depth, verbose=0)
        if len(list_aset) != 0:
            break

    if len(list_aset) == 0:
        print("Data splitting tried 20 random samples....")
        return None

    A = list_aset[-1]
    beta = list_beta[-1]

    # Inference with remaining half data ('X_valid', 'y_valid')
    XA_valid = util_interactions.construct_XA(X_valid, A)
    y_valid_pred = XA_valid.dot(beta)
    res = np.abs(y_valid - y_valid_pred)

    xA_te = util_interactions.construct_XA(x_te.T, A).T
    y_te_pred = xA_te.T.dot(beta).item()

    # Ranking on the calibration set
    sorted_residual = np.sort(res)
    index = int((X_tr.shape[0] / 2 + 1) * (1 - sig_level))

    L = sorted_residual[index]
    CP_set_Split = [y_te_pred - L, y_te_pred + L]
    # print("CP_set_Split (recomputed): {}\n".format(CP_set_Split))

    return CP_set_Split, y_te_pred
