from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
import compute_CP


def cp_split(X_tr, y_tr, X_te, model, sig_level):
    X_train, X_valid, y_train, y_valid = train_test_split(X_tr, y_tr, test_size=0.5)

    # Train with half the data ('X_train', 'y_train')
    model.fit(X_train, y_train)

    # Inference with remaining half data ('X_valid', 'y_valid')
    y_valid_pred = model.predict(X_valid)
    res = np.abs(y_valid - y_valid_pred)

    # Prediction for the 'test data'
    y_te_pred = model.predict(X_te)

    # Ranking on the calibration set
    sorted_residual = np.sort(res)
    index = int((X_tr.shape[0] / 2 + 1) * (1 - sig_level))

    L = sorted_residual[index]

    return y_te_pred, L


def comp_cp_set(y_te_pred, L):
    lb = y_te_pred - L
    ub = y_te_pred + L
    cp_set = np.array([lb.tolist(), ub.tolist()]).T.tolist()

    return cp_set


def opt_lmd_shim(dir_ms, max_depth):
    lmd_path_opt = dir_ms + 'opt_lmd_ord_' + str(max_depth) + '.npz'  # filepath to 'optimum lmd'.
    npzfile = np.load(lmd_path_opt, allow_pickle=True)
    lmd = npzfile['lmd_opt']

    return lmd


def comp_stat_shim(X_tr, y_tr, X_te, y_te, split=False, lmd=1.0, alpha=0.001, max_depth=2):
    sig_level = 0.1

    n = y_te.shape[0]
    count = 0
    L = []
    y_preds = []
    for i in range(n):
        if split:
            cp_set, y_pred_homo = compute_CP.cp_split(X_tr, y_tr, X_te[i][:, np.newaxis], lmd, alpha,
                                                      max_depth, sig_level)
        else:
            cp_set, y_pred_homo, _, _ = compute_CP.cp_homotopy(X_tr, y_tr, X_te[i][:, np.newaxis], lmd, alpha,
                                                               max_depth, sig_level)
        # pdb.set_trace()
        if cp_set[0] <= y_te[i] <= cp_set[1]:
            count += 1

        cl_homo = cp_set[1] - cp_set[0]

        L += [cl_homo]
        y_preds += [y_pred_homo]

    cov = count / n
    r2 = r2_score(y_te, y_preds)

    return cov, L, r2


def stat_shim(X_train, y_train, X_test, y_test, split=False, lmd=1.0, alpha=0.001, max_depth=2):
    n_reps = 30
    covs, Ls, r2s = [], [], []

    if split:
        for _ in range(n_reps):
            cov, L, r2 = comp_stat_shim(X_train, y_train, X_test, y_test, split, lmd=lmd, alpha=alpha,
                                        max_depth=max_depth)
            covs += [cov]
            Ls += L
            r2s += [r2]

    else:
        cov, L, r2 = comp_stat_shim(X_train, y_train, X_test, y_test, split, lmd=lmd, alpha=alpha,
                                    max_depth=max_depth)

        covs = [cov]
        Ls = L
        r2s = [r2]

    return covs, Ls, r2s


def stat(X_train, y_train, X_test, y_test, model):
    n = y_test.shape[0]

    n_reps = 30
    covs, Ls, r2s = [], [], []

    for _ in range(n_reps):  # repeat the "data splitting 'n_reps' times."

        # Compute the 'split CP'
        y_te_pred, L = cp_split(X_train, y_train, X_test, model, sig_level=0.1)
        cp_set = comp_cp_set(y_te_pred, L)

        # Compute the 'r2score'
        r2 = r2_score(y_test, y_te_pred)

        count = 0
        for i in range(n):
            if cp_set[i][0] <= y_test[i] <= cp_set[i][1]:
                count += 1

        cov = count / n

        covs += [cov]
        Ls += [2 * L]
        r2s += [r2]

    return covs, Ls, r2s