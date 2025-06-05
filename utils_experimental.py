import numpy as np
from numba import njit, prange

pi = np.pi
np.random.seed(123)


# get loss value of the segment with an array of start
@njit(parallel=True)
def L(start, end, cumsum_costs):
    n_thetas = cumsum_costs.shape[1]
    L_vals = np.empty(start.shape[0], dtype=cumsum_costs.dtype)
    for i in prange(start.shape[0]):
        row_min = cumsum_costs[end, 0] - cumsum_costs[start[i], 0]
        for j in range(1, n_thetas):
            val = cumsum_costs[end, j] - cumsum_costs[start[i], j]
            if val < row_min:
                row_min = val
        L_vals[i] = row_min
    return L_vals


@njit(parallel=True)
def compute_V(C_sub, lda, L_vals):
    V = np.empty(len(C_sub))
    for i in prange(len(C_sub)):
        V[i] = C_sub[i] + lda + L_vals[i]
    return V


# get cumsum matrix of signal given Theta
def get_cumsum(signal, Theta):
    costs = np.empty((len(signal), len(Theta)))
    for i, theta in enumerate(Theta):
        costs[:, i] = np.sum(np.square(np.minimum(np.abs(signal - theta), 2*pi - np.abs(signal - theta))), axis=1)
    return np.cumsum(np.vstack([np.zeros((1, costs.shape[1])), costs]), axis=0)


# get the list of changepoints from vector tau_star
def trace_back(tau_star):
    tau = tau_star[-1]
    chpnts = np.array([len(tau_star)], dtype=int)
    while tau > 0:
        chpnts = np.append(tau, chpnts)
        tau = tau_star[tau-1]
    return np.append(0, chpnts)


# get signal mean given
def get_signal_mean(Theta, cumsum_costs, chpnts):
    mean = np.zeros(shape=(cumsum_costs.shape[0]-1, Theta.shape[1]))
    for i in range(len(chpnts) - 1):
        start = chpnts[i]
        end = chpnts[i + 1]
        value = Theta[np.argmin(cumsum_costs[end] - cumsum_costs[start])]
        mean[start:end] = [value] * (end - start)
    return mean


def prune_candidates(R, C, t, lda, cumsum_costs):
    s_arr = np.array(R)
    total_costs = C[s_arr] + L(s_arr + 1, t, cumsum_costs) + lda
    pruned = s_arr[total_costs <= C[t] + 1e-10]
    return np.append(pruned, t)


# get changpoints and signal_mean by optimal partitiniong
def opart(signal, Theta, lda):
    cumsum_costs = get_cumsum(signal, Theta)

    C = np.zeros(len(signal) + 1)
    C[0] = -lda
    tau_star = np.zeros(len(signal) + 1, dtype=int)

    candidates = np.array([0])  # candidate changepoints (initially only 0)

    for t in range(1, len(signal) + 1):
        V = compute_V(C[candidates], lda, L(candidates, t, cumsum_costs))
        best_idx = np.argmin(V)
        tau_star[t] = candidates[best_idx]
        C[t] = V[best_idx]

        # Prune candidates
        # R_new = []
        # for s in R:
        #     if C[s] + L(np.array([s+1]), t, cumsum_costs)[0] + lda <= C[t] + 1e-10:
        #         R_new.append(s)
        # R_new.append(t)
        # R = R_new
        total_costs = C[candidates] + L(candidates + 1, t, cumsum_costs) + lda
        pruned = candidates[total_costs <= C[t] + 1e-10]
        candidates = np.append(pruned, t)
        print(candidates)
        # candidates = prune_candidates(candidates, C, t, lda, cumsum_costs)

    chpnts = trace_back(tau_star[1:])
    signal_mean = get_signal_mean(Theta, cumsum_costs, chpnts)
    chpnts = chpnts[1:-1] - 1
    return chpnts, signal_mean