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


# get changpoints and signal_mean by optimal partitiniong
def opart(signal, Theta, lda):
    cumsum_costs = get_cumsum(signal, Theta)                    # get cumsum_costs matrix
    
    # Get tau_star (list of best last changepoints)
    C = np.zeros(len(signal) + 1)                               # best cost for each segment from 0 to t
    C[0] = -lda
    tau_star = np.zeros(len(signal) + 1, dtype=int)             # initiate tau_star
    for t in range(1, len(signal) + 1):
        V = compute_V(C[:t], lda, L(np.arange(t), t, cumsum_costs))
        tau_star[t] = np.argmin(V)
        C[t] = V[tau_star[t]]

    chpnts = trace_back(tau_star[1:])                           # get set of changepoints
    signal_mean = get_signal_mean(Theta, cumsum_costs, chpnts)  # get signal means given set of changepoints
    chpnts = chpnts[1:-1] - 1
    return chpnts, signal_mean