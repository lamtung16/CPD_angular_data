import numpy as np

pi = np.pi
np.random.seed(123)


# geodesic distance
def geo_d(theta, psi):
    return np.sum(np.square(np.minimum(np.abs(psi - theta), 2*pi - np.abs(psi - theta))).reshape(-1, theta.shape[1]), axis=1)


# get cumsum matrix of signal given Theta
def get_cumsum(signal, Theta):
    costs = np.empty((len(signal), len(Theta)))
    for i, theta in enumerate(Theta):
        costs[:, i] = np.sum(np.square(np.minimum(np.abs(signal - theta), 2*pi - np.abs(signal - theta))), axis=1)
    return np.cumsum(np.vstack([np.zeros((1, costs.shape[1])), costs]), axis=0)


# get loss value of the segment
def L_segment(start, end, cumsum_costs):
    return np.min(cumsum_costs[end] - cumsum_costs[start], axis=1)


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
def pelt(signal, Theta, lda):
    cumsum_costs = get_cumsum(signal, Theta)                    # get cumsum_costs matrix
    
    # Get tau_star (list of best last changepoints)
    C = np.zeros(len(signal) + 1)                               # best cost for each segment from 0 to t
    C[0] = -lda
    tau_star = np.zeros(len(signal) + 1, dtype=int)             # initiate tau_star

    candidates = np.array([0])
    for t in range(1, len(signal) + 1):
        V = C[candidates] + lda + L_segment(candidates, t, cumsum_costs)      # calculate set V
        best_idx = np.argmin(V)
        tau_star[t] = candidates[best_idx]
        C[t] = V[best_idx]

        pruned = candidates[V - lda <= C[t] + 1e-10]
        candidates = np.append(pruned, t)

    chpnts = trace_back(tau_star[1:])                           # get set of changepoints
    signal_mean = get_signal_mean(Theta, cumsum_costs, chpnts)  # get signal means given set of changepoints
    chpnts = chpnts[1:-1] - 1
    return chpnts, signal_mean


def apart(signal, Theta, lda):
    theta_star = -1.0 * np.ones(shape=(len(signal) + 1, Theta.shape[1]))
    V = np.zeros(len(signal) + 1)
    for t in range(1, len(signal) + 1):
        V_candidates = V[t-1] + geo_d(Theta, signal[t-1]) + lda * np.any(theta_star[t - 1] != Theta, axis=1)
        best_idx = np.argmin(V_candidates)
        theta_star[t] = Theta[best_idx]
        V[t] = V_candidates[best_idx]
    chpnts = np.arange(len(signal) - 1)[np.any(theta_star[2:] != theta_star[1:-1], axis=1)]
    return chpnts, theta_star[1:]