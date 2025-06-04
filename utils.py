import numpy as np

pi = np.pi
np.random.seed(123)


# get cumsum matrix of signal given Theta
def get_cumsum(signal, Theta):
    costs = np.empty((len(signal), len(Theta)))
    for i, theta in enumerate(Theta):
        costs[:, i] = np.sum(np.square(np.minimum(np.abs(signal - theta), 2*pi - np.abs(signal - theta))), axis=1)
    return np.cumsum(np.vstack([np.zeros((1, costs.shape[1])), costs]), axis=0)


# get loss value of the segment
def L(start, end, cumsum_costs):
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
def opart(signal, Theta, lda):
    cumsum_costs = get_cumsum(signal, Theta)                    # get cumsum_costs matrix
    
    # Get tau_star (list of best last changepoints)
    C = np.zeros(len(signal) + 1)                               # best cost for each segment from 0 to t
    C[0] = -lda
    tau_star = np.zeros(len(signal) + 1, dtype=int)             # initiate tau_star
    for t in range(1, len(signal) + 1):
        V = C[:t] + lda + L(np.arange(t), t, cumsum_costs)      # calculate set V
        C[t] = np.min(V)                                        # update C_i
        tau_star[t] = np.argmin(V)                              # update tau_star

    chpnts = trace_back(tau_star[1:])                           # get set of changepoints
    signal_mean = get_signal_mean(Theta, cumsum_costs, chpnts)  # get signal means given set of changepoints
    chpnts = chpnts[1:-1] - 1
    return chpnts, signal_mean