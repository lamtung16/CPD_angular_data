#include "apart_r.h"


// constant value pi
const double pi = 3.14159265358979323846;


/**
 * @brief Computes the squared geodesic distance between two angular vectors.
 * 
 * @param vect_a First angular vector (in radians).
 * @param vect_b Second angular vector (in radians).
 * @return Squared geodesic distance between two vectors.
 */
double sq_geo_dist(const arma::rowvec& vect_a, const arma::rowvec& vect_b) {
    arma::rowvec diff = arma::abs(vect_a - vect_b);
    arma::rowvec min_diff = arma::min(diff, 2 * pi - diff);
    return arma::accu(arma::square(min_diff));
}


/**
 * @brief Computes the circular mean for each segment.
 *
 * This function calculates the circular mean for each dimension of the input segment. 
 * Angles are wrapped into the [0, 2π) interval.
 *
 * @param segment A matrix where each column contains angular values in radians.
 * @return A row vector containing the circular mean of each dimension.
 */
arma::rowvec circular_mean(const arma::mat& segment) {
    arma::rowvec mean_angles(segment.n_cols);
    for (size_t j = 0; j < segment.n_cols; ++j) {
        arma::vec col = segment.col(j);
        double sin_sum = arma::mean(arma::sin(col));
        double cos_sum = arma::mean(arma::cos(col));
        mean_angles(j) = std::atan2(sin_sum, cos_sum);
        if (mean_angles(j) < 0) {
            mean_angles(j) += 2*pi;
        }
    }
    return mean_angles;
}


/**
 * @brief Initializes centroids by computing the circular mean for equally divided segments of a signal.
 *
 * This function divides the input signal matrix into `n_states` contiguous, equally sized time segments
 * (along rows) and computes the circular mean of each segment (for each column).
 *
 * @param signal A matrix of angular data (in radians), where rows represent time points and columns represent dimensions.
 * @param n_states Number of segments (or states) to divide the signal into.
 * @return A matrix with each row is the circular mean of a segment.
 *
 * @throws std::invalid_argument if n_states is not positive or exceeds the number of rows in the signal.
 */
arma::mat init_centroids(const arma::mat& signal, int n_states) {
    if (n_states <= 0 || signal.n_rows < n_states) {
        throw std::invalid_argument("n_states must be positive and less than or equal to the signal length.");
    }

    arma::mat centroids(n_states, signal.n_cols);   // initiation of centroids

    int segment_size = signal.n_rows / n_states;
    int start_idx = 0;
    for (int i = 0; i < n_states; ++i) {
        int end_idx = start_idx + segment_size - 1;
        centroids.row(i) = circular_mean(signal.rows(start_idx, end_idx));
        start_idx = end_idx + 1;
    }

    return centroids;
}


/**
 * @brief Track back changepoints from a path vector (best last changepoint).
 *
 * @param path_vec A vector of best last changepoint.
 * @return A vector of changepoint indices, ordered from start to end.
 */
std::vector<int> track_back(const std::vector<int> &path_vec)
{
    std::vector<int> chpnts;
    int s = path_vec[path_vec.size() - 1];
    while (s >= 0) {
        chpnts.insert(chpnts.begin(), s);
        s = path_vec[s];
    }
    return chpnts;
}


/**
 * @brief Detects changepoints in an angular signal.
 *
 * Detects changepoints for the input angular signal using `n_states` states and a penalty term.
 *
 * @param signal Matrix of angular data (T × D), with T time points and D dimensions.
 * @param pen Penalty parameter to penalize the changepoint presence.
 * @param n_states Number of states (size of discrete means set).
 * @return Vector of detected changepoint indices.
 *
 * @throws std::invalid_argument if `n_states` is invalid (checked in `init_centroids`).
 */
std::vector<int> apart(const arma::mat &signal, const double pen, const int n_states) { //          signal shape (T, D)
    int T = signal.n_rows;                                          // signal length T
    arma::mat centroids = init_centroids(signal, n_states);         // init centroids               shape (n_states, D)
    arma::mat V = arma::zeros(T + 1, n_states);                     // sum of cost matrix           shape (T + 1, n_states)
    arma::imat tau = arma::ones<arma::imat>(T + 1, n_states) * -1;  // last change location         shape (T + 1, n_states)
    std::vector<int> path_vec(T, -1);                               // best last change location    shape (T, 1)

    double best_prev = 0.0;
    for (int t = 1; t <= T; ++t) {
        int best_k = 0;
        for (int k = 0; k < n_states; ++k) {
            V(t, k) = V(t - 1, k);
            tau(t, k) = tau(t - 1, k);
            if (best_prev + pen < V(t - 1, k)) {
                V(t, k) = best_prev + pen;
                tau(t, k) = t - 2;
            }
            V(t, k) += sq_geo_dist(centroids.row(k), signal.row(t - 1));
            if (V(t, k) < V(t, best_k)) {
                best_k = k;
            }
        }
        best_prev = V(t, best_k);
        path_vec[t - 1] = tau(t, best_k);
    }

    // Track changepoints
    std::vector<int> changepoints = track_back(path_vec);

    return changepoints;
}