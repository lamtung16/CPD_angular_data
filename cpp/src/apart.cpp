#include <iostream>
#include "apart.h"


// constant value pi
const double pi = 3.14159265358979323846;


// squared geodesic (arclength) distance between two angular vectors
double d2(const arma::rowvec& a, const arma::rowvec& b) {
    arma::rowvec diff = arma::abs(a - b);
    arma::rowvec min_diff = arma::min(diff, 2 * pi - diff);
    return arma::accu(arma::square(min_diff));
}


// circular mean for each segment
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


// initial centroids, getting angular mean for each segment
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


// function to track the changepoints from path vector
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


std::vector<int> apart(const arma::mat &signal, const double pen, const int n_states) {
    int T = signal.n_rows;                                          // signal length
    arma::mat centroids = init_centroids(signal, n_states);         // init centroids
    arma::mat V = arma::zeros(T + 1, n_states);                     // sum of cost matrix
    arma::imat tau = arma::ones<arma::imat>(T + 1, n_states) * -1;  // last change location
    std::vector<int> path_vec(T, -1);                               // best last change location

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
            V(t, k) += d2(centroids.row(k), signal.row(t - 1));
            if (V(t, k) < V(t, best_k)) {
                best_k = k;
            }
        }
        best_prev = V(t, best_k);
        path_vec[t - 1] = tau(t, best_k);
    }

    // Track changepoints
    std::vector<int> changepoints = track_back(path_vec);

    // print to test
    std::cout << "Input signal:\n" << signal.t() << "\n";
    std::cout << "penalty: " << pen << "\n\n";
    std::cout << "n_states: " << n_states << "\n\n";
    std::cout << "Initialized centroids:\n" << centroids << "\n";
    std::cout << "Sum of Cost matrix:\n" << V << "\n";
    std::cout << "Detected changepoints: [";
    for (int cp : changepoints) {
        std::cout << cp << ", ";
    }
    std::cout << "]";

    return changepoints;
}


int main() {
    // Example data: 10 samples, 1 features
    arma::mat signal = arma::mat({2, 2, 0.5, 2*pi-0.5, 3.5, 2.5}).t();

    int n_states = 3;
    double pen = 0.1; // penalty for changing state

    std::vector<int> changes = apart(signal, pen, n_states);

    return 0;
}