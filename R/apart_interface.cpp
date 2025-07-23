#include "../cpp/src/apart.h"
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::IntegerVector apart_rcpp(const arma::mat& signal, double pen, int n_states) {
    std::vector<int> chpnts = apart(signal, pen, n_states);
    return Rcpp::wrap(chpnts);
}