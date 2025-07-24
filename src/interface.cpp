#include "apart.h"
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::IntegerVector apart_rcpp(const arma::mat& signal, double pen, int nStates) {
    std::vector<int> chpnts = apart(signal, pen, nStates);
    return Rcpp::wrap(chpnts);
}