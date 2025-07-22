#ifndef APART_R_H
#define APART_R_H
#include <vector>
#include <RcppArmadillo.h>

std::vector<int> apart(const arma::mat &signal, const double penalty, const int nStates);

#endif