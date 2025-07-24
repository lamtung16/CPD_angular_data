#ifndef APART_H
#define APART_H
#include <vector>
#include <RcppArmadillo.h>

std::vector<int> apart(const arma::mat &signal, const double penalty, const int nStates);

#endif // APART_H