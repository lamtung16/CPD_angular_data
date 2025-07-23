# Changepoint Detection for Angular Data

Detect changepoints in multidimensional angular signals using the Approximate Partitioning (APART) algorithm.

The program is written in C++, with a user-selectable interface in either Python or R:
- [Python](#python)
- [R](#r)

---

## Python

### Installation

### Example

## R

### Installation
This repository has been confirmed to work with the following configuration:
- Environment:
  - R (v4.5.1)
  - Rtools (v4.5)
- Required Packages in R:
  - Rcpp (v1.1.0)
  - RcppArmadillo (v14.6.0-1)

Note: other configuration may also be compatible.

### Example
- Code:
```R
library(Rcpp)                       # load Rcpp library
sourceCpp("r/apart_interface.cpp")  # Load and compile the APART C++ code

signal <- matrix(c(1.0, 0.1, 1.2, 6.2, 0.8, 6.1, 3.0, 2.0, 3.2, 2.1,
                   2.8, 2.0, 3.1, 1.9, 1.0, 3.1, 1.1, 3.0, 0.9, 3.2),
                 ncol = 2, byrow = TRUE)

penalty <- 0.1
num_states <- 5
change_points <- apart_rcpp(signal, penalty, num_states)
print(change_points)
```

- Output:
```bash
[1] 2 6
```


## Developer zone

**Which Armadillo version is used?**

Armadillo is added to the project's dependencies as a git submodule in `cpp/ext/armadillo`.
The submodule is linked to the [Armadillo repo](https://gitlab.com/conradsnicta/armadillo-code) and is pinned to version 14.4.x

**How is the cpp code binded to Python?**

We use [CArma](https://github.com/RUrlus/carma) and [pybind11](https://github.com/pybind/pybind11).
CArma is added to the project's dependencies as a git submodule in `python/ext/carma`.
The code is pinned to the `stable` branch.
