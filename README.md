# Changepoint Detection for Angular Data

Detect changepoints in multidimensional angular signals using the Approximate Partitioning (APART) algorithm.

The program is written in C++, with a user-selectable interface in either Python or R:
- [Python](#python)
- [R](#r)

---

## Python

### Installation

First, clone the current repository.

```bash
git clone --recurse-submodules https://github.com/lamtung16/apartruptures.git
cd apartruptures
```

Then run the following command.

```bash
python -m pip install python/.
```

(Don't forget the trailing `/.`.)

### Example

```Python
import numpy as np

from ruptures_apart import apart

signal = np.random.random(size=(100, 3))
bkps = apart(signal=signal, pen=0.3, nStates=10)
print(bkps)
```



## R

### Installation

```bash
remotes::install_github("lamtung16/apartruptures")
```

### Example
```R
library(apartruptures)

signal <- matrix(c(1.0, 0.1, 1.2, 6.2, 0.8, 6.1, 3.0, 2.0, 3.2, 2.1,
                   2.8, 2.0, 3.1, 1.9, 1.0, 3.1, 1.1, 3.0, 0.9, 3.2),
                 ncol = 2, byrow = TRUE)

penalty <- 0.1
num_states <- 5

bkps <- apart_rcpp(signal, penalty, num_states)
print(bkps)
```


## Developer zone

**Which Armadillo version is used?**

Armadillo is added to the project's dependencies as a git submodule in `cpp/ext/armadillo`.
The submodule is linked to the [Armadillo repo](https://gitlab.com/conradsnicta/armadillo-code) and is pinned to version 14.4.x

**How is the cpp code binded to Python?**

We use [CArma](https://github.com/RUrlus/carma) and [pybind11](https://github.com/pybind/pybind11).
CArma is added to the project's dependencies as a git submodule in `python/ext/carma`.
The code is pinned to the `stable` branch.
