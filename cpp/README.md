# Simple example
### Step 0: navigate the terminal to
```
cpp\src\
```

### Step 1: create a file `example.cpp` having this template
``` cpp
#include <iostream>
#include "apart.h"

int main() {
    // signal: matrix shape (T, D), this example T = 10 and D = 2
    arma::mat signal = {
        {1.0, 0.1}, {1.2, 6.2}, {0.8, 6.1}, 
        {3.0, 2.0}, {3.2, 2.1}, {2.8, 2.0}, {3.1, 1.9}, 
        {1.0, 3.1}, {1.1, 3.0}, {0.9, 3.2}
    };

    double penalty = 0.1;
    int num_states = 5;
    
    // detect chpnts: 1D vector, no fixed length, in this example the length is 2
    std::vector<int> chpnts = apart(signal, penalty, num_states);

    // Print the changepoints
    std::cout << "change points: ";
    for (int idx : chpnts) {std::cout << idx << " ";}

    return 0;
}
```

### Step 2: using the terminal to compile and run
```
g++ example.cpp apart.cpp -o ex -larmadillo
./ex
```
it will print:
```
change points: 2 6
```