# Changepoint Detection for Angular Data
## Setup
- $Y$ be a angular signal defined over time steps $ t = 1, 2, \ldots, T $ where $Y_t \in [0, 2\pi)$
- $ \Theta = \{\theta_1, \theta_2, \ldots, \theta_K\} $ be a finite set of angular means where $\theta_k \in [0, 2\pi)$
- $ d^2(\gamma, \psi) = \sum_{j=1}^D d^2(\gamma_j, \psi_j)$ where $ d(\theta, \theta') = \min(|\theta-\theta'|, 2\pi-|\theta-\theta'|)$.

## Angular Segment Loss via Cumulative Summation
Below are steps of computing the **segment loss** of $Y_{[t_1:t_2]}$ which is $$\min_{\theta \in \Theta} \sum_{t=t_1}^{t_2} d^2(Y_t, \theta)$$

### Cost Matrix
Define the costs matrix $C \in \R^{T \times K}$:
$$
C[t, k] = d^2(Y_t, \theta_k)
\quad \text{for} \quad t = 1, \ldots, T; \quad k = 1, \ldots, K
$$

### Cumulative Sum Matrix
Define the cumulative sum matrix $ C^* \in \R^{(T+1) \times K}$ by column-wise summation of $C$:
$$
C^*[0, k] = 0 \quad \forall k \\
C^*[t, k] = \sum_{s=1}^{t} C[s, k] \quad \text{for } t = 1, \ldots, T
$$


### Segment Loss Computation

The loss of a segment $Y_{[t_1:t_2]}$ with respect to parameter $ \theta_k $ is:
$\sum_{s = t_1}^{t_2} d^2(Y_s, \theta_k)$

Using the cumulative sum matrix, this can be computed in constant time as:
$C^*[t_2, k] - C^*[t_1 - 1, k]$

So the loss of a segment $Y_{[t_1:t_2]}$ is:
$$
\min_{\theta \in \Theta} \sum_{t=t_1}^{t_2} d^2(Y_t, \theta) = \min_{k} {\big(C^*[t_2, k] - C^*[t_1 - 1, k] \big)}
$$