## Scorecard (8 ranks, 600 steps, fault at midpoint)

### op-duration jitter 3%, straggler lag 30% of op

| fault | detected | attributed (all causes) | top cause correct | false stragglers |
|---|---|---|---|---|
| clean | 30/30 | n/a | n/a | 0 |
| thermal | 30/30 | 30/30 | 30/30 | 0 |
| power | 30/30 | 30/30 | 30/30 | 0 |
| ecc | 30/30 | 30/30 | 30/30 | 0 |
| pcie | 30/30 | 30/30 | 30/30 | 0 |
| nvlink | 30/30 | 30/30 | 30/30 | 0 |
| host_stall | 30/30 | 30/30 | 30/30 | 0 |
| contention | 30/30 | 30/30 | 30/30 | 0 |
| contention_hidden | 30/30 | 30/30 | 30/30 | 0 |
| thermal+ecc | 30/30 | 30/30 | 30/30 | 0 |

### op-duration jitter 10%, straggler lag 30% of op

| fault | detected | attributed (all causes) | top cause correct | false stragglers |
|---|---|---|---|---|
| clean | 30/30 | n/a | n/a | 0 |
| thermal | 30/30 | 30/30 | 30/30 | 0 |
| power | 30/30 | 30/30 | 30/30 | 0 |
| ecc | 30/30 | 30/30 | 30/30 | 0 |
| pcie | 30/30 | 30/30 | 30/30 | 0 |
| nvlink | 30/30 | 30/30 | 30/30 | 0 |
| host_stall | 30/30 | 30/30 | 30/30 | 0 |
| contention | 30/30 | 30/30 | 30/30 | 0 |
| contention_hidden | 30/30 | 30/30 | 30/30 | 0 |
| thermal+ecc | 30/30 | 30/30 | 30/30 | 0 |

### Sensitivity: thermal fault, jitter 3%, varying straggler lag

| lag (% of op) | detected |
|---|---|
| 2% | 0/30 |
| 5% | 30/30 |
| 10% | 30/30 |
| 20% | 30/30 |
| 30% | 30/30 |
