# HW6 submission

## AI-use declaration

I used **Google Gemini** to verify my hand-calculated Bellman update equations for Part A, structure the simulation harness in `part_b.py` and the value iteration solver in `part_c.py`, and format the markdown tables in this report. All code was written, executed, and verified locally, and all numbers in this submission come directly from my own runs.

## Setup and reproduction

- **Dependency versions:**
  - Python: `3.14.7`
  - `gymnasium`: `1.4.0`
  - `numpy`: `2.5.5`

- **Installation commands:**
  ```powershell
  py -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install gymnasium numpy

## Part A — Paper calculations

https://github.com/iasinskiy/hw6-ai-iasin/blob/main/part_a_1.jpg
https://github.com/iasinskiy/hw6-ai-iasin/blob/main/part_a_2.jpg

## Part B — FrozenLake

### Fixed policy, written before the experiments

For `FrozenLake-v1` (`map_name="4x4"`, actions: `0=Left, 1=Down, 2=Right, 3=Up`), the following deterministic policy was fixed prior to running experiments:

- `0 (S)`: `Down (1)` — move down the safe left column, avoiding hole `5`.
- `1 (F)`: `Right (2)` — move along the top row toward safe column 2 if displaced here.
- `2 (F)`: `Down (1)` — descend down the safe column 2 between holes `5` and `7`.
- `3 (F)`: `Left (0)` — move away from the dead-end above hole `7`.
- `4 (F)`: `Down (1)` — continue down the safe left column, avoiding hole `5`.
- `6 (F)`: `Down (1)` — continue down column 2 between holes `5` and `7`.
- `8 (F)`: `Right (2)` — turn right before hole `12`.
- `9 (F)`: `Down (1)` — drop to the bottom safe row (`13`), avoiding hole `5` above.
- `10 (F)`: `Down (1)` — drop to `14`, avoiding hole `11` to the right.
- `13 (F)`: `Right (2)` — move right along the bottom row toward `14`.
- `14 (F)`: `Right (2)` — step directly into Goal `15`.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | random | 11 | 0 | 50 | 0 | 0.0% | 7.98 |
| False | random | 22 | 0 | 50 | 0 | 0.0% | 6.42 |
| False | random | 33 | 1 | 49 | 0 | 2.0% | 6.66 |
| False | fixed | 11 | 50 | 0 | 0 | 100.0% | 6.00 |
| False | fixed | 22 | 50 | 0 | 0 | 100.0% | 6.00 |
| False | fixed | 33 | 50 | 0 | 0 | 100.0% | 6.00 |
| True | random | 11 | 0 | 50 | 0 | 0.0% | 7.50 |
| True | random | 22 | 1 | 49 | 0 | 2.0% | 6.92 |
| True | random | 33 | 0 | 50 | 0 | 0.0% | 6.64 |
| True | fixed | 11 | 0 | 50 | 0 | 0.0% | 4.22 |
| True | fixed | 22 | 2 | 48 | 0 | 4.0% | 5.12 |
| True | fixed | 33 | 1 | 49 | 0 | 2.0% | 5.08 |

- **Saved episode rows & traces:**
  - All 600 episode rows: [outputs/part_b_episodes.csv](outputs/part_b_episodes.csv)
  - Deterministic sample text trace: [outputs/part_b_sample_episode.txt](outputs/part_b_sample_episode.txt)
  - Stochastic (`is_slippery=True`, fixed) successful trace: [outputs/part_b_slip_success_trace.txt](outputs/part_b_slip_success_trace.txt)
  - Stochastic (`is_slippery=True`, fixed) failed trace: [outputs/part_b_slip_fail_trace.txt](outputs/part_b_slip_fail_trace.txt)
- **What slipping changed:** Under `is_slippery=False`, the fixed policy succeeds **100.0% (50/50)** in exactly `6.00` steps. Enabling `is_slippery=True` makes each action move perpendicularly with probability $2/3$ ($1/3$ left, $1/3$ right). Because the safe corridor borders holes `5`, `11`, and `12`, perpendicular slips repeatedly push the agent into holes, reducing fixed-policy success to **0.0%–4.0%** (`0–2 / 50`) and cutting mean steps to `4.22–5.12`.
- **Variation across batches & why one episode is insufficient:** Deterministic runs showed zero variation across batches `11, 22, 33`. Under `is_slippery=True`, batch outcomes varied slightly (`0/50`, `2/50`, `1/50` goals; `0/50`, `1/50`, `0/50` for random), illustrating sampling variance when success probabilities are small. A single successful episode can occur purely by chance and provides no reliable estimate of expected return or failure rate. No time-limit truncations (100 steps) occurred because the dense holes terminate wandering agents early.
- **Why these runs do not train a policy:** No value estimates, Q-tables, or policy parameters are updated from observed rewards. The mapping from states to actions remains static across all episodes; this is pure Monte Carlo policy evaluation, not learning.

## Part C — Custom solver

### Compare code with paper

- **$V_1$ from code:**
  ```text
  [[-1. -1. 10.  0.]
   [-1.  0. -1.  0.]
   [-1. -1. -1. -1.]]

- **Requested $V_2$ values from code:**
  - $V_2(1,2) = 8.00$
  - $V_2(1,3) = 10.00$
  - $V_2(2,3) = 5.21$
- **Discrepancies:** None. The solver's $V_1$ table and all three $V_2$ values match the hand calculations from Part A to the exact decimal place.

### Values, policies and convergence

- **Full solver output log:** [outputs/part_c_results.txt](outputs/part_c_results.txt)
- **Tie-breaking rule:** During policy extraction, actions are evaluated in fixed priority order `["Up", "Right", "Down", "Left"]`, updating the chosen action only when $Q(s,a) > Q_{\text{best}} + 10^{-12}$.
- **Original Part A Map:** Converged in **12 iterations** (final max change $\Delta = 2.17 \times 10^{-7} < 10^{-6}$).
  ```text
  Value Map:                      Policy Map:
  |  4.200|  8.000| 10.000|  G(0) |   |  →  |  →  |  →  |  G  |
  |  2.780|   #   |  5.824|  H(0) |   |  ↑  |  #  |  ↑  |  H  |
  |  1.536|  2.818|  4.242|  2.818|   |  →  |  →  |  ↑  |  ←  |

  Extended Map (D = Sand -2 at (3,2), W = Shallow Water -5 at (3,4)): Converged in 11 iterations (final max change $\Delta = 2.41 \times 10^{-7} < 10^{-6}$).

  Value Map:                      Policy Map:
|  4.200|  8.000| 10.000|  G(0) |   |  →  |  →  |  →  |  G  |
|  2.780|   #   |  5.824|  H(0) |   |  ↑  |  #  |  ↑  |  H  |
|  1.502|  2.818|  4.242|  2.818|   |  ↑  |  →  |  ↑  |  ←  |

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| Original Part A Map | 11 | 47 | 3 | 0 | 5.04 | 2.230 |
| Original Part A Map | 22 | 44 | 6 | 0 | 4.96 | 1.442 |
| Original Part A Map | 33 | 42 | 8 | 0 | 4.88 | 0.970 |
| Extended Map (with D and W) | 11 | 50 | 0 | 0 | 5.00 | 1.502 |
| Extended Map (with D and W) | 22 | 50 | 0 | 0 | 5.00 | 1.502 |
| Extended Map (with D and W) | 33 | 50 | 0 | 0 | 5.00 | 1.502 |

### Explain your results

- **Which values/actions changed with sand and water:** Replacing `(3,2)` with Sand `D` (`-2`) reduced the value of Start `S` at `(3,1)` from `1.536` to `1.502` and flipped its optimal action from `→` (Right) to `↑` (Up). Replacing `(3,4)` with Shallow Water `W` (`-5`) had zero effect on any state's value or policy because optimal actions never enter `(3,4)` (from `(3,4)`, the agent moves `←` to `(3,3)`).
- **Where cost and risk mattered:** On the Original Map, the lower route through `(3,2)` (`-1`) has expected return $-1 + 0.9(2.818) = 1.536$, slightly outperforming the deterministic upper route through Mud `M` (`-3`) at $-1 + 0.9(2.780) = 1.502$, despite exposing the agent to slip risk on Ice `I` `(2,3)` (causing 3–8 hole failures per 50 episodes). On the Extended Map, entering Sand `D` (`-2`) drops the lower route's value to $-2 + 0.9(2.818) = 0.536$, making the deterministic upper path through Mud `M` strictly better (`1.502`) and yielding `50/50` safe completions. On Ice `I` `(2,3)`, the optimal action remains `Up` (`↑`, value `5.824`) on both maps because the $80\%$ chance of reaching `(1,3)` (`+10` on the next step) outweighs the $10\%$ risk of slipping `Right` into Hole `H` (`-10`).
- **How simulations relate to computed values:** Value iteration computes exact infinite-horizon expected returns ($t \to \infty$), whereas simulations stop at 100 steps. With $\gamma = 0.9$, any reward beyond step 100 is weighted by at most $0.9^{100} \approx 2.65 \times 10^{-5}$, so finite-horizon truncation bias is negligible. On the Extended Map's deterministic optimal path, every episode takes 5 steps with return $-1 + 0.9(-1) + 0.9^2(-3) + 0.9^3(-1) + 0.9^4(10) = 1.502$, matching $V^*(3,1) = 1.502$ identically. On the Original Map, stochastic slips on ice cause batch averages (`2.230`, `1.442`, `0.970`) to fluctuate around the theoretical expectation $V^*(3,1) = 1.536$ (overall simulation mean `1.547`).

### Rule checks

Saved in [outputs/part_c_checks.txt](outputs/part_c_checks.txt) and verified via assertions before solving:

```text
=== SANITY & RULE VERIFICATION CHECKS ===
[PASS] Wall & boundary stay-in-place: wall->(1, 0) (rew -1.0), edge->(0, 1) (rew -3.0)
[PASS] Ice transition probabilities sum to 1.0 for all 4 actions.
[PASS] Floor rewards verified: Mud=-3, Sand=-2, Shallow Water=-5.
[PASS] Zero rewards and self-loop after Goal/Hole termination.
