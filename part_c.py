import os
import numpy as np

os.makedirs("outputs", exist_ok=True)

# Конфигурация типов полов отделена от солвера (расширяемо без переписывания алгоритма)
FLOOR_RULES = {
    "S": {"reward": -1.0, "slip": False, "terminal": False, "walkable": True},
    ".": {"reward": -1.0, "slip": False, "terminal": False, "walkable": True},
    "M": {"reward": -3.0, "slip": False, "terminal": False, "walkable": True},
    "I": {"reward": -1.0, "slip": True,  "terminal": False, "walkable": True},
    "D": {"reward": -2.0, "slip": False, "terminal": False, "walkable": True},
    "W": {"reward": -5.0, "slip": False, "terminal": False, "walkable": True},
    "G": {"reward": 10.0, "slip": False, "terminal": True,  "walkable": True},
    "H": {"reward": -10.0,"slip": False, "terminal": True,  "walkable": True},
    "#": {"reward": 0.0,  "slip": False, "terminal": False, "walkable": False},
}

PART_A_MAP = [
    [".", "M", ".", "G"],
    [".", "#", "I", "H"],
    ["S", ".", ".", "."],
]

EXTENDED_MAP = [
    [".", "M", ".", "G"],
    [".", "#", "I", "H"],
    ["S", "D", ".", "W"],
]

# Порядок действий (используется для детерминированного разрешения ничьих - tie-breaking)
ACTIONS = ["Up", "Right", "Down", "Left"]
DELTAS = {
    "Up": (-1, 0),
    "Right": (0, 1),
    "Down": (1, 0),
    "Left": (0, -1),
}
PERPENDICULAR = {
    "Up": ["Left", "Right"],
    "Down": ["Left", "Right"],
    "Left": ["Up", "Down"],
    "Right": ["Up", "Down"],
}
ARROWS = {"Up": "↑", "Right": "→", "Down": "↓", "Left": "←"}


class GridWorldMDP:
    def __init__(self, grid, floor_rules, gamma=0.9):
        self.grid = grid
        self.rules = floor_rules
        self.gamma = gamma
        self.rows = len(grid)
        self.cols = len(grid[0])

    def tile(self, r, c):
        return self.grid[r][c]

    def is_terminal(self, r, c):
        return self.rules[self.tile(r, c)]["terminal"]

    def is_walkable(self, r, c):
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        return self.rules[self.tile(r, c)]["walkable"]

    def step_deterministic(self, r, c, action):
        if self.is_terminal(r, c):
            return (r, c), 0.0
        dr, dc = DELTAS[action]
        nr, nc = r + dr, c + dc
        if not self.is_walkable(nr, nc):
            nr, nc = r, c
        reward = self.rules[self.tile(nr, nc)]["reward"]
        return (nr, nc), reward

    def get_transitions(self, r, c, action):
        """Возвращает список (prob, (nr, nc), reward)."""
        if self.is_terminal(r, c):
            return [(1.0, (r, c), 0.0)]

        if not self.rules[self.tile(r, c)]["slip"]:
            next_s, rew = self.step_deterministic(r, c, action)
            return [(1.0, next_s, rew)]

        # Скольжение на льду: 0.8 прямо, по 0.1 перпендикулярно
        outcomes = {}
        dirs = [(action, 0.8)] + [(p, 0.1) for p in PERPENDICULAR[action]]
        for d, prob in dirs:
            next_s, rew = self.step_deterministic(r, c, d)
            key = (next_s, rew)
            outcomes[key] = outcomes.get(key, 0.0) + prob

        return [(prob, s, rew) for (s, rew), prob in outcomes.items()]

    def value_iteration(self, tol=1e-6, max_iter=10000):
        V = np.zeros((self.rows, self.cols))
        v1_snapshot = None
        v2_snapshot = None

        for it in range(1, max_iter + 1):
            V_new = np.zeros_like(V)
            max_diff = 0.0

            for r in range(self.rows):
                for c in range(self.cols):
                    t = self.tile(r, c)
                    if not self.rules[t]["walkable"] or self.rules[t]["terminal"]:
                        V_new[r, c] = 0.0
                        continue

                    q_vals = []
                    for a in ACTIONS:
                        q = sum(
                            prob * (rew + self.gamma * V[nr, nc])
                            for prob, (nr, nc), rew in self.get_transitions(r, c, a)
                        )
                        q_vals.append(q)
                    V_new[r, c] = max(q_vals)
                    max_diff = max(max_diff, abs(V_new[r, c] - V[r, c]))

            V = V_new
            if it == 1:
                v1_snapshot = V.copy()
            elif it == 2:
                v2_snapshot = V.copy()

            if max_diff < tol:
                policy = self.extract_policy(V)
                return V, policy, it, max_diff, True, v1_snapshot, v2_snapshot

        policy = self.extract_policy(V)
        return V, policy, max_iter, max_diff, False, v1_snapshot, v2_snapshot

    def extract_policy(self, V):
        policy = {}
        for r in range(self.rows):
            for c in range(self.cols):
                t = self.tile(r, c)
                if not self.rules[t]["walkable"] or self.rules[t]["terminal"]:
                    continue
                best_a = None
                best_q = -1e18
                # Tie-breaking: при равных Q (с точностью 1e-12) выбирается первое действие по порядку ACTIONS
                for a in ACTIONS:
                    q = sum(
                        prob * (rew + self.gamma * V[nr, nc])
                        for prob, (nr, nc), rew in self.get_transitions(r, c, a)
                    )
                    if q > best_q + 1e-12:
                        best_q = q
                        best_a = a
                policy[(r, c)] = best_a
        return policy


def run_verification_checks():
    """Обязательные проверки правил перед доверием солверу."""
    mdp = GridWorldMDP(EXTENDED_MAP, FLOOR_RULES)
    lines = ["=== SANITY & RULE VERIFICATION CHECKS ==="]

    # 1. Границы и стены
    s_wall, r_wall = mdp.step_deterministic(1, 0, "Right")  # удар о '#' в (1,1)
    assert s_wall == (1, 0) and r_wall == -1.0
    s_edge, r_edge = mdp.step_deterministic(0, 1, "Up")     # удар о верхний край из 'M'
    assert s_edge == (0, 1) and r_edge == -3.0
    lines.append(f"[PASS] Wall & boundary stay-in-place: wall->{s_wall} (rew {r_wall}), edge->{s_edge} (rew {r_edge})")

    # 2. Сумма вероятностей на льду
    for a in ACTIONS:
        trans = mdp.get_transitions(1, 2, a)
        prob_sum = sum(p for p, _, _ in trans)
        assert abs(prob_sum - 1.0) < 1e-9
    lines.append("[PASS] Ice transition probabilities sum to 1.0 for all 4 actions.")

    # 3. Награды грязи, песка и воды
    assert mdp.step_deterministic(0, 0, "Right")[1] == -3.0  # M
    assert mdp.step_deterministic(2, 0, "Right")[1] == -2.0  # D (sand)
    assert mdp.step_deterministic(2, 2, "Right")[1] == -5.0  # W (water)
    lines.append("[PASS] Floor rewards verified: Mud=-3, Sand=-2, Shallow Water=-5.")

    # 4. Отсутствие наград после терминации
    assert mdp.step_deterministic(0, 3, "Left") == ((0, 3), 0.0)  # Goal
    assert mdp.step_deterministic(1, 3, "Down") == ((1, 3), 0.0)  # Hole
    lines.append("[PASS] Zero rewards and self-loop after Goal/Hole termination.")

    report = "\n".join(lines)
    with open("outputs/part_c_checks.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report + "\n")


def format_maps(mdp, V, policy):
    val_lines = []
    pol_lines = []
    for r in range(mdp.rows):
        v_row = []
        p_row = []
        for c in range(mdp.cols):
            t = mdp.tile(r, c)
            if t == "#":
                v_row.append("   #   ")
                p_row.append("  #  ")
            elif mdp.is_terminal(r, c):
                v_row.append(f"  {t}(0) ")
                p_row.append(f"  {t}  ")
            else:
                v_row.append(f"{V[r, c]:7.3f}")
                p_row.append(f"  {ARROWS[policy[(r, c)]]}  ")
        val_lines.append("|" + "|".join(v_row) + "|")
        pol_lines.append("|" + "|".join(p_row) + "|")
    return "\n".join(val_lines), "\n".join(pol_lines)


def simulate_policy(mdp, policy, max_steps=100):
    batch_seeds = [11, 22, 33]
    results = []

    for b_seed in batch_seeds:
        goals = holes = timeouts = 0
        steps_all = []
        disc_returns = []

        for i in range(50):
            ep_seed = 1000 * b_seed + i
            rng = np.random.default_rng(ep_seed)
            r, c = 2, 0  # Старт S в (3,1) -> индекс (2,0)
            steps = 0
            disc_ret = 0.0

            while steps < max_steps:
                if mdp.is_terminal(r, c):
                    break
                action = policy[(r, c)]
                trans = mdp.get_transitions(r, c, action)
                probs = [p for p, _, _ in trans]
                idx = int(rng.choice(len(trans), p=probs))
                _, (r, c), rew = trans[idx]

                disc_ret += (mdp.gamma ** steps) * rew
                steps += 1

                if mdp.is_terminal(r, c):
                    break

            t = mdp.tile(r, c)
            if t == "G":
                goals += 1
            elif t == "H":
                holes += 1
            else:
                timeouts += 1

            steps_all.append(steps)
            disc_returns.append(disc_ret)

        results.append({
            "Batch": b_seed,
            "Goals": goals,
            "Holes": holes,
            "Timeouts": timeouts,
            "Mean Steps": f"{np.mean(steps_all):.2f}",
            "Mean Disc Return": f"{np.mean(disc_returns):.3f}"
        })
    return results


def main():
    run_verification_checks()

    out_log = []
    for name, grid in [("Part A Map", PART_A_MAP), ("Extended Map (with D and W)", EXTENDED_MAP)]:
        mdp = GridWorldMDP(grid, FLOOR_RULES, gamma=0.9)
        V, policy, iters, final_diff, conv, v1, v2 = mdp.value_iteration()

        header = f"=== {name} ==="
        print(header)
        out_log.append(header)

        if name == "Part A Map":
            print("V1 table:\n", np.round(v1, 2))
            print(f"V2 check: (1,2)={v2[0,1]:.2f}, (1,3)={v2[0,2]:.2f}, (2,3)={v2[1,2]:.2f}")
            out_log.append(f"V1:\n{np.round(v1, 2)}")
            out_log.append(f"V2 check: (1,2)={v2[0,1]:.2f}, (1,3)={v2[0,2]:.2f}, (2,3)={v2[1,2]:.2f}")

        status = f"Converged: {conv} in {iters} iterations (final delta = {final_diff:.2e})"
        print(status)
        out_log.append(status)

        v_str, p_str = format_maps(mdp, V, policy)
        print("Value Map:\n" + v_str)
        print("Policy Map:\n" + p_str)
        out_log.extend(["Value Map:\n" + v_str, "Policy Map:\n" + p_str])

        sim_res = simulate_policy(mdp, policy)
        print("Simulations (3 batches x 50 episodes):")
        for row in sim_res:
            line = f"  Batch {row['Batch']}: Goals={row['Goals']}/50, Holes={row['Holes']}/50, Timeouts={row['Timeouts']}/50, MeanSteps={row['Mean Steps']}, MeanDiscReturn={row['Mean Disc Return']}"
            print(line)
            out_log.append(line)
        print()

    with open("outputs/part_c_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_log))


if __name__ == "__main__":
    main()