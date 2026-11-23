import os
import csv
import numpy as np
import gymnasium as gym

os.makedirs("outputs", exist_ok=True)

# Фиксированная политика для 4x4 FrozenLake (0:Left, 1:Down, 2:Right, 3:Up)
# Карта:
#  0(S)  1(F)  2(F)  3(F)
#  4(F)  5(H)  6(F)  7(H)
#  8(F)  9(F) 10(F) 11(H)
# 12(H) 13(F) 14(F) 15(G)
FIXED_POLICY = {
    0: 1,   # Down -> 4
    1: 2,   # Right -> 2
    2: 1,   # Down -> 6
    3: 0,   # Left -> 2
    4: 1,   # Down -> 8
    6: 1,   # Down -> 10
    8: 2,   # Right -> 9
    9: 1,   # Down -> 13 (или 2 -> 10, оба ведут к цели без ям при slippery=False)
    10: 1,  # Down -> 14
    13: 2,  # Right -> 14
    14: 2,  # Right -> 15 (Goal)
}

TERMINAL_HOLES = {5, 7, 11, 12}
GOAL_STATE = 15

def run_episode(env, policy_type, rng, max_steps=100, record_trace=False):
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    trace_lines = []

    if record_trace:
        trace_lines.append(f"Initial state: {obs}\n{env.render()}")

    while steps < max_steps:
        if policy_type == "random":
            action = int(rng.integers(0, 4))
        else:
            action = FIXED_POLICY[obs]

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        if record_trace:
            trace_lines.append(f"\nStep {steps}, Action: {action}, New State: {obs}, Reward: {reward}\n{env.render()}")

        if terminated or truncated:
            break

    if obs == GOAL_STATE:
        outcome = "goal"
    elif obs in TERMINAL_HOLES:
        outcome = "hole"
    else:
        outcome = "time limit"

    return steps, total_reward, outcome, "\n".join(trace_lines)

def main():
    batch_seeds = [11, 22, 33]
    conditions = [
        (False, "random"),
        (False, "fixed"),
        (True, "random"),
        (True, "fixed"),
    ]

    rows = []
    summary = []

    # Сохраняем один текстовый прогон для отчёта
    demo_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="ansi")
    demo_env.reset(seed=11000)
    _, _, _, demo_trace = run_episode(demo_env, "fixed", np.random.default_rng(11000), record_trace=True)
    with open("outputs/part_b_sample_episode.txt", "w", encoding="utf-8") as f:
        f.write(demo_trace)
    demo_env.close()

    # Сохраняем один успешный и один провальный прогон при is_slippery=True для отчёта
    slip_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="ansi")
    saved_succ, saved_fail = False, False
    for s in range(11000, 11050):
        slip_env.reset(seed=s)
        _, _, out, tr = run_episode(slip_env, "fixed", np.random.default_rng(s), record_trace=True)
        if out == "goal" and not saved_succ:
            with open("outputs/part_b_slip_success_trace.txt", "w", encoding="utf-8") as f:
                f.write(tr)
            saved_succ = True
        elif out == "hole" and not saved_fail:
            with open("outputs/part_b_slip_fail_trace.txt", "w", encoding="utf-8") as f:
                f.write(tr)
            saved_fail = True
        if saved_succ and saved_fail:
            break
    slip_env.close()

    # Основные 4 условия x 3 батча x 50 эпизодов
    for is_slippery, policy_type in conditions:
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)
        cond_name = f"slippery={is_slippery}_{policy_type}"

        for b_seed in batch_seeds:
            goals = holes = timeouts = 0
            step_list = []

            for i in range(50):
                ep_seed = 1000 * b_seed + i
                env.reset(seed=ep_seed)
                rng = np.random.default_rng(ep_seed)

                steps, total_reward, outcome, _ = run_episode(env, policy_type, rng)
                rows.append([cond_name, b_seed, ep_seed, policy_type, steps, total_reward, outcome])

                step_list.append(steps)
                if outcome == "goal":
                    goals += 1
                elif outcome == "hole":
                    holes += 1
                else:
                    timeouts += 1

            summary.append({
                "Slippery?": str(is_slippery),
                "Policy": policy_type,
                "Batch seed": b_seed,
                "Goals /50": goals,
                "Holes /50": holes,
                "Time limits /50": timeouts,
                "Success %": f"{(goals / 50) * 100:.1f}%",
                "Mean steps": f"{np.mean(step_list):.2f}"
            })
        env.close()

    with open("outputs/part_b_episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "batch_seed", "episode_seed", "policy", "steps", "total_reward", "outcome"])
        writer.writerows(rows)

    print("| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |")
    print("|---|---|---|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['Slippery?']} | {s['Policy']} | {s['Batch seed']} | {s['Goals /50']} | {s['Holes /50']} | {s['Time limits /50']} | {s['Success %']} | {s['Mean steps']} |")

if __name__ == "__main__":
    main()