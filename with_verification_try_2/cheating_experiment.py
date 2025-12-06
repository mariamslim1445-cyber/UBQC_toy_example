import os
import time
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adjust these if your filenames are different
ALICE_SCRIPT = "client_Alice_fixed.py"
BOB_SCRIPT = "server_Bob_fixed.py"

HOST = None  # both scripts already agree on host/port internally
PORT = 5050  # kept only as a reminder


def start_bob(cheat_flip_prob):
    """Start Bob's server as a subprocess with a given CHEAT_FLIP_PROB."""
    env = os.environ.copy()
    env["CHEAT_FLIP_PROB"] = str(cheat_flip_prob)
    # quiet Bob's output; change DEVNULL to PIPE if you want to see logs
    proc = subprocess.Popen(
        [sys.executable, BOB_SCRIPT],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    # give server a moment to bind the socket
    time.sleep(1.0)
    return proc


def run_one_protocol(num_traps, s_reps):
    """Run Alice once (which internally does s_reps repetitions).

    Returns (accepted_flag, full_output_text).
    """
    env = os.environ.copy()
    env["NUM_TRAPS"] = str(num_traps)
    env["S_REPS"] = str(s_reps)

    result = subprocess.run(
        [sys.executable, ALICE_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
    )
    output = result.stdout + "\n" + result.stderr

    accepted = "Alice accepts" in output
    return accepted, output


def main():
    # grid of parameters
    trap_values = [0, 1, 2, 3, 4, 5]          # number of traps (T)
    cheat_probs = [0.05, 0.1, 0.2, 0.3, 0.4]  # CHEAT_FLIP_PROB = p_dev
    s_reps = 1                                 # repetitions per protocol (as in your client)
    runs_per_point = 20                        # full protocols per (T, p_dev) -> tune this

    empirical = np.zeros((len(trap_values), len(cheat_probs)))
    theory = np.zeros_like(empirical)

    for j, p_dev in enumerate(cheat_probs):
        print(f"=== Starting experiments for cheat probability p_dev={p_dev} ===")
        bob_proc = start_bob(p_dev)
        try:
            for i, T in enumerate(trap_values):
                print(f"  -> T = {T} traps")
                accepts = 0
                for run in range(runs_per_point):
                    ok, out = run_one_protocol(T, s_reps)
                    if ok:
                        accepts += 1
                empirical[i, j] = accepts / runs_per_point

                # simple theory: one repetition passes with (1 - p_dev)^T,
                # s_reps independent repetitions -> (1 - p_dev)^(T * s_reps)
                theory[i, j] = (1.0 - p_dev) ** (T)

                print(f"     empirical P_undetected ≈ {empirical[i,j]:.3f}, theory ≈ {theory[i,j]:.3f}")
        finally:
            bob_proc.terminate()
            try:
                bob_proc.wait(timeout=5)
            except Exception:
                pass

    # make heatmaps
    T_grid = np.array(trap_values)
    P_grid = np.array(cheat_probs)
    extent = [P_grid[0], P_grid[-1], T_grid[0], T_grid[-1]]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(empirical, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(label="Empirical P(cheating undetected)")
    plt.xlabel("cheat probability p_dev (CHEAT_FLIP_PROB)")
    plt.ylabel("Number of traps T")
    plt.title("Empirical undetected cheating")

    plt.subplot(1, 2, 2)
    plt.imshow(theory, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(label="Theoretical P(cheating undetected)")
    plt.xlabel("cheat probability p_dev (CHEAT_FLIP_PROB)")
    plt.ylabel("Number of traps T")
    plt.title("Theoretical undetected cheating")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
