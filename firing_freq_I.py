import numpy as np
import matplotlib.pyplot as plt
import argparse

# PWL nonlinearity
def f(v, s):
    return np.where(v >= 0, v, -s * v)

def simulate_pwlif(I, beta, omega=1.0, s=0.35, k=0.4,
                   v_th=60, v_reset=20, t_max=2000, dt=0.001, return_trace=False):
    v = v_reset
    a = 0
    t = 0
    spike_times = []
    v_trace, t_trace = [], []

    while t < t_max:
        if return_trace:
            v_trace.append(v)
            t_trace.append(t)

        dvdt = f(v, s) - a + I
        dadt = omega * (beta * v - a)
        v += dvdt * dt
        a += dadt * dt
        t += dt

        if v >= v_th:
            spike_times.append(t)
            v = v_reset
            a += k

    ISIs = np.diff(spike_times)
    freq = 1.0 / np.mean(ISIs) if len(ISIs) > 1 else 0.0

    if return_trace:
        return freq, v_trace, t_trace, ISIs, spike_times
    else:
        return freq
# Debug single value of I for beta = 1.2
freq, v_trace, t_trace, ISIs, spikes = simulate_pwlif(
    I=6.3, beta=1.2, return_trace=True, dt=0.0005, t_max=2000
)

print("ISIs:", ISIs)
print("Approx. firing frequency:", freq)

plt.figure(figsize=(10, 4))
plt.plot(t_trace, v_trace)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane voltage v")
plt.title("Voltage trace for I = 6.3, β = 1.2")
plt.grid(True)
plt.show()

# Plotting helper
def plot_fi_curve(beta_val, ax, label, args):
    I_vals = np.linspace(-2, 20, 400)
    freqs = [simulate_pwlif(I, beta_val, omega=args.omega, s=args.s,
                            k=args.k, v_th=args.v_th, v_reset=args.v_R,
                            t_max=args.t_max, dt=args.dt) for I in I_vals]

    ax.plot(I_vals, freqs, label=label)
    ax.set_xlabel("Drive $I$")
    ax.set_ylabel("Firing Frequency $f$")
    ax.set_title(f"$\\beta = {beta_val}$")
    ax.grid(True)
    ax.legend()

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 7: F-I curve from Coombes et al.")
    parser.add_argument("--s", type=float, default=0.35, help="PWL slope for v<0")
    parser.add_argument("--k", type=float, default=0.4, help="Reset increment for a")
    parser.add_argument("--omega", type=float, default=1.0, help="Adaptation rate ω")
    parser.add_argument("--v_th", type=float, default=60, help="Threshold voltage")
    parser.add_argument("--v_R", type=float, default=20, help="Reset voltage")
    parser.add_argument("--t_max", type=float, default=1000, help="Simulation time")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    args = parser.parse_args()

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    plot_fi_curve(beta_val=1.2, ax=axs[0], label=r"$\beta=1.2$", args=args)
    plot_fi_curve(beta_val=0.9, ax=axs[1], label=r"$\beta=0.9$", args=args)

    plt.tight_layout()
    plt.show()
