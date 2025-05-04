import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse
from tqdm import tqdm

# AIF model with spike adaptation
def aif_model(t, y, I, omega, beta):
    v, a = y  
    f_v = abs(v)
    dvdt = f_v - a + I
    dadt = omega * (beta * v - a)
    return [dvdt, dadt]

def simulate_aif(I, omega, beta, vth, vR, k, t_end=50, dt=0.01):
    t = 0
    v, a = vR, 0.0
    t_list, v_list, a_list = [0], [vR], [0.0]
    spike_count = 0

    while t < t_end:
        sol = solve_ivp(aif_model, [t, t_end], [v, a],args=(I, omega, beta),max_step=dt, rtol=1e-6, atol=1e-8)
        
        v_sol = sol.y[0]
        a_sol = sol.y[1]
        t_sol = sol.t
        
        spike_idx = np.where(v_sol >= vth)[0]
        
        if len(spike_idx) == 0:
            t_list.extend(t_sol[1:])
            v_list.extend(v_sol[1:])
            a_list.extend(a_sol[1:])
            break
            
        first_spike = spike_idx[0]
        
        # Record up to spike
        t_list.extend(t_sol[1:first_spike+1])
        v_list.extend(v_sol[1:first_spike+1])
        a_list.extend(a_sol[1:first_spike+1])
        
        # Add explicit reset points
        t_spike = t_sol[first_spike]
        a_after_spike = a_sol[first_spike] + k
        
        # Spike peak
        t_list.append(t_spike)
        v_list.append(vth)
        a_list.append(a_sol[first_spike])
        
        # Reset point
        t_list.append(t_spike)
        v_list.append(vR)
        a_list.append(a_after_spike)
        
        # Update state
        t = t_spike
        v = vR
        a = a_after_spike
        spike_count += 1

    return np.array(t_list), np.array(v_list), np.array(a_list), spike_count

def plot_spike_vs_k():
    ks = np.linspace(0, 1.0, 30)
    spikes = []
    for k in tqdm(ks, desc="Simulating spike counts"):
        _, _, _, spike_count = simulate_aif(I=args.I_ext, omega=args.omega, beta=0, vth=args.V_th, vR=args.V_reset, k=k, t_end=100)
        spikes.append(spike_count)

    plt.figure(figsize=(7, 4))
    plt.plot(ks, spikes, 'o-', color='purple')
    plt.xlabel('Adaptation Strength $k$')
    plt.ylabel('Spike Count (100 ms)')
    plt.title('Spike Count vs $k$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('spike_vs_k.png', dpi=300)
    plt.show()

def plot_AIF():
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    # Top Left: Tonic firing
    t, v, a, _ = simulate_aif(I=0.1, omega=1/3, beta=0, vth=1, vR=0.2, k=0.25, t_end=100)
    axs[0,0].plot(t, v, 'k')
    axs[0,0].set_title('Tonic Firing v vs t at ($\\omega=1/3$)')
    axs[0,0].set_ylabel('v')
    axs[0,0].set_xlabel('t')
    axs[0,0].set_xlim(0, 50)
    axs[0,0].set_ylim(0, 1.1)

    # Top Right: Burst firing 
    t, v, a, _ = simulate_aif(I=0.1, omega=1/75, beta=0, vth=1, vR=0.2, k=2/75, t_end=500, dt=0.001)
    axs[0,1].plot(t, v, 'k', drawstyle='steps-post')
    axs[0,1].set_title('Burst Firing v vs t at ($\\omega=1/75$)')
    axs[0,1].set_ylabel('v')
    axs[0,1].set_xlabel('t')
    axs[0,1].set_xlim(0, 500)
    axs[0,1].set_ylim(0, 1.1)

    # Bottom Left: Phase plane tonic
    t, v, a, _ = simulate_aif(I=0.1, omega=1/3, beta=0, vth=1, vR=0.2, k=0.25, t_end=200)
    mask = t > 20
    axs[1,0].plot(v[mask], a[mask], 'g')  
    axs[1,0].axhline(0, color='r')
    axs[1,0].plot(np.linspace(-1,1,100), np.abs(np.linspace(-1,1,100)) + 0.1, 'r')
    axs[1,0].axvline(1, color='b', ls='--')
    axs[1,0].axvline(0.2, color='b', ls='--')
    axs[1,0].set_title('Tonic Firing phase portrait at ($\\omega=1/3$)')
    axs[1,0].set_xlabel('v')
    axs[1,0].set_ylabel('a')

    # Bottom Right: Phase plane burst
    t, v, a, _ = simulate_aif(I=0.1, omega=1/75, beta=0, vth=1, vR=0.2, k=2/75, t_end=800, dt=0.001)
    mask = (t > 200) & (t < 750)
    axs[1,1].plot(v[mask], a[mask], 'g')  
    axs[1,1].axhline(0, color='r')
    axs[1,1].plot(np.linspace(-1,1,100), np.abs(np.linspace(-1,1,100)) + 0.1, 'r')
    axs[1,1].axvline(1, color='b', ls='--')
    axs[1,1].axvline(0.2, color='b', ls='--')
    axs[1,1].set_title('Burst Firing phase portrait at ($\\omega=1/75$)')
    axs[1,1].set_xlabel('v')
    axs[1,1].set_ylabel('a')

    plt.tight_layout()
    plt.savefig('AIF.png', dpi=300)
    plt.show()

# Poincare map for tonic firing
def plot_poincare_tonic():
    params = {
        'I': 0.1, 'omega': 1/3, 'beta': 0,
        'vth': 1, 'vR': 0.2, 'k': 0.25,
        't_end': 1000, 'dt': 0.01
    }
    
    # Simulate and extract post-reset points
    _, v, a, _ = simulate_aif(**params)
    post_reset_mask = np.isclose(v, params['vR'], atol=1e-3)
    post_reset_a = a[post_reset_mask][50:]  # Remove first 50 transients
    
    # Create Poincaré map
    plt.figure(figsize=(8, 6))
    plt.plot(post_reset_a[:-1], post_reset_a[1:], 'o', markersize=5, alpha=0.7)
    plt.plot([0, 0.3], [0, 0.3], 'k--', lw=1)  # Identity line
    plt.xlabel('$a_n$ (Current adaptation)')
    plt.ylabel('$a_{n+1}$ (Next adaptation)')
    plt.title('Tonic Firing Poincare Map ($\\omega=1/3$)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('poincare_tonic.png', dpi=300)
    plt.show()

# Poincare map for burst firing  
def plot_poincare_burst():
    params = {
        'I': 0.1, 'omega': 1/75, 'beta': 0,
        'vth': 1, 'vR': 0.2, 'k': 2/75,
        't_end': 2000, 'dt': 0.01
    }
    
    # Simulate and extract post-reset points
    _, v, a, _ = simulate_aif(**params)
    post_reset_mask = np.isclose(v, params['vR'], atol=1e-3)
    post_reset_a = a[post_reset_mask][50:]  # Remove first 50 transients
    
    # Create Poincare map
    plt.figure(figsize=(8, 6))
    plt.plot(post_reset_a[:-1], post_reset_a[1:], 'o', markersize=5, alpha=0.7)
    plt.plot([0, 0.3], [0, 0.3], 'k--', lw=1)  # Identity line
    plt.xlabel('$a_n$ (Current adaptation)')
    plt.ylabel('$a_{n+1}$ (Next adaptation)')
    plt.title('Burst Firing Poincare Map ($\\omega=1/75$)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('poincare_burst.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AIF Model Analysis')
    parser.add_argument('--plot', action='store_true', help='Generate 4-panel figure')
    parser.add_argument('--spike_vs_k', action='store_true', help='Spike count vs adaptation strength')
    parser.add_argument('--poincare_tonic', action='store_true', help='Generate tonic Poincare map')
    parser.add_argument('--poincare_burst', action='store_true', help='Generate burst Poincare map')
    parser.add_argument('--I_ext', type=float, default=0.1, help='External current')
    parser.add_argument('--V_th', type=float, default=1, help='Threshold voltage')
    parser.add_argument('--V_reset', type=float, default=0.2, help='Reset voltage')
    parser.add_argument('--omega', type=float, default=1/75, help='omega')
    args = parser.parse_args()

    if args.plot: plot_AIF()
    if args.spike_vs_k: plot_spike_vs_k()
    if args.poincare_tonic: plot_poincare_tonic()
    if args.poincare_burst: plot_poincare_burst()

 