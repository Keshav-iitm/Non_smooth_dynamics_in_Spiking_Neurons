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

    #_______________end of the code___________________________________________________
    
#import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# import argparse
# from tqdm import tqdm

# # Define the AIF model with spike adaptation
# def aif_model(t, y, I, omega, beta):
#     v, a = y  
#     f_v = abs(v)
#     dvdt = f_v - a + I
#     dadt = omega * (beta * v - a)
#     return [dvdt, dadt]

# # Simulate one run with resets
# def simulate_aif(I, omega, beta, vth, vR, k, t_end=50, dt=0.01, I_end=None):
#     t = 0
#     v, a = vR, 0.0
#     t_list, v_list, a_list, i_list = [], [], [], []
#     spike_count = 0

#     while t < t_end:  # Single loop only
#         sol = solve_ivp(aif_model, [t, t_end], [v, a], 
#                           args=(I, omega, beta, t_end, I_end),
#                           max_step=dt, rtol=1e-3, atol=1e-6)
#         v_sol, a_sol, t_sol = sol.y[0], sol.y[1], sol.t
        
#         # Spike handling
#         spike_idx = np.where(v_sol >= vth)[0]
#         if len(spike_idx) == 0:
#             t_list.extend(t_sol)
#             v_list.extend(v_sol)
#             a_list.extend(a_sol)
#             break

#         spike_time = t_sol[spike_idx[0]]
#         t_sol = t_sol[:spike_idx[0]+1]
#         v_sol = v_sol[:spike_idx[0]+1]
#         a_sol = a_sol[:spike_idx[0]+1]

#         t_list.extend(t_sol)
#         v_list.extend(v_sol)
#         a_list.extend(a_sol)

#         t = t_sol[-1]
#         v = vR
#         a = a_sol[-1] + k
#         spike_count += 1

#     return np.array(t_list), np.array(v_list), np.array(a_list), spike_count

# # def simulate_aif(I, omega, beta, vth, vR, k, t_end=50, dt=0.01, I_end=None):
# #     t = 0
# #     v, a = vR, 0.0
# #     t_list, v_list, a_list, i_list = [], [], [], []
# #     spike_count = 0

# #     while t < t_end:
# #         # Pass ramped current parameters if I_end is provided
# #         if I_end is not None:
# #             sol = solve_ivp(aif_model, [t, t_end], [v, a], 
# #                           args=(I, omega, beta, t_end, I_end),
# #                           max_step=dt, rtol=1e-3, atol=1e-6)
# #         else:
# #             sol = solve_ivp(aif_model, [t, t_end], [v, a], 
# #                           args=(I, omega, beta),
# #                           max_step=dt, rtol=1e-3, atol=1e-6)
                          
# #         v_sol, a_sol, t_sol = sol.y[0], sol.y[1], sol.t
        
# #         # Calculate current at each time point (for plotting)
# #         if I_end is not None:
# #             i_sol = I + (I_end - I) * (t_sol / t_end)
# #         else:
# #             i_sol = np.ones_like(t_sol) * I
# #     # t = 0
# #     # v, a = vR, 0.0
# #     # t_list, v_list, a_list = [], [], []
# #     # spike_count = 0

# #     while t < t_end:
# #         sol = solve_ivp(aif_model, [t, t_end], [v, a], args=(I, omega, beta),
# #                         max_step=dt, rtol=1e-3, atol=1e-6)
# #         v_sol, a_sol, t_sol = sol.y[0], sol.y[1], sol.t
# #         spike_idx = np.where(v_sol >= vth)[0]
# #         if len(spike_idx) == 0:
# #             t_list.extend(t_sol)
# #             v_list.extend(v_sol)
# #             a_list.extend(a_sol)
# #             break

# #         spike_time = t_sol[spike_idx[0]]
# #         t_sol = t_sol[:spike_idx[0]+1]
# #         v_sol = v_sol[:spike_idx[0]+1]
# #         a_sol = a_sol[:spike_idx[0]+1]

# #         t_list.extend(t_sol)
# #         v_list.extend(v_sol)
# #         a_list.extend(a_sol)

# #         t = t_sol[-1]
# #         v = vR
# #         a = a_sol[-1] + k
# #         spike_count += 1

# #     return np.array(t_list), np.array(v_list), np.array(a_list), np.array(i_list), spike_count

#     #     if len(spike_idx) == 0:
#     #         t_list.extend(t_sol)
#     #         v_list.extend(v_sol)
#     #         a_list.extend(a_sol)
#     #         break

#     #     spike_time = t_sol[spike_idx[0]]
#     #     t_sol = t_sol[:spike_idx[0]+1]
#     #     v_sol = v_sol[:spike_idx[0]+1]
#     #     a_sol = a_sol[:spike_idx[0]+1]

#     #     t_list.extend(t_sol)
#     #     v_list.extend(v_sol)
#     #     a_list.extend(a_sol)

#     #     t = t_sol[-1]
#     #     v = vR
#     #     a = a_sol[-1] + k
#     #     spike_count += 1

#     # return np.array(t_list), np.array(v_list), np.array(a_list), spike_count

# # New plot: spike count vs adaptation strength k
# def plot_spike_vs_k():
#     ks = np.linspace(0, 1.0, 30)  # Fewer points for speed
#     spikes = []
#     for k in tqdm(ks, desc="Simulating for different k values"):
#         _, _, _, spike_count = simulate_aif(I=0.1, omega=1/75, beta=0, vth=1, vR=0.2, k=k, t_end=100)
#         spikes.append(spike_count)

#     plt.figure(figsize=(7, 4))
#     plt.plot(ks, spikes, 'o-', color='purple')
#     plt.xlabel('Adaptation Strength $k$')
#     plt.ylabel('Spike Count (in 100 ms)')
#     plt.title('Spike Count vs Adaptation Strength ($\omega = 1/75$)')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('spike_vs_k.png', dpi=300)
#     plt.show()

# # Plotting function for original Figure 5 recreation
# def plot_figure5():
#     fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    
#     # Panel configuration
#     panels = [
#         {  # Top Left: Tonic firing
#             'params': {'I': 0.1, 'omega': 1/3, 'beta': 0, 'vth': 1, 'vR': 0.2, 'k': 0.15, 't_end': 100},
#             'mask': lambda t: t <= 50,
#             'type': 'time',
#             'title': 'Tonic Firing ($\\omega=1/3$)'
#         },
#         {  # Top Right: Burst firing
#             'params': {'I': 0.1, 'omega': 1/75, 'beta': 0, 'vth': 1, 'vR': 0.2, 'k': 2/75, 't_end': 500},
#             'mask': lambda t: t <= 150,
#             'type': 'time',
#             'title': 'Burst Firing ($\\omega=1/75$)'
#         },
#         {  # Bottom Left: Tonic phase
#             'params': {'I': 0.1, 'omega': 1/3, 'beta': 0, 'vth': 1, 'vR': 0.2, 'k': 0.15, 't_end': 500},
#             'mask': lambda t: t > 100,
#             'type': 'phase',
#             'title': 'Tonic Phase Plane'
#         },
#         {  # Bottom Right: Burst phase
#             'params': {'I': 0.1, 'omega': 1/75, 'beta': 0, 'vth': 1, 'vR': 0.2, 'k': 2/75, 't_end': 800},
#             'mask': lambda t: (t > 200) & (t < 750),
#             'type': 'phase',
#             'title': 'Burst Phase Plane'
#         }
#     ]

#     # Plot with progress bar
#     for i, panel in tqdm(enumerate(panels), total=4, desc="Generating panels"):
#         ax = axs.flat[i]
#         t, v, a, _ = simulate_aif(**panel['params'])
#         mask = panel['mask'](t)
        
#         if panel['type'] == 'time':
#             ax.plot(t[mask], v[mask], 'k')
#             ax.set_ylim(0, 1.1)
#             ax.set_ylabel('v')
#         else:
#             ax.plot(v[mask], a[mask], 'g', lw=1)
#             ax.set_xlabel('v')
#             ax.set_ylabel('a')
            
#             # Nullclines
#             v_vals = np.linspace(-1, 1, 100)
#             ax.plot(v_vals, np.abs(v_vals) + panel['params']['I'], 'r', lw=0.8)  # v-nullcline
#             ax.axhline(0, color='r', lw=0.8)  # a-nullcline
#             ax.axvline(1, color='b', ls='--', lw=0.8)  # Threshold
#             ax.axvline(0.2, color='b', ls='--', lw=0.8)  # Reset
        
#         ax.set_title(panel['title'])
#         ax.grid(True)

#     plt.tight_layout()
#     plt.savefig('figure5_final.png', dpi=300)
#     plt.show()

# # def plot_figure5():
# #     fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# #     # Top Left: Tonic firing (omega = 1/3)
# #     t, v, a, _ = simulate_aif(I=0.1, omega=1/3, beta=0, vth=1, vR=0.2, k=0.25,t_end=100)
# #     mask = t > 50
# #     axs[0, 0].plot(t, v, 'k')
# #     axs[0, 0].set_title('Tonic Firing ($\\omega=1/3$)')
# #     axs[0, 0].set_ylabel('v')
# #     axs[0, 0].set_xlim([0, max(t)])
# #     axs[0, 0].set_ylim([0, 1.1])
# #     axs[0, 0].set_xlim([0,50])

# #     # Top Right: Burst firing (omega = 1/75)
# #     t, v, a, _ = simulate_aif(I=0.1, omega=1/75, beta=0, vth=1, vR=0.2, k=2/75, t_end=500)
# #     axs[0, 1].plot(t, v, 'k')
# #     axs[0, 1].set_title('Burst Firing ($\\omega=1/75$)')
# #     axs[0, 1].set_xlim([0, max(t)])
# #     axs[0, 1].set_ylim([0, 1.1])

# #     # Bottom Left: Phase plane tonic
# #     t, v, a, _ = simulate_aif(I=0.15, omega=1/3, beta=0, vth=1, vR=0.2, k=0.25, t_end=200)
# #     mask = t > 20  # Remove first 20ms transient
# #     axs[1,0].plot(v[mask], a[mask], 'g')
# #     axs[1, 0].axhline(y=0, color='r')  # a-nullcline
# #     axs[1, 0].plot(np.linspace(-1, 1, 100), np.abs(np.linspace(-1, 1, 100)) + 0.1, 'r')  # v-nullcline
# #     axs[1, 0].axvline(x=1, color='b', linestyle='--')  # reset threshold
# #     axs[1, 0].axvline(x=0.2, color='b', linestyle='--')  # reset to vR
# #     axs[1, 0].set_xlabel('v')
# #     axs[1, 0].set_ylabel('a')

# #     # Bottom Right: Phase plane burst
# #     # Define ramp parameters first
# #     # I_start = 0.0
# #     # I_end = 0.1
# #     # t_end = 600

# #     # # Option 1: Modify your aif_model function to accept time-dependent current
# #     # def aif_model(t, y, I_start, I_end, t_end, omega, beta):
# #     #     v, a = y
# #     #     I = I_start + (I_end - I_start) * (t / t_end)  # Calculate I for current time
# #     #     f_v = abs(v)
# #     #     dvdt = f_v - a + I
# #     #     dadt = omega * (beta * v - a)
# #     #     return [dvdt, dadt]

# #     # # Then update your simulate_aif function to pass these parameters
# #     # t, v, a, _ = simulate_aif(I_start=I_start, I_end=I_end, omega=1/75, beta=0, 
# #     #                       vth=1, vR=0.2, k=2/75, t_end=t_end)

   
# #     # # t, v, a, _ = simulate_aif(I=0.1, omega=1/75, beta=0, vth=1, vR=0.2, k=2/75)
# #     t, v, a, i, _ = simulate_aif(I=0.05, I_end=0.1, omega=1/75, beta=0, vth=1, vR=0.2, k=2/75, t_end=300)
# #     mask = t > 50  # Remove transient
# #     axs[1, 1].plot(v[mask], a[mask], 'g')
# #     # axs[1, 1].plot(v, a, 'g')
# #     axs[1, 1].axhline(y=0, color='r')
# #     axs[1, 1].plot(np.linspace(-1, 1, 100), np.abs(np.linspace(-1, 1, 100)) + 0.1, 'r')
# #     axs[1, 1].axvline(x=1, color='b', linestyle='--')
# #     axs[1, 1].axvline(x=0.2, color='b', linestyle='--')
# #     axs[1, 1].set_xlabel('v')

# #     plt.tight_layout()
# #     plt.savefig('figure5_recreation.png', dpi=300)
# #     plt.show()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='AIF simulations and plots')
#     parser.add_argument('--plot', action='store_true', help='Generate the 4-panel Figure 5 recreation')
#     parser.add_argument('--spike_vs_k', action='store_true', help='Generate spike count vs adaptation strength plot')
#     args = parser.parse_args()

#     if args.plot:
#         plot_figure5()
#     if args.spike_vs_k:
#         plot_spike_vs_k()
