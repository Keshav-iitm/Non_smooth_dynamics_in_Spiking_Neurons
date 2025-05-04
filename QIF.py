import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='QIF neuron with analytical solution')
parser.add_argument('--I_ext', type=float, default=1, help='External current')
parser.add_argument('--V_th', type=float, default=10, help='Threshold voltage')
parser.add_argument('--V_reset', type=float, default=-1, help='Reset voltage')
args = parser.parse_args()

# QIF analytical solution function
def qif_analytical(T, T0, V0, I):
    if I <= 0:
        return V0 * np.exp(-(T-T0)) + I * (1 - np.exp(-(T-T0)))
    else:
        return np.sqrt(I) * np.tan(np.sqrt(I)*(T-T0) + np.arctan(V0/np.sqrt(I))) #VR and Tn in paper are coded as V0 and T0.

# For a single spike - calculate time to reach threshold
if args.I_ext > 0:
    T_spike = (np.arctan(args.V_th/np.sqrt(args.I_ext)) - 
               np.arctan(args.V_reset/np.sqrt(args.I_ext)))/np.sqrt(args.I_ext)
else:
    T_spike = 10  # Default for I≤0 case

# For multiple spikes - calculate spikes over longer interval
T_max = 15
num_spikes = int(T_max / T_spike) + 1

# Generate plot data points 
plot_T = []
plot_V = []

# Current time and voltage
T_current = 0
V_current = args.V_reset

# Generate data for multiple spikes
for spike in range(num_spikes):
    # Time points for this spike cycle
    T_points = np.linspace(0, T_spike*0.9999999, 100)  # 0.999 to avoid singularity at tan
    
    # Calculate voltage trajectory for this spike
    V_points = qif_analytical(T_points, 0, V_current, args.I_ext)
    
    # Add to plot arrays (with offset for current time)
    plot_T.extend(T_points + T_current)
    plot_V.extend(V_points)
    
    # Add the threshold point and reset
    plot_T.append(T_current + T_spike)
    plot_V.append(args.V_th)
    
    plot_T.append(T_current + T_spike)
    plot_V.append(args.V_reset)
    
    # Update current time
    T_current += T_spike
    
    # Reset voltage for next spike
    V_current = args.V_reset

# Plot
plt.figure(figsize=(10, 6))
plt.plot(plot_T, plot_V, 'red', linewidth=2)
plt.axhline(args.V_th, color='blue', linestyle='--', linewidth=2, label='Upper Threshold')
plt.axhline(args.V_reset, color='green', linestyle='--', linewidth=2, label='Lower Threshold')

plt.xlabel('Non-dimensional Time T = t/τ', fontsize=12)
plt.ylabel('Membrane Potential (V)', fontsize=12)
plt.title('Voltage trace of QIF', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(args.V_reset-0.1, args.V_th+0.1)
plt.tight_layout()
plt.show()
