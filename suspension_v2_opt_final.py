# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lsim, StateSpace
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Simulation parameters
c, k = 100, 100
ms, mus = 325, 65
kus = 232.5e3
v = 10
dt, tmax = 0.1, 10



# Define plotting function for animation
def plot_susp(x, road_x, road_z, curr_x, umf,c,k,text):
    z0, z1, z2, t = x

    # Suspension geometry parameters
    h1, h2, h3, h4 = 0.35, 1.1, 0.2, 0.35
    w1, w2, w3, w4, w5 = 0.4, 0.5, 0.1, 0.15, 0.25
    
    # Preliminary calculations
    x0_r, x0_s = z0, h1 + z1 + h3 / 2
    x0_t, x0_b = h1 + z1 - h3 / 2, h2 + z2 - h4 / 2
    L1, L2 = x0_t - x0_r, x0_b - x0_s

    # Plot road profile
    dx = road_x[1] - road_x[0]
    xstart = max(curr_x - 0.7, 0)
    istart = np.searchsorted(road_x, xstart)
    xend = curr_x + 0.7
    iend = np.searchsorted(road_x, xend)
    zp = road_z[istart:iend] * umf
    xp = np.linspace(-0.7, 0.7, len(zp))

    plt.clf()
    plt.plot(xp, zp, 'k-', label="Road Profile")

    # Plot unsprung and sprung mass blocks
    plt.plot([0, 0], [x0_t, x0_t + h3], 'b', lw=5, label="Unsprung Mass")
    plt.plot([0, 0], [x0_b, x0_b + h4], 'r', lw=5, label="Sprung Mass")

    # Plot springs and damper (simplified for visualization)
    plt.plot([0, 0], [x0_r, x0_t], 'g--', lw=2, label="Tire Spring")
    plt.plot([0, 0], [x0_s, x0_b], 'm--', lw=2, label="Suspension Spring")
    

    plt.title(f"Quarter-Car Suspension at Time {t:.2f}s \n  c = {c}, k = {k}")
    plt.ylim(-0.1, 2.5)
    plt.xlabel("Horizontal Position")
    plt.ylabel("Vertical Position")
    plt.legend()
    plt.savefig(f"sus_{text}_{str(round(t,3))}.png")
    plt.pause(0.01)




# State-space matrices
Aqcar = np.array([[0, 1, 0, 0],
                  [-kus / mus, -c / mus, k / mus, c / mus],
                  [0, -1, 0, 1],
                  [0, c / ms, -k / ms, -c / ms]])
Bqcar = np.array([-1, 0, 0, 0]).reshape(-1, 1)
Cqcar = np.eye(4)
Dqcar = np.zeros((4, 1))

# Create state-space model
qcar = StateSpace(Aqcar, Bqcar, Cqcar, Dqcar)

# Load road profile data
road_x = np.linspace(0, 99.99, 10000)
road_z = np.random.uniform(1e-3, 1e-2, len(road_x))
dx = road_x[1] - road_x[0]
dt2 = dx / v
z0dot = np.gradient(road_z, dx)

# Simulation setup
x0 = np.zeros(4)
t = np.arange(0, tmax, dt)
x = v * t
u = np.interp(x, road_x, z0dot)

# Simulate quarter car model
_, y, _ = lsim(qcar, U=u, T=t, X0=x0)
z0 = np.interp(x, road_x, road_z) * 3
z1 = z0 + y[:, 0]
z2 = z1 + y[:, 2]

h1, h2, h3, h4 = 0.35, 1.1, 0.2, 0.35
spr_mass = h2 + z2 - h4 / 2

# Animate response
for i in range(len(t)):
    plot_susp([z0[i], z1[i], z2[i], t[i]], road_x, road_z, x[i], 3,c,k,"bef")

# Turunan numerik dz2/dt
dz2_dt = np.gradient(z2, t)

###OPTIMASI###
def optimize_suspension(initial_c, initial_k):
    from scipy.optimize import differential_evolution
    
    # First calculate the baseline mean velocity
    def calculate_mean_velocity(y):
        # Calculate velocity as derivative of displacement
        z2 = z1 + y[:, 2]
        velocity = np.gradient(z2, t)
        return np.mean(np.abs(velocity))
    
    # Get baseline velocity
    _, y_baseline, _ = lsim(qcar, U=u, T=t, X0=x0)
    baseline_velocity = calculate_mean_velocity(y_baseline)
    target_velocity = baseline_velocity * 0.3  # Target 70% reduction
    
    def simulate_response(params):
        c_test, k_test = params
        
        # Create state space matrices with test parameters
        A_test = np.array([[0, 1, 0, 0],
                          [-kus/mus, -c_test/mus, k_test/mus, c_test/mus],
                          [0, -1, 0, 1],
                          [0, c_test/ms, -k_test/ms, -c_test/ms]])
        B_test = np.array([-1, 0, 0, 0]).reshape(-1, 1)
        C_test = np.eye(4)
        D_test = np.zeros((4, 1))
        
        # Create state-space model
        sys_test = StateSpace(A_test, B_test, C_test, D_test)
        
        # Simulate
        _, y_test, _ = lsim(sys_test, U=u, T=t, X0=x0)
        
        # Calculate velocity metrics
        mean_velocity = calculate_mean_velocity(y_test)
        
        # Penalize if velocity reduction is not in desired range
        velocity_reduction = (baseline_velocity - mean_velocity) / baseline_velocity
        if velocity_reduction < 0.5 or velocity_reduction > 1.0:
            penalty = 1000
        else:
            penalty = 0
            
        # Calculate displacement metrics for stability
        z2_test = z1 + y_test[:, 2]
        mean_abs_disp = np.mean(np.abs(z2_test))
        max_disp = np.max(np.abs(z2_test))
        
        # Combined objective with emphasis on velocity target
        objective = (abs(mean_velocity - target_velocity) * 2.0 + 
                    mean_abs_disp * 0.5 + 
                    max_disp * 0.3 + 
                    penalty)
        
        return objective
    
    # Define bounds for c and k
    bounds = [(0, 2000), (0, 50000)]  # Batas menyesuiakan dengan data-data berikut : https://www.researchgate.net/figure/PARAMETER-VALUES-FOR-THE-QUARTER-CAR-MODEL-BY-12_tbl1_261051573
    
    # Use differential evolution with increased population and iterations
    result = differential_evolution(
        simulate_response,
        bounds,
        strategy='best1bin',
        maxiter=100,
        popsize=20,
        tol=1e-4,
        mutation=(0.5, 1.2),
        recombination=0.8
    )
    
    # Return optimized parameters and baseline velocity
    return result.x[0], result.x[1], baseline_velocity

# Run optimization
c_opt, k_opt, base_vel = optimize_suspension(c,k)
print(f"Optimized parameters: c = {c_opt:.2f}, k = {k_opt:.2f}")
print(f"Original mean velocity: {base_vel:.4f}")

# Create new state-space matrices with optimized parameters
Aqcar_o = np.array([[0, 1, 0, 0],
                    [-kus/mus, -c_opt/mus, k_opt/mus, c_opt/mus],
                    [0, -1, 0, 1],
                    [0, c_opt/ms, -k_opt/ms, -c_opt/ms]])
Bqcar_o = np.array([-1, 0, 0, 0]).reshape(-1, 1)
Cqcar_o = np.eye(4)
Dqcar_o = np.zeros((4, 1))

# Create optimized state-space model
qcar_o = StateSpace(Aqcar_o, Bqcar_o, Cqcar_o, Dqcar_o)

# Simulate with optimized parameters
_, y_o, _ = lsim(qcar_o, U=u, T=t, X0=x0)
z0_o = np.interp(x, road_x, road_z) * 3
z1_o = z0_o + y_o[:, 0]
z2_o = z1_o + y_o[:, 2]

# Calculate and print velocity reduction
final_velocity = np.mean(np.abs(np.gradient(z2_o, t)))
reduction_percent = ((base_vel - final_velocity) / base_vel) * 100
print(f"Optimized mean velocity: {final_velocity:.4f}")
print(f"Velocity reduction: {reduction_percent:.1f}%")




# Animate response
for i in range(len(t)):
    plot_susp([z0_o[i], z1_o[i], z2_o[i], t[i]], road_x, road_z, x[i], 3,c_opt,k_opt,"aft")


# Plot displacement dan turunan dz2/dt
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot z2
axs[0].plot(t, z2, label="z2(Displacement)", color='blue')
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("z2(m)")
axs[0].set_title(f"Displacement of Sprung Mass \n Mean = {np.mean(z2):.4f}")
axs[0].grid(True)
axs[0].legend()


# Plot dz2/dt
axs[1].plot(t, z2_o, label="z2(Displacement)", color='red')
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("z2(m)")
axs[1].set_title(f"Displacement of Sprung Mass(Optimized) \n Mean = {np.mean(z2_o):.4f} \n Reduction_percent = {reduction_percent:.1f}%")
axs[1].grid(True)
axs[1].legend()

# Ambil nilai ylim dari plot pertama
ylim_first_plot = axs[0].get_ylim()

# Set nilai ylim yang sama pada plot kedua
axs[1].set_ylim(ylim_first_plot)

plt.tight_layout()
plt.show()