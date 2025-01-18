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
grav, v = 9.81, 10
dt, tmax = 0.005, 2

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
    plt.xlabel("Horizontal Position")
    plt.ylabel("Vertical Position")
    plt.legend()
    plt.savefig(f"sus_{text}_{str(t)}.png")
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

# Animate response
for i in range(len(t)):
    plot_susp([z0[i], z1[i], z2[i], t[i]], road_x, road_z, x[i], 3,c,k,"bef")



# Define optimization functions
def suspension_system(t, y, m, c, k, road_z_interp):
    return [y[1], -(c/m) * y[1] - (k/m) * (y[0] - road_z_interp(t))]

def objective_function(params):
    k, c = params
    sol = solve_ivp(suspension_system, (0, tmax), [0, 0], args=(ms, c, k, lambda t: np.interp(v * t, road_x, road_z)), t_eval=t)
    return np.max(np.abs(sol.y[0]))

# Optimization
initial_guess = [2000, 50]
result = minimize(objective_function, initial_guess, bounds=[(1000, 3000), (10, 100)])
optimal_k, optimal_c = result.x

# Simulate with optimized parameters
sol_optimal = solve_ivp(suspension_system, (0, tmax), [0, 0], args=(ms, optimal_c, optimal_k, lambda t: np.interp(v * t, road_x, road_z)), t_eval=t)

# Plot displacement vs time after optimization
plt.figure()
plt.plot(t, z2, label=f"Chassis Displacement(Before Optimization) \n c = {c}, k = {k}",color="red")
plt.plot(t, sol_optimal.y[0], label="Chassis Displacement (After Optimization)", color="green")
plt.title(f"Chassis Displacement Over Time \n Optimal k: {optimal_k:.2f}, Optimal c: {optimal_c:.2f}")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.legend()
plt.grid()
plt.show()


# State-space matrices
Aqcar_o = np.array([[0, 1, 0, 0],
                  [-kus / mus, -optimal_c / mus, optimal_k / mus, optimal_c / mus],
                  [0, -1, 0, 1],
                  [0, optimal_c / ms, optimal_k / ms, -optimal_c  / ms]])
Bqcar_o = np.array([-1, 0, 0, 0]).reshape(-1, 1)
Cqcar_o = np.eye(4)
Dqcar_o = np.zeros((4, 1))

# Create state-space model
qcar_o = StateSpace(Aqcar_o, Bqcar_o, Cqcar_o, Dqcar_o)

# Load road profile data
road_x_o = np.linspace(0, 99.99, 10000)
road_z_o = road_z
dx_o = road_x_o[1] - road_x_o[0]
dt2_o = dx_o / v
z0dot_o = np.gradient(road_z_o, dx_o)

# Simulation setup
x0_o = np.zeros(4)
t_o = np.arange(0, tmax, dt)
x_o = v * t_o
u_o = np.interp(x, road_x_o, z0dot_o)

# Simulate quarter car model
_, y_o, _ = lsim(qcar_o, U=u_o, T=t_o, X0=x0_o)
z0_o = np.interp(x_o, road_x_o, road_z_o) * 3
z1_o = z0_o + y_o[:, 0]
z2_o = z1_o + y_o[:, 2]

# Animate response after optimization
for i in range(len(t_o)):
    plot_susp([z0_o[i], z1_o[i], z2_o[i], t_o[i]], road_x_o, road_z_o, x_o[i], 3,optimal_c,optimal_k,"aft")