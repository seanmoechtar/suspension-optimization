import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Step 1: Generate synthetic road surface data with random bumps
np.random.seed(0)
x = np.linspace(0, 100, 200)  # Increased resolution for smoother profile

# Base sine wave for the road profile
base_profile = 0.1 * np.sin(0.1 * x)

# Adding random bumps
num_bumps = 10  # Number of random bumps
bump_heights = np.random.uniform(0.05, 0.15, num_bumps)  # Random heights for bumps
bump_positions = np.random.uniform(0, 100, num_bumps)  # Random positions for bumps
bump_widths = np.random.uniform(1, 5, num_bumps)  # Random widths for bumps

# Create the bumps
bump_profile = np.zeros_like(x)
for height, position, width in zip(bump_heights, bump_positions, bump_widths):
    bump_profile += height * np.exp(-((x - position) ** 2) / (2 * width ** 2))

# Combine base profile and bumps
y = base_profile + bump_profile + np.random.normal(0, 0.02, x.shape)  # Add some noise

# Step 2: Linear Regression for Road Surface Fitting
x = x.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Step 3: Spline Interpolation for Smooth Road Profile
cs = CubicSpline(x.flatten(), y_pred)

# Step 4: Numerical Integration for Car Motion Simulation
def suspension_system(t, y, m, c, k, y_road):
    return [y[1], -(c/m) * y[1] - (k/m) * (y[0] - y_road(t))]

# Initial conditions
y0 = [0, 0]  # Initial position and velocity
m = 1000  # Mass of the car

# Step 5: Optimization of Suspension Parameters
def objective_function(params):
    k, c = params
    sol = solve_ivp(suspension_system, (0, 10), y0, args=(m, c, k, cs), t_eval=np.linspace(0, 10, 100))
    return np.max(np.abs(sol.y[0]))  # Example metric

# Initial guess for parameters
initial_guess = [2000, 50]

# Optimize
result = minimize(objective_function, initial_guess, bounds=[(1000, 3000), (10, 100)])
optimal_k, optimal_c = result.x

# Step 6: Simulate with optimal parameters
sol_optimal = solve_ivp(suspension_system, (0, 10), y0, args=(m, optimal_c, optimal_k, cs), t_eval=np.linspace(0, 10, 100))

# Step 7: Parameter Sensitivity Analysis
k_values = np.linspace(1000, 3000, 5)  # Range of spring constants
c_values = np.linspace(10, 100, 5)  # Range of damping coefficients
max_displacements = np.zeros((len(k_values), len(c_values)))

for i, k in enumerate(k_values):
    for j, c in enumerate(c_values):
        sol = solve_ivp(suspension_system, (0, 10), y0, args=(m, c, k, cs), t_eval=np.linspace(0, 10, 100))
        max_displacements[i, j] = np.max(np.abs(sol.y[0]))

# Step 8: Plotting Results
plt.figure(figsize=(12, 8))

# Plot road profile
plt.subplot(3, 1, 1)
plt.plot(x.flatten(), y_pred, label='Fitted Road Surface', color='blue')
plt.plot(x.flatten(), y, 'o', label='Measured Road Surface', color='orange', markersize=5)
plt.title('Road Surface Profile with Random Bumps')
plt.xlabel('Distance along the track (m)')
plt.ylabel('Elevation (m)')
plt.legend()

# Plot car motion
plt.subplot(3, 1, 2)
plt.plot(sol_optimal.t, sol_optimal.y[0], label='Car Position (Optimal Parameters)', color='green')
plt.title('Car Motion Simulation with Optimal Parameters')
plt.xlabel

plt.tight_layout()
plt.show()