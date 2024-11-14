import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
randSeed=2
threshold_distance = 0.15

# Define the unicycle model dynamics
def unicycle_dynamics(state, control, dt):
    x, y, theta = state
    v, omega = control

    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt

    return np.array([x_next, y_next, theta_next])

# Attractive potential function
def attractive_potential(pos, target, k_att):
    distance = np.linalg.norm(pos - target)
    potential = 0.5 * k_att * distance**2
    force = -k_att * (pos - target)
    return potential, force

# Repulsive potential function
def repulsive_potential(pos, obstacles, k_rep, obstacle_radius, influence_radius):
    potential = 0.0
    force = np.zeros(2)
    obstacles_in_range = 0  # Count of obstacles within influence radius
    for obstacle in obstacles:
        obs_pos = obstacle[:2]
        distance = np.linalg.norm(pos - obs_pos)
        if distance <= obstacle_radius:
            distance = obstacle_radius  # Avoid division by zero or negative distances
        if distance < influence_radius:
            obstacles_in_range += 1
            # Potential
            pot = 0.5 * k_rep * (1.0 / distance - 1.0 / influence_radius)**2
            potential += pot
            # Force
            repulsion = k_rep * (1.0 / distance - 1.0 / influence_radius) * (1.0 / distance**2) * (pos - obs_pos) / distance
            force += repulsion
    return potential, force, obstacles_in_range

# Normalize angle to [-pi, pi]
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Simulation parameters
dt = 0.1  # Time step
max_iterations = 200

# Initial state [x, y, theta]
initial_state = np.array([0.0, 0.0, 0.0])

# Target position [x, y]
target = np.array([6.0, 5.0])


# Random obstacle positions [x, y]

np.random.seed(randSeed)  # For reproducibility

obstacles = np.array([
    np.random.uniform(0, 5, 2),
    np.random.uniform(0, 5, 2),
    np.random.uniform(0, 5, 2),
    np.random.uniform(0, 5, 2)
])

# Obstacle parameters
obstacle_radius = 0.75
influence_radius = 1.5  # Radius of influence for repulsive force

# Control input limits
v_max = 1.0  # Linear velocity limit
omega_max = np.pi / 4  # Angular velocity limit

# Potential field parameters
k_att = 1.0    # Attractive potential gain
k_rep = 100.0  # Repulsive potential gain

# Initialize state history
state = initial_state.copy()
states = [state.copy()]

# Initialize lists to store metrics
path_length = 0.0
min_distance = float('inf')
total_control_effort = 0.0
control_inputs = []
distances_to_target = []
force_calculations_per_iteration = []  # Number of force calculations per iteration
obstacles_influence_count = []         # Number of obstacles influencing the robot per iteration
computation_times = []  # Time per iteration
control_smoothness = []  # To calculate control smoothness

# Create a grid to compute the potential field (for plotting)
x_min, x_max = -1, 7
y_min, y_max = -1, 7
resolution = 0.05
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

# Vectorized computation of the potential field
def compute_potential_field(x_grid, y_grid, target, obstacles, k_att, k_rep, obstacle_radius, influence_radius):
    pos_grid = np.stack((x_grid, y_grid), axis=2)  # Shape: (rows, cols, 2)

    # Attractive potential
    U_att = 0.5 * k_att * np.linalg.norm(pos_grid - target, axis=2)**2

    # Repulsive potential
    U_rep = np.zeros_like(U_att)
    for obstacle in obstacles:
        obs_pos = obstacle[:2]
        distance = np.linalg.norm(pos_grid - obs_pos, axis=2)
        # Avoid division by zero
        distance = np.maximum(distance, obstacle_radius)
        mask = distance < influence_radius
        U_rep += 0.5 * k_rep * ((1.0 / distance - 1.0 / influence_radius)**2) * mask

    # Total potential
    U_total = U_att + U_rep
    # Clip for visualization purposes
    U_total = np.clip(U_total, 0, 100)
    return U_total

# Compute the potential field
potential_field = compute_potential_field(x_grid, y_grid, target, obstacles, k_att, k_rep, obstacle_radius, influence_radius)

# Plotting the Potential Field (2D Contour and 3D Surface)
fig = plt.figure(figsize=(20, 8))

# 2D Contour Plot
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor('white')  # Set background to white
contour = ax1.contourf(x_grid, y_grid, potential_field, levels=100, cmap='viridis')
fig.colorbar(contour, ax=ax1, label='Potential Field Value')
ax1.plot(target[0], target[1], 'rx', markersize=10, label='Target')

# Plot obstacles on 2D Contour
for idx, obstacle in enumerate(obstacles):
    circle = plt.Circle(obstacle, obstacle_radius, color='k', alpha=0.5)
    ax1.add_patch(circle)
    ax1.text(obstacle[0], obstacle[1], f'O{idx+1}', color='white', ha='center', va='center')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Potential Field (2D Contour)')
ax1.legend()
ax1.axis('equal')
ax1.set_xlim(x_min, x_max)  # Set X-axis limits
ax1.set_ylim(y_min, y_max)  # Set Y-axis limits
ax1.grid(True)

# 3D Surface Plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_facecolor('white')  # Set background to white
surface = ax2.plot_surface(x_grid, y_grid, potential_field, cmap='viridis', edgecolor='none', alpha=0.8)
fig.colorbar(surface, ax=ax2, shrink=0.5, aspect=10, label='Potential Field Value')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Potential')
ax2.set_title('Potential Field (3D Surface)')
ax2.set_xlim(x_min, x_max)  # Set X-axis limits
ax2.set_ylim(y_min, y_max)  # Set Y-axis limits
ax2.set_zlim(0, 100)         # Optional: Set Z-axis limits for consistency
ax2.view_init(elev=30, azim=225)  # Adjust viewing angle for better visualization

plt.tight_layout()
plt.show()

# Simulation loop
plt.figure(figsize=(10, 8))
plt.ion()  # Interactive mode on for dynamic updating
plt.gca().set_facecolor('white')  # Set background to white

iteration = 0
max_iterations_reached = False

while np.linalg.norm(state[:2] - target) > threshold_distance and iteration < max_iterations:
    # Start timing the computation
    start_time = time.time()

    # Current position
    pos = state[:2]
    theta = state[2]

    # Compute attractive force
    _, F_att = attractive_potential(pos, target, k_att)

    # Compute repulsive force and count obstacles influencing the robot
    _, F_rep, obstacles_in_range = repulsive_potential(pos, obstacles, k_rep, obstacle_radius, influence_radius)

    # End timing the computation
    end_time = time.time()
    computation_time = end_time - start_time
    computation_times.append(computation_time)

    # Total force calculations (1 attractive + number of obstacles in range)
    total_force_calculations = 1 + obstacles_in_range
    force_calculations_per_iteration.append(total_force_calculations)
    obstacles_influence_count.append(obstacles_in_range)

    # Total force
    F_total = F_att + F_rep

    # Desired direction
    desired_theta = np.arctan2(F_total[1], F_total[0])

    # Compute control inputs
    # Linear velocity is proportional to the projection of F_total onto the robot's heading
    v = v_max * np.cos(normalize_angle(desired_theta - theta))
    omega = omega_max * normalize_angle(desired_theta - theta)

    # Saturate control inputs
    v = np.clip(v, -v_max, v_max)
    omega = np.clip(omega, -omega_max, omega_max)
    control_input = np.array([v, omega])
    control_inputs.append(control_input)

    # Calculate control effort
    control_effort = np.sum(np.square(control_input))
    total_control_effort += control_effort

    # Update the state
    prev_state = state.copy()
    state = unicycle_dynamics(state, control_input, dt)
    states.append(state.copy())

    # Calculate path length
    delta = np.linalg.norm(state[:2] - prev_state[:2])
    path_length += delta

    # Calculate minimum distance to obstacles
    for obstacle in obstacles:
        distance = np.linalg.norm(state[:2] - obstacle[:2]) - obstacle_radius
        if distance < min_distance:
            min_distance = distance

    # Record distance to target
    distance_to_target = np.linalg.norm(state[:2] - target)
    distances_to_target.append(distance_to_target)

    # Calculate control smoothness (difference between consecutive control inputs)
    if iteration > 0:
        delta_u = np.linalg.norm(control_inputs[-1] - control_inputs[-2])
        control_smoothness.append(delta_u)

    # Visualization
    states_array = np.array(states)
    plt.clf()
    plt.gca().set_facecolor('white')  # Set background to white

    plt.plot(states_array[:, 0], states_array[:, 1], 'b-', label='Executed Trajectory')
    plt.plot(target[0], target[1], 'rx', markersize=10, label='Target')

    # Plot obstacles
    for idx, obstacle in enumerate(obstacles):
        circle = plt.Circle(obstacle, obstacle_radius, color='gray', alpha=0.7)
        plt.gca().add_patch(circle)
        plt.text(obstacle[0], obstacle[1], f'O{idx+1}', color='black', ha='center', va='center')

    # Plot the unicycle as a triangle to indicate orientation
    x_pos, y_pos, theta_pos = state
    triangle_size = 0.3
    triangle = np.array([
        [x_pos + triangle_size * np.cos(theta_pos), y_pos + triangle_size * np.sin(theta_pos)],
        [x_pos + triangle_size * np.cos(theta_pos + 2.5), y_pos + triangle_size * np.sin(theta_pos + 2.5)],
        [x_pos + triangle_size * np.cos(theta_pos - 2.5), y_pos + triangle_size * np.sin(theta_pos - 2.5)]
    ])
    plt.fill(triangle[:, 0], triangle[:, 1], 'g', label='Unicycle')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Unicycle Model Trajectory - APF')
    plt.legend()
    plt.axis('equal')
    plt.xlim(x_min, x_max)  # Set X-axis limits
    plt.ylim(y_min, y_max)  # Set Y-axis limits
    plt.grid(True)
    plt.pause(0.01)

    iteration += 1

if iteration >= max_iterations:
    max_iterations_reached = True

plt.ioff()
plt.show()

# Convert lists to numpy arrays for easier handling
states_array = np.array(states)
control_inputs_array = np.array(control_inputs)

# Calculate average computation time
average_computation_time = np.mean(computation_times)

# Calculate control smoothness metrics
control_smoothness_array = np.array(control_smoothness)
control_smoothness_metric = np.sum(control_smoothness_array**2)  # L2 norm of control input changes

# Calculate final distance to target
final_distance = np.linalg.norm(state[:2] - target)

# Success flag
success = not max_iterations_reached and final_distance <= threshold_distance

# Total force calculations
total_force_calculations = sum(force_calculations_per_iteration)

# Plot metrics
fig, axs = plt.subplots(6, 1, figsize=(10, 24))

# Number of Force Calculations per Iteration
axs[0].plot(force_calculations_per_iteration, 'm-o', label='Force Calculations per Iteration')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Count')
axs[0].set_title('Number of Force Calculations per Iteration')
axs[0].legend()
axs[0].grid(True)
axs[0].set_xlim(0, iteration)

# Number of Obstacles Influencing Robot per Iteration
axs[1].plot(obstacles_influence_count, 'c-o', label='Obstacles Influencing per Iteration')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Count')
axs[1].set_title('Number of Obstacles Influencing Robot per Iteration')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xlim(0, iteration)

# Distance to Target
axs[2].plot(distances_to_target, 'g-o', label='Distance to Target')
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Distance')
axs[2].set_title('Distance to Target per Iteration')
axs[2].legend()
axs[2].grid(True)
axs[2].set_xlim(0, iteration)

# Control Inputs
axs[3].plot(control_inputs_array[:, 0], 'b-', label='Linear Velocity (v)')
axs[3].plot(control_inputs_array[:, 1], 'r-', label='Angular Velocity (ω)')
axs[3].set_xlabel('Iteration')
axs[3].set_ylabel('Control Input')
axs[3].set_title('Control Inputs over Time')
axs[3].legend()
axs[3].grid(True)
axs[3].set_xlim(0, iteration)

# Path Length Over Time
cumulative_path_length = np.cumsum(np.linalg.norm(np.diff(states_array[:, :2], axis=0), axis=1))
axs[4].plot(cumulative_path_length, 'k-o', label='Cumulative Path Length')
axs[4].set_xlabel('Iteration')
axs[4].set_ylabel('Path Length')
axs[4].set_title('Cumulative Path Length over Time')
axs[4].legend()
axs[4].grid(True)
axs[4].set_xlim(0, iteration)

# Computation Times
axs[5].plot(computation_times, 'k-o', label='Computation Time per Iteration')
axs[5].set_xlabel('Iteration')
axs[5].set_ylabel('Time (s)')
axs[5].set_title('Computation Time per Iteration')
axs[5].legend()
axs[5].grid(True)
axs[5].set_xlim(0, iteration)

plt.tight_layout()
plt.show()

import csv
import os

# Define the CSV file name
csv_filename = 'apf_summary_metrics.csv'

# Define the APF metrics as a dictionary
metrics = {
    "Map number": randSeed,
    "Total Path Length": path_length,
    "Final Distance to Target": final_distance,
    "Minimum Distance to Obstacles": min_distance,
    "Total Control Effort": total_control_effort,
    "Average Computation Time per Iteration": average_computation_time,
    "Total Number of Iterations": iteration,
    "Total Force Calculations": total_force_calculations,
    "L2 Norm of Control Input Changes": control_smoothness_metric,
    "Success": success
}

# Define the column headers explicitly to ensure consistent order
fieldnames = [
    "Map number",
    "Total Path Length",
    "Final Distance to Target",
    "Minimum Distance to Obstacles",
    "Total Control Effort",
    "Average Computation Time per Iteration",
    "Total Number of Iterations",
    "Total Force Calculations",
    "L2 Norm of Control Input Changes",
    "Success"
]

# Check if the CSV file already exists
file_exists = os.path.isfile(csv_filename)

# Open the CSV file in append mode
with open(csv_filename, mode='a', newline='') as csv_file:
    # Create a CSV DictWriter
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header only if the file is new
    if not file_exists:
        writer.writeheader()
    
    # Write the metrics to the CSV file
    writer.writerow(metrics)

# Print Summary Metrics to Console
print(f"Map number: {metrics['Map number']}")
print("\n--- APF Summary Metrics ---")
print(f"Total Path Length: {metrics['Total Path Length']:.4f}")
print(f"Final Distance to Target: {metrics['Final Distance to Target']:.4f}")
print(f"Minimum Distance to Obstacles: {metrics['Minimum Distance to Obstacles']:.4f}")
print(f"Total Control Effort: {metrics['Total Control Effort']:.4f}")
print(f"Average Computation Time per Iteration: {metrics['Average Computation Time per Iteration']:.6f} seconds")
print(f"Total Number of Iterations: {metrics['Total Number of Iterations']}")
print(f"Total Force Calculations: {metrics['Total Force Calculations']}")
print(f"L2 Norm of Control Input Changes: {metrics['L2 Norm of Control Input Changes']:.4f}")
print(f"Success: {metrics['Success']}")
