import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import csv
import os
randSeed = 14




# Define the unicycle model dynamics
def unicycle_dynamics(state, control, dt):
    x, y, theta = state
    v, omega = control

    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt

    return np.array([x_next, y_next, theta_next])

# Simulate the predicted trajectory based on control inputs
def simulate_predicted_trajectory(state, u_opt, N, dt):
    predicted_states = [state.copy()]
    current_state = state.copy()
    for i in range(N):
        control = u_opt[i*2:(i+1)*2]
        current_state = unicycle_dynamics(current_state, control, dt)
        predicted_states.append(current_state.copy())
    return np.array(predicted_states)

# Define the cost function for MPC
def cost_function(u, *args):
    state, target, N, dt, obstacles, Q, R, Q_terminal, obstacle_weight, alpha, obstacle_radius = args
    cost = 0.0
    current_state = state.copy()

    for i in range(N):
        control = u[i*2:(i+1)*2]
        current_state = unicycle_dynamics(current_state, control, dt)

        # State error
        state_error = current_state[:2] - target[:2]
        cost += state_error.T @ Q @ state_error

        # Control effort
        cost += control.T @ R @ control

        # Obstacle avoidance (Exponential Penalty)
        for obstacle in obstacles:
            distance = np.linalg.norm(current_state[:2] - obstacle[:2])
            if distance < (obstacle_radius + 0.5):  # Additional buffer
                obstacle_penalty = obstacle_weight * np.exp(-alpha * (distance - obstacle_radius))
                cost += obstacle_penalty

    # Terminal cost
    terminal_error = current_state[:2] - target[:2]
    cost += terminal_error.T @ Q_terminal @ terminal_error

    return cost

# MPC parameters
N = 20  # Prediction horizon
dt = 0.1  # Time step

# Initial state [x, y, theta]
initial_state = np.array([0.0, 0.0, 0.0])

# Target state [x, y]
target = np.array([6.0, 5.0])

# Random obstacle positions [x, y]
np.random.seed(randSeed)  # For reproducibility

# Adjusted obstacle generation to match the APF script
obstacles = np.array([
    np.random.uniform(0, 5, 2),
    np.random.uniform(0, 5, 2),
    np.random.uniform(0, 5, 2),
    np.random.uniform(0, 5, 2)
])

# Obstacle radius for collision avoidance
obstacle_radius = 0.75

# Control input limits
v_min, v_max = -1.0, 1.0  # Linear velocity limits
omega_min, omega_max = -np.pi/4, np.pi/4  # Angular velocity limits

# Define weight matrices for the cost function
Q = np.eye(2) * 10.0        # State error weight
R = np.eye(2) * 1           # Control effort weight

Q_terminal = np.eye(2) * 50.0  # Terminal cost weight
obstacle_weight = 100.0         # Obstacle avoidance weight
alpha = 10.0                    # Exponential penalty factor for obstacle avoidance

# Initial guess for control inputs (straight forward movement)
u0 = np.tile([0.5, 0.0], N)  # [v, omega] for each time step

# Bounds for control inputs
bounds = [ (v_min, v_max), (omega_min, omega_max) ] * N

# Create lists to store metrics
computation_times = []
cost_values = []
path_length = 0.0
min_distance = float('inf')
total_control_effort = 0.0
control_inputs = []
distances_to_target = []
nfev_list = []  # Number of function evaluations per iteration
nit_list = []   # Number of optimizer iterations per iteration
control_smoothness = []  # To calculate control smoothness

# Initialize state and control histories
state = initial_state.copy()
states = [state.copy()]
predicted_trajectories = []  # To store predicted trajectories for visualization

plt.ion()  # Interactive mode on for dynamic updating
plt.figure(figsize=(10, 8))
plt.gca().set_facecolor('white')  # Set background to white

iteration = 0
max_iterations = 150  # Prevent infinite loops

while np.linalg.norm(state[:2] - target) > 0.1 and iteration < max_iterations:
    # Start timing the computation
    start_time = time.time()

    # Arguments for the cost function
    args = (state, target, N, dt, obstacles, Q, R, Q_terminal, obstacle_weight, alpha, obstacle_radius)

    # Solve the MPC optimization problem
    result = minimize(cost_function, u0, args=args, bounds=bounds, method='SLSQP', options={'ftol':1e-4, 'maxiter': 1000})

    # End timing the computation
    end_time = time.time()
    computation_time = end_time - start_time
    computation_times.append(computation_time)

    # Check if the optimization was successful
    if not result.success:
        print(f'Iteration {iteration}: Optimization failed. {result.message}')
        break

    # Store optimizer metrics
    nfev_list.append(result.nfev)
    nit_list.append(result.nit)

    # Store cost
    cost_values.append(result.fun)

    # Extract the optimal control inputs
    u_opt = result.x
    control_input = u_opt[:2]  # Apply only the first control input
    control_inputs.append(control_input)

    # Calculate control effort
    control_effort = np.sum(np.square(control_input))
    total_control_effort += control_effort

    # Simulate the predicted trajectory based on u_opt
    predicted_states = simulate_predicted_trajectory(state, u_opt, N, dt)
    predicted_trajectories.append(predicted_states)

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

    # Update the initial guess for the next iteration (Shift and append last control)
    u0 = np.roll(u_opt, -2)
    u0[-2:] = [0.0, 0.0]  # Assume zero control for the last step

    # Visualization
    states_array = np.array(states)
    plt.clf()
    plt.gca().set_facecolor('white')  # Set background to white

    # Plot executed trajectory
    plt.plot(states_array[:, 0], states_array[:, 1], 'b-', label='Executed Trajectory')

    # Plot predicted trajectory
    if predicted_states.shape[0] > 1:
        plt.plot(predicted_states[:, 0], predicted_states[:, 1], 'm--', label='Predicted Trajectory')

    # Plot target
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
    plt.title('MPC for Unicycle Model - Moving to Target')
    plt.legend()
    plt.axis('equal')
    plt.xlim(-1, 7)
    plt.ylim(-1, 7)
    plt.grid(True)
    plt.pause(0.01)

    iteration += 1

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
success = iteration < max_iterations and final_distance <= 0.1

# Total number of function evaluations and optimizer iterations
total_nfev = sum(nfev_list)
total_nit = sum(nit_list)

# Plot optimizer metrics
fig_metrics, axs = plt.subplots(6, 1, figsize=(10, 24))

# Number of Function Evaluations per Iteration
axs[0].plot(nfev_list, 'm-o', label='Function Evaluations (nfev)')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Count')
axs[0].set_title('Number of Function Evaluations per Iteration')
axs[0].legend()
axs[0].grid(True)
axs[0].set_xlim(0, iteration)

# Number of Optimizer Iterations per Iteration
axs[1].plot(nit_list, 'c-o', label='Optimizer Iterations (nit)')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Count')
axs[1].set_title('Number of Optimizer Iterations per Iteration')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xlim(0, iteration)

# Cost Values
axs[2].plot(cost_values, 'b-o', label='Cost')
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Cost')
axs[2].set_title('Cost per Iteration')
axs[2].legend()
axs[2].grid(True)
axs[2].set_xlim(0, iteration)

# Distance to Target
axs[3].plot(distances_to_target, 'g-o', label='Distance to Target')
axs[3].set_xlabel('Iteration')
axs[3].set_ylabel('Distance')
axs[3].set_title('Distance to Target per Iteration')
axs[3].legend()
axs[3].grid(True)
axs[3].set_xlim(0, iteration)

# Control Inputs
axs[4].plot(control_inputs_array[:, 0], 'b-', label='Linear Velocity (v)')
axs[4].plot(control_inputs_array[:, 1], 'r-', label='Angular Velocity (ω)')
axs[4].set_xlabel('Iteration')
axs[4].set_ylabel('Control Input')
axs[4].set_title('Control Inputs over Time')
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
csv_filename = 'mpc_summary_metrics.csv'

# Define the metrics as a dictionary
metrics = {
    "Map number": randSeed,
    "Total Path Length": path_length,
    "Final Distance to Target": final_distance,
    "Minimum Distance to Obstacles": min_distance,
    "Total Control Effort": total_control_effort,
    "Average Computation Time per Iteration": average_computation_time,
    "Total Number of Iterations": iteration,
    "Total Number of Function Evaluations": total_nfev,
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
    "Total Number of Function Evaluations",
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
print("\n--- MPC Summary Metrics ---")
print(f"Map number: {metrics['Map number']}")
print(f"Total Path Length: {metrics['Total Path Length']:.4f}")
print(f"Final Distance to Target: {metrics['Final Distance to Target']:.4f}")
print(f"Minimum Distance to Obstacles: {metrics['Minimum Distance to Obstacles']:.4f}")
print(f"Total Control Effort: {metrics['Total Control Effort']:.4f}")
print(f"Average Computation Time per Iteration: {metrics['Average Computation Time per Iteration']:.6f} seconds")
print(f"Total Number of Iterations: {metrics['Total Number of Iterations']}")
print(f"Total Number of Function Evaluations: {metrics['Total Number of Function Evaluations']}")
print(f"L2 Norm of Control Input Changes: {metrics['L2 Norm of Control Input Changes']:.4f}")
print(f"Success: {metrics['Success']}")
