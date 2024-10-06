import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# Define the unicycle model dynamics
def unicycle_dynamics(state, control, dt):
    x, y, theta = state
    v, omega = control

    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt

    return np.array([x_next, y_next, theta_next])

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
target = np.array([5.0, 5.0])

# Random obstacle positions [x, y]
#np.random.seed(41)  # For reproducibility
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
R = np.eye(2) * 0.1         # Control effort weight
Q_terminal = np.eye(2) * 50.0  # Terminal cost weight
obstacle_weight = 100.0     # Obstacle avoidance weight
alpha = 10.0                # Exponential penalty factor for obstacle avoidance

# Initial guess for control inputs (straight forward movement)
u0 = np.tile([0.5, 0.0], N)  # [v, omega] for each time step

# Bounds for control inputs
bounds = [ (v_min, v_max), (omega_min, omega_max) ] * N

# Create lists to store computation times and costs
computation_times = []
cost_values = []

# Initialize state and control histories
state = initial_state.copy()
states = [state.copy()]

plt.figure(figsize=(10, 8))
plt.ion()

iteration = 0
max_iterations = 100  # Prevent infinite loops

while np.linalg.norm(state[:2] - target) > 0.1 and iteration < max_iterations:
    # Arguments for the cost function
    args = (state, target, N, dt, obstacles, Q, R, Q_terminal, obstacle_weight, alpha, obstacle_radius)
    
    # Solve the MPC optimization problem
    start_time = time.time()
    result = minimize(cost_function, u0, args=args, bounds=bounds, method='SLSQP', options={'ftol':1e-4, 'maxiter': 1000})
    end_time = time.time()
    
    # Check if the optimization was successful
    if not result.success:
        print(f'Iteration {iteration}: Optimization failed. {result.message}')
        break
    
    # Store computation time and cost
    computation_time = end_time - start_time
    computation_times.append(computation_time)
    cost_values.append(result.fun)
    print(f'Iteration {iteration}: Computation Time = {computation_time:.4f}s, Cost = {result.fun:.4f}')
    
    # Extract the optimal control inputs
    u_opt = result.x
    control_input = u_opt[:2]  # Apply only the first control input
    
    # Update the state
    state = unicycle_dynamics(state, control_input, dt)
    states.append(state.copy())
    
    # Update the initial guess for the next iteration (Shift and append last control)
    u0 = np.roll(u_opt, -2)
    u0[-2:] = [0.0, 0.0]  # Assume zero control for the last step
    
    # Visualization
    states_array = np.array(states)
    plt.clf()
    plt.plot(states_array[:, 0], states_array[:, 1], 'b-', label='Trajectory')
    plt.plot(target[0], target[1], 'rx', markersize=10, label='Target')
    for idx, obstacle in enumerate(obstacles):
        circle = plt.Circle(obstacle, obstacle_radius, color='k', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.text(obstacle[0], obstacle[1], f'O{idx+1}', color='white', ha='center', va='center')
    
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
    plt.grid(True)
    plt.pause(0.01)
    
    iteration += 1

plt.ioff()
plt.show()

# Plot computation times and costs
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Computation Time
axs[0].plot(computation_times, 'r-o', label='Computation Time (s)')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Time (s)')
axs[0].set_title('Computation Time per Iteration')
axs[0].legend()
axs[0].grid(True)

# Cost Values
axs[1].plot(cost_values, 'b-o', label='Cost')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Cost')
axs[1].set_title('Cost per Iteration')
axs[1].legend()
axs[1].grid(True)

# Distance to Obstacles at Final State
final_state = states[-1][:2]
distances_to_obstacles = [np.linalg.norm(final_state - obstacle) for obstacle in obstacles]
axs[2].bar(range(1, len(obstacles)+1), distances_to_obstacles, color='g')
axs[2].set_xlabel('Obstacle Index')
axs[2].set_ylabel('Distance')
axs[2].set_title('Distance from Final State to Obstacles')
axs[2].grid(True)

plt.tight_layout()
plt.show()
