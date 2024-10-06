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
    state, target, N, dt, obstacles, Q, R, Q_terminal, obstacle_radius = args
    cost = 0.0
    
    # Simulate the system over the prediction horizon
    for i in range(N):
        control = u[i*2:(i+1)*2]
        state = unicycle_dynamics(state, control, dt)
        
        # Calculate the cost as the distance to the target and penalize large control inputs
        state_error = state[:2] - target[:2]
        cost += state_error.T @ Q @ state_error + control.T @ R @ control
        
        # Add a strict obstacle avoidance cost for each obstacle
        for obstacle in obstacles:
            distance_to_obstacle = np.linalg.norm(state[:2] - obstacle[:2])
            if distance_to_obstacle < obstacle_radius:
                cost += 10000.0 / (distance_to_obstacle + 0.1)  # Increased penalty to avoid collision
    
    # Terminal cost to encourage reaching the target
    state_error = state[:2] - target[:2]
    cost += state_error.T @ Q_terminal @ state_error
    
    return cost

# MPC parameters
N = 20  # Prediction horizon
dt = 0.1  # Time step

# Initial state [x, y, theta]
initial_state = np.array([0.0, 0.0, 0.0])

# Target state [x, y]
target = np.array([5.0, 5.0])

# Random obstacle positions [x, y]
np.random.seed(42)  # For reproducibility
obstacles = [
    np.random.uniform(0, 5, 2),
    np.random.uniform(0, 5, 2),
    np.random.uniform(0, 5, 2),
    np.random.uniform(0, 5, 2)
]

# Obstacle radius for collision avoidance
obstacle_radius = 0.75

# Initial guess for control inputs (provide a slight forward velocity to start)
u0 = np.tile([0.5, 0.0], N)  # [v, omega] for each time step

# Bounds for control inputs (v, omega)
bounds = [(-1.5, 1.5), (-np.pi/4, np.pi/4)] * N  # Increased bounds for more maneuverability

# Define weight matrices for the cost function
Q = np.eye(2) * 10  # State error weight
R = np.eye(2) * 0.1  # Control effort weight
Q_terminal = np.eye(2) * 50  # Terminal cost weight

# Create lists to store computation times and costs
computation_times = []
cost_values = []

# Solve the MPC optimization problem
start_time = time.time()
result = minimize(cost_function, u0, args=(initial_state, target, N, dt, obstacles, Q, R, Q_terminal, obstacle_radius), bounds=bounds, method='SLSQP')
end_time = time.time()

# Store computation time and cost
computation_times.append(end_time - start_time)
cost_values.append(result.fun)
print(f'Iteration 0: Computation Time = {computation_times[-1]:.4f}s, Cost = {cost_values[-1]:.4f}')

# Extract the optimal control inputs
u_opt = result.x

# Simulate the system with the optimal control inputs
state = initial_state
states = [state]
plt.figure()
for i in range(N):
    control = u_opt[i*2:(i+1)*2]
    state = unicycle_dynamics(state, control, dt)
    states.append(state)
    
    # Plot the current state
    states_array = np.array(states)
    plt.clf()
    plt.plot(states_array[:, 0], states_array[:, 1], 'b-', label='Trajectory')
    plt.plot(target[0], target[1], 'rx', label='Target')
    for obstacle in obstacles:
        circle = plt.Circle(obstacle, obstacle_radius, color='k', fill=True, label='Obstacle')
        plt.gca().add_patch(circle)
    
    # Plot the unicycle as a triangle
    x, y, theta = state
    triangle = plt.Polygon(
        [[x, y], [x - 0.2 * np.cos(theta + np.pi / 2), y - 0.2 * np.sin(theta + np.pi / 2)],
         [x - 0.2 * np.cos(theta - np.pi / 2), y - 0.2 * np.sin(theta - np.pi / 2)]],
        color='g'
    )
    plt.gca().add_patch(triangle)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.title('MPC for Unicycle Model - Optimization Step')
    plt.pause(0.1)

# Continue simulating until close to the target
iteration = 1
while np.linalg.norm(state[:2] - target) > 0.1:
    # Recalculate optimal control inputs at each step to avoid oscillations
    start_time = time.time()
    result = minimize(cost_function, u0, args=(state, target, N, dt, obstacles, Q, R, Q_terminal, obstacle_radius), bounds=bounds, method='SLSQP')
    end_time = time.time()
    
    # Store computation time and cost
    computation_times.append(end_time - start_time)
    cost_values.append(result.fun)
    print(f'Iteration {iteration}: Computation Time = {computation_times[-1]:.4f}s, Cost = {cost_values[-1]:.4f}')
    iteration += 1
    
    u_opt = result.x
    control = u_opt[:2]
    state = unicycle_dynamics(state, control, dt)
    states.append(state)
    
    # Plot the current state
    states_array = np.array(states)
    plt.clf()
    plt.plot(states_array[:, 0], states_array[:, 1], 'b-', label='Trajectory')
    plt.plot(target[0], target[1], 'rx', label='Target')
    for obstacle in obstacles:
        circle = plt.Circle(obstacle, obstacle_radius, color='k', fill=True, label='Obstacle')
        plt.gca().add_patch(circle)
    
    # Plot the unicycle as a triangle
    x, y, theta = state
    triangle = plt.Polygon(
        [[x, y], [x - 0.2 * np.cos(theta + np.pi / 2), y - 0.2 * np.sin(theta + np.pi / 2)],
         [x - 0.2 * np.cos(theta - np.pi / 2), y - 0.2 * np.sin(theta - np.pi / 2)]],
        color='g'
    )
    plt.gca().add_patch(triangle)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.title('MPC for Unicycle Model - Moving to Target')
    plt.pause(0.1)

# Plot computation times and costs
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(computation_times, 'r-', label='Computation Time (s)')
plt.xlabel('Iteration')
plt.ylabel('Time (s)')
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(cost_values, 'b-', label='Cost')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.grid()

# Plot distance from each obstacle at the end
final_state = states[-1][:2]
distances_to_obstacles = [np.linalg.norm(final_state - obstacle) for obstacle in obstacles]
plt.subplot(3, 1, 3)
plt.bar(range(len(obstacles)), distances_to_obstacles, color='g', label='Distance to Obstacles')
plt.xlabel('Obstacle Index')
plt.ylabel('Distance')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()