import numpy as np
import matplotlib.pyplot as plt
import time

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
    for obstacle in obstacles:
        obs_pos = obstacle[:2]
        distance = np.linalg.norm(pos - obs_pos)
        if distance <= obstacle_radius:
            distance = obstacle_radius  # Avoid division by zero
        if distance < influence_radius:
            # Potential
            pot = 0.5 * k_rep * (1.0 / distance - 1.0 / influence_radius)**2
            potential += pot
            # Force
            repulsion = k_rep * (1.0 / distance - 1.0 / influence_radius) * (1.0 / distance**2) * (pos - obs_pos) / distance
            force += repulsion
    return potential, force

# Normalize angle to [-pi, pi]
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Simulation parameters
dt = 0.1  # Time step
max_iterations = 500

# Initial state [x, y, theta]
initial_state = np.array([0.0, 0.0, 0.0])

# Target position [x, y]
target = np.array([5.0, 5.0])

# Random obstacle positions [x, y]
np.random.seed(45)  # For reproducibility

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

# Initialize lists to store computation times
computation_times = []

# Create a grid to compute the potential field
x_min, x_max = -1, 6
y_min, y_max = -1, 6
resolution = 0.05
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
potential_field = np.zeros_like(x_grid)

# Compute the potential field over the grid
for i in range(x_grid.shape[0]):
    for j in range(x_grid.shape[1]):
        pos = np.array([x_grid[i, j], y_grid[i, j]])
        # Compute attractive potential
        U_att, _ = attractive_potential(pos, target, k_att)
        # Compute repulsive potential
        U_rep, _ = repulsive_potential(pos, obstacles, k_rep, obstacle_radius, influence_radius)
        # Total potential
        potential_field[i, j] = U_att + U_rep

# Normalize the potential field for better visualization
potential_field = np.clip(potential_field, 0, 100)

plt.figure(figsize=(10, 8))
plt.contourf(x_grid, y_grid, potential_field, levels=100, cmap='viridis')
plt.colorbar(label='Potential Field Value')
plt.plot(target[0], target[1], 'rx', markersize=10, label='Target')

# Plot obstacles
for idx, obstacle in enumerate(obstacles):
    circle = plt.Circle(obstacle, obstacle_radius, color='k', alpha=0.5)
    plt.gca().add_patch(circle)
    plt.text(obstacle[0], obstacle[1], f'O{idx+1}', color='white', ha='center', va='center')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Potential Field Surface')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# Simulation loop
plt.figure(figsize=(10, 8))
plt.ion()  # Interactive mode on for dynamic updating

iteration = 0

while np.linalg.norm(state[:2] - target) > 0.1 and iteration < max_iterations:
    start_time = time.time()

    # Current position
    pos = state[:2]
    theta = state[2]

    # Compute attractive and repulsive forces
    _, F_att = attractive_potential(pos, target, k_att)
    _, F_rep = repulsive_potential(pos, obstacles, k_rep, obstacle_radius, influence_radius)

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

    # Update the state
    state = unicycle_dynamics(state, control_input, dt)
    states.append(state.copy())

    end_time = time.time()
    computation_time = end_time - start_time
    computation_times.append(computation_time)

    # Visualization
    states_array = np.array(states)
    plt.clf()
    plt.contourf(x_grid, y_grid, potential_field, levels=100, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Potential Field Value')
    plt.plot(states_array[:, 0], states_array[:, 1], 'b-', label='Executed Trajectory')
    plt.plot(target[0], target[1], 'rx', markersize=10, label='Target')

    # Plot obstacles
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
    plt.title('Artificial Potential Fields for Unicycle Model')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.pause(0.01)

    iteration += 1

plt.ioff()
plt.show()

# Plot computation times
plt.figure(figsize=(10, 4))
plt.plot(computation_times, 'r-o', label='Computation Time (s)')
plt.xlabel('Iteration')
plt.ylabel('Time (s)')
plt.title('Computation Time per Iteration')
plt.legend()
plt.grid(True)
plt.show()
