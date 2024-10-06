import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

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
            distance = obstacle_radius  # Avoid division by zero or negative distances
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
np.random.seed(5)  # For reproducibility

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
ax1.grid(True)

# 3D Surface Plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surface = ax2.plot_surface(x_grid, y_grid, potential_field, cmap='viridis', edgecolor='none', alpha=0.8)
fig.colorbar(surface, ax=ax2, shrink=0.5, aspect=10, label='Potential Field Value')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Potential')
ax2.set_title('Potential Field (3D Surface)')
ax2.view_init(elev=30, azim=225)  # Adjust viewing angle for better visualization

plt.tight_layout()
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
    contour = plt.contourf(x_grid, y_grid, potential_field, levels=100, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='Potential Field Value')
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
