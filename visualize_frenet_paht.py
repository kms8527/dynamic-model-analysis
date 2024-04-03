# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# # Define the time range for the path simulation
# T = np.linspace(0, 4, 100)

# # Function to draw a car with a given center position and yaw
# def draw_car(ax, center_x, center_y, yaw, length, width, color='blue'):
#     corner_x = center_x - length / 2
#     corner_y = center_y - width / 2

#     # Create a rectangle in the angle of the yaw
#     car = patches.Rectangle(
#         (corner_x, corner_y), length, width, angle=np.degrees(yaw),
#         color=color, fill=False
#     )

#     # Add the car to the axes
#     ax.add_patch(car)

# # Function to calculate velocity coefficients
# def calculate_velocity_coefficients(v_s, v_f, r_t):
#     k0 = v_s
#     k1 = 0
#     k2 = 3 * (v_f - v_s) / (r_t ** 2)
#     k3 = -2 * (v_f - v_s) / (r_t ** 3)
#     k4 = 0
#     return [k4, k3, k2, k1, k0]

# # Function to get cartesian coordinates from frenet coordinates
# def get_cartesian_coordinates(s, d):
#     x = s
#     y = y_offset + amplitude * np.sin(frequency * s) + d
#     return x, y

# # Curved Reference Path parameters
# y_offset = 1.0
# amplitude = 1.5
# frequency = 0.1

# # Starting lateral position at d=1.0
# start_d = 1.0

# # Visualization
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot curved reference path
# s_values = T*5  # Scale for the x-axis to represent distance along the path
# x_values, y_values = get_cartesian_coordinates(s_values, np.zeros_like(s_values))
# ax.plot(x_values, y_values, 'k--', label="Curved Reference Line")

# # Colors for gradients
# colors = plt.cm.viridis(np.linspace(0, 1, 100))

# # Starting lateral position and desired ending lateral position
# start_d = 0.0
# end_d = 2.0  # Ending lateral offset

# # Loop to generate multiple paths
# for i, color in enumerate(colors):
#     # v_s, v_f, and r_t are changed according to some logic or randomly
#     # For example, they are linearly spaced for illustration purposes
#     v_s = 2.0 + (5.0 / 100) * i
#     v_f = 3.0 + (5.0 / 100) * i
#     r_t = 2.0 + (2.0 / 100) * i

#     # Calculate velocity profile coefficients
#     coeffs = calculate_velocity_coefficients(v_s, v_f, r_t)

#     # Generate velocity profile and corresponding s values
#     velocity_profile = np.polyval(coeffs, T)
#     s_values = np.cumsum(velocity_profile) * (T[1] - T[0])

#     # Generate lateral trajectory candidates
#     # d_traj = start_d - np.linspace(-0.001, 0.001, 100)[i] * s_values**2
#     d_values = np.linspace(start_d, end_d, len(s_values))

#     x_traj, y_traj = get_cartesian_coordinates(s_values, d_values)
#     ax.plot(x_traj, y_traj, color=color)
# # Car dimensions
# car_length = 4.5
# car_width = 2.0
# start_x, start_y = get_cartesian_coordinates(0, start_d)

# # Starting yaw angle, based on the derivative of the reference path at s=0

# start_yaw = np.arctan2(amplitude * frequency, 1)
# draw_car(ax, start_x, start_y, start_yaw, car_length, car_width)

# ax.set_xlabel("X [m]")
# ax.set_ylabel("Y [m]")
# ax.set_title("Frenet Path Candidates with Gradient Colors")
# ax.grid(True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define the centerline as a function or a set of points.
centerline_x = np.linspace(-10, 10, 100)
centerline_y = np.sin(centerline_x)

# Define the agent's trajectory.
trajectory_x = centerline_x + np.random.normal(0, 0.5, size=centerline_x.size)
trajectory_y = centerline_y + np.random.normal(0, 0.5, size=centerline_y.size)

# Plot the centerline
plt.plot(centerline_x, centerline_y, label='centerline')

# Plot the agent's trajectory
plt.plot(trajectory_x, trajectory_y, 'r--', label='agent trajectory')

# Add reward points
reward_points_x = [1, 3, 5, 7]
reward_points_y = np.sin(reward_points_x)
plt.scatter(reward_points_x, reward_points_y, color='green', label='delta-progress reward (r)')

# Annotate the reward points
for (i, (x, y)) in enumerate(zip(reward_points_x, reward_points_y), start=1):
    plt.annotate(f'cp{i}', (x, y), textcoords="offset points", xytext=(-15,10), ha='center')

# Annotate the starting point
plt.annotate('r1', (trajectory_x[0], trajectory_y[0]), textcoords="offset points", xytext=(-15,10), ha='center')

# Add legend and titles
plt.legend()
plt.title('Agent Trajectory and Centerline Progress')
plt.xlabel('Position on X')
plt.ylabel('Position on Y')

# Display the plot
plt.show()
