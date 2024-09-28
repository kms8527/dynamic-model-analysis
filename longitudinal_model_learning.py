import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Load data from CSV
def load_data(file_name):
    data = pd.read_csv(file_name)
    return data['time'].values, data['velocity_x'].values, data['acceleration'].values

# Construct the regressor matrix and output vector
def build_regression_matrix(y, u, n=2, m=2):
    """
    y: velocity (output)
    u: acceleration (control input)
    n: number of past outputs to consider
    m: number of past inputs to consider
    """
    N = len(y)  # Total number of data points
    rows = N - max(n, m)  # Number of rows for the regression matrix
    if rows <= 0:
        raise ValueError("Not enough data points to construct the regression matrix.")
    
    Phi = np.zeros((rows, n + m))  # Regression matrix with size (rows, n+m)
    Y = np.zeros(rows)  # Output vector with size (rows,)

    # Build the regression matrix and output vector
    for k in range(max(n, m), N):
        # Ensure there are enough past values for both outputs and inputs
        if k - n < 0 or k - m < 0:
            continue

        # Correct the slicing logic for past values of y and u
        past_outputs = -y[k-n:k]  # Past outputs: -y(k-1), -y(k-2), ..., -y(k-n)
        past_inputs = u[k-m:k]    # Past inputs: u(k-1), u(k-2), ..., u(k-m)

        # Create the regression vector (phi)
        phi_k = np.concatenate((past_outputs, past_inputs))

        # Check that phi_k has the correct length
        if len(phi_k) == n + m:
            Phi[k - max(n, m)] = phi_k  # Store the regression vector
            Y[k - max(n, m)] = y[k]  # Store the output y(k+1)

    return Phi, Y

# Perform model identification using Least Squares
def identify_model(Phi, Y):
    # Solve the normal equation to get the parameters theta (A matrix)
    theta = inv(Phi.T @ Phi) @ Phi.T @ Y
    return theta

# Apply the model to predict future outputs
def predict_output(theta, Phi):
    # Predict future outputs using the estimated model parameters
    Y_pred = Phi @ theta
    return Y_pred

def accumulate_prediction(theta, y, u, n=2, m=2):
    
    N = len(y)
    y_pred = np.zeros(N)
    y_pred[:n] = y[:n]  # Initialize the first n values with actual data

    # Simulate future steps using accumulated predictions
    for k in range(n, N-1):
        # Construct the regressor with past predictions and input
        past_outputs = -y_pred[k-n:k]  # Use predicted outputs
        past_inputs = u[k-m:k]  # Use actual control inputs

        # Create the regression vector for this step
        phi_k = np.concatenate((past_outputs, past_inputs))

        # Predict the next output using the identified model parameters
        y_pred[k] = np.dot(theta, phi_k)

    return y_pred

# Load the data from CSV file
time, velocity, acceleration = load_data('./2024_competition_dataset/avante_cn7_info_cmd_vel_data.csv')

# Number of past values to use for outputs and inputs
n = 10 # Number of past outputs (adjust based on system order)
m = 10 # Number of past inputs (adjust based on system order)

# Build the regression matrix (Phi) and the output vector (Y)
Phi, Y = build_regression_matrix(velocity, acceleration, n, m)

# Perform least squares to identify the model parameters (theta)
theta = identify_model(Phi, Y)

# Predict the output using the identified model
Y_pred = predict_output(theta, Phi)
Y_pred_accumulated = accumulate_prediction(theta, velocity, acceleration, n, m)
# Print the identified parameters (theta corresponds to A matrix)
print("Identified Parameters (Theta):", theta)

# Optional: Compare actual vs predicted outputs
plt.figure()
plt.plot(time[n:], Y, label='Actual Output (y)')
plt.plot(time[n:], Y_pred, label='Predicted Output (y_pred)')
plt.xlabel('Time Step')
plt.ylabel('Velocity (m/s)')
plt.title('Actual vs Predicted Velocity')
plt.legend()

# acumulate the prediction output
plt.figure()
plt.plot(time, velocity, label='Actual Output (y)')
plt.plot(time, Y_pred_accumulated, label='Predicted Output (y_pred)')
plt.xlabel('Time Step')
plt.ylabel('Velocity (m/s)')
plt.title('Actual vs Accumulated Predicted Velocity')
plt.legend() 

plt.show()



# print("A : ", theta[0:2])
# print("B :", theta[2:4])

# RMSE Calculation
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Calculate RMSE
rmse = calculate_rmse(Y, Y_pred)
print("Root Mean Squared Error (RMSE):", rmse)
