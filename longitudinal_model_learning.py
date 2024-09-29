import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

load_train_file_name = 'simple_simulation_data1.csv'
load_valid_file_name = 'simple_simulation_data2.csv'

# Number of past values to use for outputs and inputs
n = 1  # Number of past outputs (adjust based on system order)
m = 2 # Number of past inputs (adjust based on system order)

Ts = 0.01
test_Ts = 0.01

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

import numpy as np

def stochastic_gradient_descent(Phi, Y, alpha, num_iters, batch_size, theta=None):
    """
    확률적 경사 하강법(SGD)을 사용하여 파라미터 theta를 학습하는 함수

    Parameters:
    - Phi: (N, M) 행렬, 설명 변수 벡터
    - Y: (N, 1) 또는 (N,) 벡터, 종속 변수 벡터
    - alpha: 학습률
    - num_iters: 총 반복 횟수
    - batch_size: 미니배치 크기
    - theta: (M, 1) 벡터, 초기화된 파라미터 (기본값은 무작위 초기화)

    Returns:
    - theta: 학습된 파라미터 벡터
    - cost_history: 비용 함수 값들의 기록
    """

    N, M = Phi.shape
    
    # 초기 theta 값 설정 (주어지지 않았을 경우)
    if theta is None:
        theta = np.random.randn(M, 1)
    
    cost_history = []

    # Y가 1차원 벡터일 경우, 2차원으로 변환
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    for i in range(num_iters):
        # 미니배치 선택
        batch_indices = np.random.choice(N, batch_size, replace=False)
        Phi_batch = Phi[batch_indices, :]
        Y_batch = Y[batch_indices]  # 1차원일 때를 고려한 인덱싱

        # 예측값 계산 (y = Phi @ theta)
        predictions = Phi_batch @ theta

        if (np.isnan(predictions).any()):
            print("Nan value detected")
            break
        
        # 오차 계산
        errors = predictions - Y_batch

        # 경사 계산 (평균 제곱 오차에 대한 gradient)
        gradient = (Phi_batch.T @ errors) / batch_size

        # 파라미터 업데이트 (theta 갱신)
        theta -= alpha * gradient

        # 비용 함수 계산 및 기록
        cost = (1 / (2 * batch_size)) * np.sum(errors ** 2)
        cost_history.append(cost)

    return theta, cost_history


# Apply the model to predict future outputs
def predict_output(theta, Phi):
    # Predict future outputs using the estimated model parameters
    Y_pred = Phi @ theta
    return Y_pred

def accumulate_prediction(theta, y, u, n=2, m=2):
    
    N = len(y)
    y_pred = np.zeros(N)
    # max_dim = 
    y_pred[:n] = y[:n]  # Initialize the first n values with actual data

    # Simulate future steps using accumulated predictions
    for k in range(n, N-1):
        # Construct the regressor with past predictions an d input
        past_outputs = -y_pred[k-n:k]  # Use predicted outputs
        
        # if(k-m < 0):
        #     past_inputs = u[:k]
        
        past_inputs = u[k-m:k]  # Use actual control inputs

        # Create the regression vector for this step
        phi_k = np.concatenate((past_outputs, past_inputs))

        # Predict the next output using the identified model parameters
        if theta.ndim > 1:
            theta = theta.squeeze()
        y_pred[k] = np.dot(theta, phi_k)
        if(np.isnan(y_pred[k])):
            print("Nan value detected")
        elif(np.isinf(y_pred[k])):
            print("Inf value detected")
    
    return y_pred

# Load the data from CSV file
time, velocity, acceleration = load_data('./2024_competition_dataset/' + load_train_file_name)

start_idx_offset = 0
time = time[start_idx_offset:]
velocity = velocity[start_idx_offset:]
acceleration = acceleration[start_idx_offset:]

#set to 0 the first value of the time
time = time - time[0]

# interpolate the data with time step
cubic_velocity = CubicSpline(time, velocity)
cubic_acceleration = CubicSpline(time, acceleration)

time = np.arange(time[0], time[-1], Ts)
velocity = cubic_velocity(time)
acceleration = cubic_acceleration(time)


# Build the regression matrix (Phi) and the output vector (Y)
Phi, Y = build_regression_matrix(velocity, acceleration, n, m)

# Perform least squares to identify the model parameters (theta)
theta = identify_model(Phi, Y)
# theta = np.array([-2.3/1.32, 1/1.32, 0.02/1.32])

# Perform gradient descent to identify the model parameters (theta)
# gradient_theta = gradient_descent(Phi, Y, 0.001, 10000)
# gradient_theta, cost_history = stochastic_gradient_descent(Phi, Y, 0.0001, 10000, 524)
# theta = gradient_theta

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
# plt.plot(time, Y_pred_accumulated, label='Predicted Output (y_pred)', linestyle='-.')
plt.scatter(time, Y_pred_accumulated, label='Predicted Output (y_pred)', s=5, color='orange')
plt.xlabel('Time Step')
plt.ylabel('Velocity (m/s)')
plt.title('Actual vs Accumulated Predicted Velocity')
plt.legend() 

plt.figure()
plt.plot(time, acceleration, label='Acceleration')
plt.xlabel('Time Step')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration')
plt.legend()

# save the plot
plt.savefig(f'./2024_competition_dataset/result/velocity_prediction_{n}_{m}_{Ts}.png')

# plt.figure()
# plt.plot(cost_history)
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Cost History')

plt.show()


# print("A : ", theta[0:2])
# print("B :", theta[2:4])

# RMSE Calculation
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Calculate RMSE
rmse = calculate_rmse(Y, Y_pred)
print("Root Mean Squared Error (RMSE):", rmse)


#get validation data
time, velocity, acceleration = load_data('./2024_competition_dataset/' + load_valid_file_name)

start_idx_offset = 0
time = time[start_idx_offset:]
velocity = velocity[start_idx_offset:]
acceleration = acceleration[start_idx_offset:]

#set to 0 the first value of the time
time = time - time[0]

# interpolate the data with time step
cubic_velocity = CubicSpline(time, velocity)
cubic_acceleration = CubicSpline(time, acceleration)

time = np.arange(time[0], time[-1], test_Ts)
velocity = cubic_velocity(time)
acceleration = cubic_acceleration(time)

# Build the regression matrix (Phi) and the output vector (Y)
Phi, Y = build_regression_matrix(velocity, acceleration, n, m)

# predict the output using the identified model
Y_pred = predict_output(theta, Phi)

Y_pred_accumulated = accumulate_prediction(theta, velocity, acceleration, n, m)

# visualize the result
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
# plt.plot(time, Y_pred_accumulated, label='Predicted Output (y_pred)', linestyle='-.') 
plt.scatter(time, Y_pred_accumulated, label='Predicted Output (y_pred)', s=5, color='red')
plt.xlabel('Time Step')
plt.ylabel('Velocity (m/s)')
plt.title('Actual vs Accumulated Predicted Velocity')
plt.legend()

plt.savefig(f'./2024_competition_dataset/result/velocity_prediction_{n}_{m}_{test_Ts}_valid.png')

plt.show()

