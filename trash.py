import csv
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

filename = "tire_model_data.csv"

# Read the CSV file and store the data as a list of dictionaries
with open(filename, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    data = [row for row in reader]

# Skip the second and third lines
data = data[2:]

# Convert the list of dictionaries to a dictionary of lists
dic = {}
for key in data[0].keys():
    dic[key] = [float(row[key]) for row in data]

keys = ['F_fx', 'F_fy', 'F_fz', 'F_rx', 'F_ry', 'F_rz', 'delta' 'yaw', 'yaw_rate', 'lon_slip', 'slip_f', 'slip_r']
lf = 1.169  # [m]
lr = 1.801  # [m]
Iz = 3637.526  # [kg x m^2]
vx_list = dic['Car.vx']
vy_list = dic['Car.vy']
if(dic.get("DM.Gas")):
    gas_list = dic['DM.Gas']
elif(dic.get["Driver.Gas"]):
    gas_list = dic["Driver.Gas"]
real_result = {}
pred_result = {}

# 실제 차량 앞 타이어 힘 구하기
real_result['F_fx'] = [(FL + FR) / 2 for FL, FR in zip(dic['Car.CFL.Tire.FrcC.x'], dic['Car.CFR.Tire.FrcC.x'])]
real_result['F_fy'] = [(FL + FR) / 2 for FL, FR in zip(dic['Car.CFL.Tire.FrcC.y'], dic['Car.CFR.Tire.FrcC.y'])]
real_result['F_fz'] = [(FL + FR) / 2 for FL, FR in zip(dic['Car.CFL.Tire.FrcC.z'], dic['Car.CFR.Tire.FrcC.z'])]
real_result['F_rx'] = [(FL + FR) / 2 for FL, FR in zip(dic['Car.CRL.Tire.FrcC.x'], dic['Car.CRR.Tire.FrcC.x'])]
real_result['F_ry'] = [(FL + FR) / 2 for FL, FR in zip(dic['Car.CRL.Tire.FrcC.y'], dic['Car.CRR.Tire.FrcC.y'])]
real_result['F_rz'] = [(FL + FR) / 2 for FL, FR in zip(dic['Car.CRL.Tire.FrcC.z'], dic['Car.CRR.Tire.FrcC.z'])]

# 실제 차량 앞 타이어 회전각도 구하기
real_result['delta'] = [(FL + FR) / 2.0 for FL, FR in zip(dic['Vhcl.FL.rz'], dic['Vhcl.FR.rz'])]

# 실제 차량 yaw 값 구하기
real_result['yaw'] = [float(data) for data in dic['Car.Yaw']]

# 실제 차량 yaw rate 값 구하기
real_result['yaw_rate'] = [float(data) for data in dic['Car.YawRate']]

# 실제 차량 longitudinal slip 값 구하기
real_result['lon_slip'] = [(FL + FR) / 2 for FL, FR in zip(dic['Car.LongSlipRL'], dic['Car.LongSlipRR'])]

# 실제 차량 앞 바퀴 lateral slip 값 구하기
real_result['slip_f'] = [(FL + FR) / 2 for FL, FR in zip(dic['Car.SlipAngleFL'], dic['Car.SlipAngleFR'])]

# 실제 차량 뒷 바퀴 lateral slip 값 구하기
real_result['slip_r'] = [(FL + FR) / 2 for FL, FR in zip(dic['Car.SlipAngleRL'], dic['Car.SlipAngleRR'])]

# 차량 앞 바퀴 lateral slip 값 계산
# -std::atan2(x.vy+x.r*param_.lf,x.vx) + x.delta;
pred_result['slip_f'] = [math.atan2(vy + yaw_rate * lf, vx) + delta for vx, vy, yaw_rate, delta in
                         zip(vx_list, vy_list, dic['Car.YawRate'], real_result['delta'])]

# 차량 뒷 바퀴 lateral slip 값 계산
# -std::atan2(x.vy-x.r*param_.lr,x.vx);
pred_result['slip_r'] = [-math.atan2(vy - yaw_rate * lr, vx) for vx, vy, yaw_rate in
                         zip(vx_list, vy_list, dic['Car.YawRate'])]

# pred_result['lon_slip'] =
pred_result['F_fx'] = 0

# param_.Df * std::sin(param_.Cf * std::atan(param_.Bf * alpha_f ));
Df = 9000; Cf = 1; Bf = 1;
pred_result['F_fy'] = [Df * math.sin(Cf* math.atan(Bf*alpha_f)) for alpha_f in pred_result['slip_f']]

# ############
def predict_fy(alpha_f, Df, Cf, Bf):
    return Df * np.sin(Cf * np.arctan(Bf * alpha_f))

# Define the function to calculate the squared error
def squared_error(params, alpha_f, real_F_fy):
    Df, Cf, Bf = params
    predicted_F_fy = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
    return np.sum((real_F_fy - predicted_F_fy) ** 2)

# Define the gradient of the squared error function
def gradient_squared_error(params, alpha_f, real_F_fy):
    epsilon = 1e-8
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus_epsilon = params.copy()
        params_plus_epsilon[i] += epsilon
        grad[i] = (squared_error(params_plus_epsilon, alpha_f, real_F_fy) - squared_error(params, alpha_f, real_F_fy)) / epsilon
    return grad

# Gradient descent function
def gradient_descent(alpha_f, real_F_fy, learning_rate=1e-7, num_iterations=1000, initial_guess=[9000, 1, 1]):
    params = np.array(initial_guess, dtype=np.float64)  # Cast the initial guess as a float array
    for _ in range(num_iterations):
        gradient = gradient_squared_error(params, alpha_f, real_F_fy)
        params -= learning_rate * gradient
        print(params)
    return params

# Convert your real data into NumPy arrays
alpha_f = np.array(pred_result['slip_f'])
real_F_fy = np.array(real_result['F_fy'])

# Run gradient descent to find the optimal coefficients
optimal_params = gradient_descent(alpha_f, real_F_fy)
optimal_Df, optimal_Cf, optimal_Bf = optimal_params

# Print the optimal coefficients
print(f"Optimal Df: {optimal_Df}, Optimal Cf: {optimal_Cf}, Optimal Bf: {optimal_Bf}")

# Now you can use these optimal coefficients to predict F_fy
pred_result['F_fy'] = [predict_fy(alpha_f, optimal_Df, optimal_Cf, optimal_Bf) for alpha_f in pred_result['slip_f']]
# ###################

# pred_result['F_fz'] =


# param_.Dr * std::sin(param_.Cr * std::atan(param_.Br * alpha_r ));
Dr = 130000; Cr = 1; Br = 1
pred_result['F_rx'] = [Dr * math.sin(Cr * math.atan(Br* alpha_r)) for alpha_r in pred_result['slip_r']]

# param_.Cm1*x.D - param_.Cm2*x.D*x.vx;
Cm1 = 1; Cm2 = 1
pred_result['F_ry'] = [(Cm1 - Cm2 * vx) * throttle for vx, throttle in zip(vx_list, gas_list)]
# pred_result['F_rz'] =
# pred_result['yaw_rate'] =
plt.figure(1)
plt.title('F_fy Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('F_fy[N]')
# Plot the predicted results
plt.plot(dic['Time'], pred_result['F_fy'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['F_fy'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.figure(2)
plt.title('F_rx Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('F_rx[N]')
plt.plot(dic['Time'], pred_result['F_rx'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['F_rx'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')
# Display the graph
plt.show()


eval_keys = ['F_fy', 'F_rx']
for key in eval_keys:
    error_list = [abs(real - pred) for real, pred in zip(pred_result[key], real_result[key])]
    error_avg = sum(error_list) / len(error_list)
    print("avg", key ," error : ", error_avg)