import csv
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Example of using a different model (Random Forest)
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

filename = "accel_data.CSV"
plot3D = True

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

keys = ['F_fx', 'F_fy', 'F_fz', 'F_rx', 'F_ry', 'F_rz',
        'delta' 'yaw', 'yaw_rate', 'lon_slip', 'slip_f', 'slip_r', 'ax']
m = 2065.03  # [kg]
lf = 1.169  # [m]
lr = 1.801  # [m]
Iz = 3637.526  # [kg x m^2]
g = 9.81  # [m/s^2]
vx_list = dic['Car.vx']
# vy_list = dic['Car.vy']
# 실제 차량 타이어 z측 감쇄비 구하기
C = 406884.0  # [N/m]
C1 = C/2.0  # [N/m]
C2 = C/4.0  # [N/m]
K = 4608.6  # [N/m]
if (dic.get("VC.Gas")):
    gas_list = dic['VC.Gas']
elif (dic.get["Driver.Gas"]):
    gas_list = dic["Driver.Gas"]
real_result = {}
pred_result = {}

# 실제 차량 앞 타이어 힘 구하기
# real_result['F_fx'] = [
#     (FL + FR) for FL, FR in zip(dic['Car.CFL.Tire.FrcC.x'], dic['Car.CFR.Tire.FrcC.x'])]
# real_result['F_fy'] = [
#     (FL + FR) for FL, FR in zip(dic['Car.CFL.Tire.FrcC.y'], dic['Car.CFR.Tire.FrcC.y'])]
# real_result['F_fz'] = [
#     (FL + FR) for FL, FR in zip(dic['Car.CFL.Tire.FrcC.z'], dic['Car.CFR.Tire.FrcC.z'])]
# real_result['F_rx'] = [
#     (FL + FR) for FL, FR in zip(dic['Car.CRL.Tire.FrcC.x'], dic['Car.CRR.Tire.FrcC.x'])]
# real_result['F_ry'] = [
#     (FL + FR) for FL, FR in zip(dic['Car.CRL.Tire.FrcC.y'], dic['Car.CRR.Tire.FrcC.y'])]
# real_result['F_rz'] = [
#     (FL + FR) for FL, FR in zip(dic['Car.CRL.Tire.FrcC.z'], dic['Car.CRR.Tire.FrcC.z'])]
# real_result['tire_w'] = [(W_RL + W_RR)/2 for W_RL,
#                          W_RR in zip(dic['Vhcl.RL.rotv'], dic['Vhcl.RR.rotv'])]


# # 실제 차량 앞 타이어 회전각도 구하기
# real_result['delta'] = [(FL + FR) / 2.0 for FL,
#                         FR in zip(dic['Vhcl.FL.rz'], dic['Vhcl.FR.rz'])]

# # 실제 차량 yaw 값 구하기
# real_result['yaw'] = [float(data) for data in dic['Car.Yaw']]

# # 실제 차량 yaw rate 값 구하기
# real_result['yaw_rate'] = [float(data) for data in dic['Car.YawRate']]

# # 실제 차량 x_dot 값 구하기
# real_result['x_dot'] = [vx * math.cos(yaw) - vy * math.sin(yaw)
#                         for vx, vy, yaw in zip(vx_list, vy_list, real_result['yaw'])]

# # 실제 차량 y_dot 값 구하기
# real_result['y_dot'] = [vx * math.sin(yaw) + vy * math.cos(yaw)
#                         for vx, vy, yaw in zip(vx_list, vy_list, real_result['yaw'])]

# # 실제 차량 longitudinal slip 값 구하기
# real_result['lon_slip'] = [
#     (FL + FR) / 2 for FL, FR in zip(dic['Car.LongSlipRL'], dic['Car.LongSlipRR'])]

# # 실제 차량 앞 바퀴 lateral slip 값 구하기
# real_result['slip_f'] = [
#     (FL + FR) / 2 for FL, FR in zip(dic['Car.SlipAngleFL'], dic['Car.SlipAngleFR'])]

# # 실제 차량 뒷 바퀴 lateral slip 값 구하기
# real_result['slip_r'] = [
#     (FL + FR) / 2 for FL, FR in zip(dic['Car.SlipAngleRL'], dic['Car.SlipAngleRR'])]

real_result['ax'] = dic['Car.ax']
# real_result['ay'] = dic['Car.ay']

# real_result['F_x'] = [(F_rx + F_fx) for F_rx,
#                       F_fx in zip(real_result['F_rx'], real_result['F_fx'])]


# real_result['F_z'] = [(F_rz + F_fz) for F_rz,
#                       F_fz in zip(real_result['F_rz'], real_result['F_fz'])]

# real_result['yaw_acc'] = dic['Car.YawAcc']


data = pd.DataFrame({
    'accel': real_result['ax'][:-1],
    'velocity': vx_list[:-1],
    'throttle': gas_list[:-1]
})


import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(data)
plt.show()
import time
data['target_accel'] = data['accel'].shift(-1)
data = data.dropna()

# Split your data into training and testing sets
X = data[['accel', 'velocity', 'target_accel']]
y = data['throttle']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)

# Train a simple linear regression model
model = RandomForestRegressor(n_estimators=5, max_depth=5)
model.fit(X_train, y_train)

# Evaluate the model using the test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')

# Making a prediction with new data
current_accel = 0.1  # replace with your actual value
velocity = 24  # replace with your actual value
target_accel = 0.1  # replace with your actual value
start_time = time.time()

output_throttle = model.predict([[current_accel, velocity, target_accel]])
print(f'Output Throttle: {output_throttle[0]}') 

curr_time = time.time()
print("time: ", curr_time - start_time)

save = False
if(save):
    # Convert the model to ONNX format
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save the ONNX model to disk
    with open("random_forest.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())