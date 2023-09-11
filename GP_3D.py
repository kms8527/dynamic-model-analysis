import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import csv
import math
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

filename = "tire_model_data_v4.csv"
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
vy_list = dic['Car.vy']
# 실제 차량 타이어 z측 감쇄비 구하기
C0 = 406884.0  # [N/m]
C1 = C0/2.0  # [N/m]
C2 = C0/4.0  # [N/m]
K = 4608.6  # [N/m]
if (dic.get("DM.Gas")):
    gas_list = dic['DM.Gas']
elif (dic.get["Driver.Gas"]):
    gas_list = dic["Driver.Gas"]
real_result = {}
pred_result = {}

# 실제 차량 앞 타이어 힘 구하기
real_result['F_fx'] = [
    (FL + FR) for FL, FR in zip(dic['Car.CFL.Tire.FrcC.x'], dic['Car.CFR.Tire.FrcC.x'])]
real_result['F_fy'] = [
    (FL + FR) for FL, FR in zip(dic['Car.CFL.Tire.FrcC.y'], dic['Car.CFR.Tire.FrcC.y'])]
real_result['F_fz'] = [
    (FL + FR) for FL, FR in zip(dic['Car.CFL.Tire.FrcC.z'], dic['Car.CFR.Tire.FrcC.z'])]
real_result['F_rx'] = [
    (FL + FR) for FL, FR in zip(dic['Car.CRL.Tire.FrcC.x'], dic['Car.CRR.Tire.FrcC.x'])]
real_result['F_ry'] = [
    (FL + FR) for FL, FR in zip(dic['Car.CRL.Tire.FrcC.y'], dic['Car.CRR.Tire.FrcC.y'])]
real_result['F_rz'] = [
    (FL + FR) for FL, FR in zip(dic['Car.CRL.Tire.FrcC.z'], dic['Car.CRR.Tire.FrcC.z'])]
real_result['tire_w'] = [(W_RL + W_RR)/2 for W_RL,
                         W_RR in zip(dic['Vhcl.RL.rotv'], dic['Vhcl.RR.rotv'])]


# 실제 차량 앞 타이어 회전각도 구하기
real_result['delta'] = [(FL + FR) / 2.0 for FL,
                        FR in zip(dic['Vhcl.FL.rz'], dic['Vhcl.FR.rz'])]

# 실제 차량 yaw 값 구하기
real_result['yaw'] = [float(data) for data in dic['Car.Yaw']]

# 실제 차량 yaw rate 값 구하기
real_result['yaw_rate'] = [float(data) for data in dic['Car.YawRate']]

# 실제 차량 x_dot 값 구하기
real_result['x_dot'] = [vx * math.cos(yaw) - vy * math.sin(yaw)
                        for vx, vy, yaw in zip(vx_list, vy_list, real_result['yaw'])]

# 실제 차량 y_dot 값 구하기
real_result['y_dot'] = [vx * math.sin(yaw) + vy * math.cos(yaw)
                        for vx, vy, yaw in zip(vx_list, vy_list, real_result['yaw'])]

# 실제 차량 longitudinal slip 값 구하기
real_result['lon_slip'] = [
    (FL + FR) / 2 for FL, FR in zip(dic['Car.LongSlipRL'], dic['Car.LongSlipRR'])]

# 실제 차량 앞 바퀴 lateral slip 값 구하기
real_result['slip_f'] = [
    (FL + FR) / 2 for FL, FR in zip(dic['Car.SlipAngleFL'], dic['Car.SlipAngleFR'])]

# 실제 차량 뒷 바퀴 lateral slip 값 구하기
real_result['slip_r'] = [
    (FL + FR) / 2 for FL, FR in zip(dic['Car.SlipAngleRL'], dic['Car.SlipAngleRR'])]

real_result['ax'] = dic['Car.ax']
real_result['ay'] = dic['Car.ay']

real_result['F_x'] = [(F_rx + F_fx) for F_rx,
                      F_fx in zip(real_result['F_rx'], real_result['F_fx'])]


real_result['F_z'] = [(F_rz + F_fz) for F_rz,
                      F_fz in zip(real_result['F_rz'], real_result['F_fz'])]

# real_result['yaw_acc'] = dic['Car.YawAcc']


# 랜덤 2D 데이터 생성
ax_array = np.array(real_result["ax"])
positive_ax1 = ax_array[ax_array > 0]
positive_vx1 = np.array(vx_list)[ax_array > 0]
positive_gas1 = np.array(gas_list)[ax_array > 0]

positive_ax_list = []
positive_vx_list = []
positive_gas_list = []
for i in range(0,len(positive_ax1),100):
    positive_ax_list.append(positive_ax1[i])
    positive_vx_list.append(positive_vx1[i])
    positive_gas_list.append(positive_gas1[i])
    
positive_ax = np.array(positive_ax_list)
positive_vx = np.array(positive_vx_list)
positive_gas = np.array(positive_gas_list)

train_X = np.array(list(map(np.array, np.stack((positive_ax, positive_vx), axis=-1))))

# train_X = np.random.rand(100, 2)
# train_y = np.sin(5 * train_X[:, 0]) * np.cos(5 * train_X[:, 1])
train_y = np.array(positive_gas)

# Gaussian Process 모델 학습
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr.fit(train_X, train_y)

# 격자로 예측
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
x, y = np.meshgrid(x, y)
test_X = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
predicted_y = gpr.predict(test_X)

# 3D 플롯 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_X[:, 0], train_X[:, 1], train_y,
           c='r', marker='o')  # training data
ax.plot_surface(x, y, predicted_y.reshape(10, 10), alpha=0.2)  # GPR surface
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()

