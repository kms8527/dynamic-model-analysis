import csv
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

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


if(plot3D):
    # Create a new figure
    fig = plt.figure()

    # Add 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    ax_array = np.array(real_result["ax"])
    positive_ax = ax_array[ax_array > 0]
    positive_vx = np.array(vx_list)[ax_array > 0]
    positive_gas = np.array(gas_list)[ax_array > 0]

    idx_array = ax_array > 0

    # for i in range(len(ax_array)):
    #     y = -2/15 * (vx_list[i] - 30) + 2
    #     if (ax_array[i] < y):
    #         idx_array[i] = True

    # positive_ax = ax_array[idx_array]
    # positive_vx = np.array(vx_list)[idx_array]
    # positive_gas = np.array(gas_list)[idx_array]

    # convert to list
    positive_ax = positive_ax.tolist()
    positive_vx = positive_vx.tolist()
    positive_gas = positive_gas.tolist()
    # Scatter plot
    # ax.scatter(real_result["ax"], vx_list, gas_list, c='r', marker='o')
    ## 3D plot ##
    ax.scatter(positive_gas, positive_vx, positive_ax, c='r', marker='.')

    ax.set_xlabel('Throttle')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Acceleration')

    plt.show()

    #################
    ## 2D plot ##
    # plt.figure(figsize=(10, 6))
    # plt.scatter(positive_ax, positive_gas, color='blue')
    # plt.title("Velocity vs Throttle")
    # plt.xlabel("Velocity")
    # plt.ylabel("Throttle")
    # plt.grid(True)
    # plt.show()

# # Convert lists to numpy arrays
X1 = np.array(real_result["ax"])
X2 = np.array(vx_list)
Y = np.array(gas_list)

# Generate polynomial features
X1_2 = X1 ** 2
X2_2 = X2 ** 2
X1_X2 = X1 * X2

# Stack features into a matrix and add a constant for intercept
X_poly = np.column_stack((X1, X2, X1_2, X2_2, X1_X2))
X_poly = sm.add_constant(X_poly)

model = sm.OLS(Y, X_poly)
results = model.fit()

print(results.summary())

def poly_plane_func(accel, vel):
    return (results.params[0] +
            results.params[1] * accel +
            results.params[2] * vel +
            results.params[3] * accel**2 +
            results.params[4] * vel**2 +
            results.params[5] * accel * vel)

accel_grid, vel_grid = np.meshgrid(np.linspace(min(X1), max(X1), 100),
                                   np.linspace(min(X2), max(X2), 100))
throttle_grid = poly_plane_func(accel_grid, vel_grid)

# The rest of the plotting code remains largely the same


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of original data
ax.scatter(X1, X2, Y, c='r', marker='o', label='Data points')

# Surface plot of the regressed polynomial plane
ax.plot_surface(accel_grid, vel_grid, throttle_grid, alpha=0.5, cmap='viridis')

ax.set_xlabel('Acceleration')
ax.set_ylabel('Velocity')
ax.set_zlabel('Throttle')
ax.legend()

plt.show()
