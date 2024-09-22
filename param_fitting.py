import csv
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

filename = "tire_model_data_v5.csv"

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

keys = ['time', 'F_fx', 'F_fy', 'F_fz', 'F_rx', 'F_ry', 'F_rz',
        'delta' 'yaw', 'yaw_rate', 'lon_slip', 'slip_f', 'slip_r', 'ax']
m = 2065.03  # [kg]
lf = 1.169  # [m]
lr = 1.801  # [m]
Iz = 3637.526  # [kg x m^2]
g = 9.81  # [m/s^2]
vx_list = dic['Car.vx']
vy_list = dic['Car.vy']
# 실제 차량 타이어 z측 감쇄비 구하기
C = 406884.0  # [N/m]
C1 = C/2.0  # [N/m]
C2 = C/4.0  # [N/m]
K = 4608.6  # [N/m]
if (dic.get("DM.Gas")):
    gas_list = dic['DM.Gas']
elif (dic.get["Driver.Gas"]):
    gas_list = dic["Driver.Gas"]
real_result = {}
pred_result = {}

# real_result['time'] = dic['time']
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

real_result['yaw_acc'] = dic['Car.YawAcc']


def pacejka_model(alpha_f, Df, Cf, Bf):
    return -Df * np.sin(Cf * np.arctan(Bf * alpha_f))

# Example data; replace these with your actual data
initial_guess = [100000.0, 1.0, 1.0] # Df, Cf, Bf

# Combine and sort the data using the zip function
combined_data = sorted(zip(real_result['slip_f'],  real_result['F_fy']))

# Unzip the combined data into separate lists
alpha_f_data_sorted, F_fy_data_sorted = zip(*combined_data)
alpha_f_data_sorted = np.array([alpha_f_data_sorted])
F_fy_data_sorted = np.array([F_fy_data_sorted])

alpha_f_data_sorted = np.ravel(alpha_f_data_sorted)
F_fy_data_sorted = np.ravel(F_fy_data_sorted)

# Find the optimal parameters Df, Cf, and Bf
popt, _ = curve_fit(pacejka_model, alpha_f_data_sorted, F_fy_data_sorted, p0=initial_guess, maxfev=100000)

Df_opt, Cf_opt, Bf_opt = popt

print(f"Optimal parameters: Df={Df_opt:.4f}, Cf={Cf_opt:.4f}, Bf={Bf_opt:.4f}")

# Generate fitted F_fy data
fitted_F_fy = pacejka_model(alpha_f_data_sorted, Df_opt, Cf_opt, Bf_opt)

# Plot the original data points
plt.scatter(alpha_f_data_sorted, F_fy_data_sorted, label="Original data")

# Plot the fitted model
plt.plot(alpha_f_data_sorted, fitted_F_fy, linestyle="--", color="red", label="Fitted model")

plt.xlabel("alpha_f")
plt.ylabel("F_fy")
plt.title("Pacejka Model Fitting")
plt.legend()
plt.show()


# Example data; replace these with your actual data
initial_guess = [5000.0, 1.0, 1.0] # Df, Cf, Bf

# Combine and sort the data using the zip function
combined_data = sorted(zip(real_result['slip_r'],  real_result['F_ry']))

# Unzip the combined data into separate lists
alpha_r_data_sorted, F_fy_data_sorted = zip(*combined_data)
alpha_r_data_sorted = np.array([alpha_r_data_sorted])
F_fy_data_sorted = np.array([F_fy_data_sorted])

alpha_r_data_sorted = np.ravel(alpha_r_data_sorted)
F_fy_data_sorted = np.ravel(F_fy_data_sorted)


# Find the optimal parameters Df, Cf, and Bf
popt, _ = curve_fit(pacejka_model, alpha_r_data_sorted, F_fy_data_sorted, p0=initial_guess, maxfev=100000)

Df_opt, Cf_opt, Bf_opt = popt

print(f"Optimal parameters: Df={Df_opt:.4f}, Cf={Cf_opt:.4f}, Bf={Bf_opt:.4f}")

# Generate fitted F_fy data
fitted_F_fy = pacejka_model(alpha_r_data_sorted, Df_opt, Cf_opt, Bf_opt)

# Plot the original data points
plt.scatter(alpha_r_data_sorted, F_fy_data_sorted, label="Original data")

# Plot the fitted model
plt.plot(alpha_r_data_sorted, fitted_F_fy, linestyle="--", color="red", label="Fitted model")

plt.xlabel("alpha_r")
plt.ylabel("F_ry")
plt.title("Pacejka Model Fitting")
plt.legend()
plt.show()

initial_guess = [28700,4550,300,10] # Cm1, Cm2, Cr_rx, Cr_rx

# Combine and sort the data using the zip function
combined_data = sorted(zip(real_result['lon_slip'],  gas_list, vx_list))


# Unzip the combined data into separate lists
lon_slip_sorted, gas_sorted, vx_sorted = zip(*combined_data)
lon_slip_sorted = np.array([lon_slip_sorted])
gas_sorted = np.array([gas_sorted])
vx_sorted = np.array([vx_sorted])


lon_slip_sorted = np.ravel(lon_slip_sorted)
gas_sorted = np.ravel(gas_sorted)
vx_sorted = np.ravel(vx_sorted)

def calc_Fx(x, Cm1, Cm2, Cr_rx, Cd_rx):
    lon_slip, throttle, vx = x
    # return Cm1 * lon_slip + (Cm2 - Cr_rx * vx) * throttle - Cd_rx * vx ** 2
    return (Cm1 - Cm2 * vx) * throttle - Cr_rx - Cd_rx * vx ** 2

# Prepare combined independent data
x_data = np.vstack((lon_slip_sorted, gas_sorted, vx_sorted))

# Find the optimal parameters
popt, _ = curve_fit(calc_Fx, x_data, gas_sorted, p0=initial_guess, maxfev=10000)

Cm1_opt, Cm2_opt, Cr_rx_opt, Cd_rx_opt = popt

print(f"Optimal parameters: Cm1={Cm1_opt:.4f}, Cm2={Cm2_opt:.4f}, Cr_rx={Cr_rx_opt:.4f}, Cd_rx={Cd_rx_opt:.4f}")

# Generate fitted F_x data
fitted_F_x = calc_Fx(x_data, *popt)
# Plot the original data points
plt.scatter(dic['Time'], real_result['F_x'], label="Original data")

# Plot the fitted model
plt.plot(lon_slip_sorted, fitted_F_x, linestyle="--", color="red", label="Fitted model")

plt.xlabel("lon_slip_sorted")
plt.ylabel("F_x")
plt.title("F_x Fitting")
plt.legend()
plt.show()
