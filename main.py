import csv
import math
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
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

# 차량 앞 바퀴 lateral slip 값 계산
# -std::atan2(x.vy+x.r*param_.lf,x.vx) + x.delta;
pred_result['slip_f'] = []
for vx, vy, yaw_rate, delta in zip(vx_list, vy_list, dic['Car.YawRate'], real_result['delta']):
    if (vx < 1):
        pred_result['slip_f'].append(0)
    else:
        pred_result['slip_f'].append(
            math.atan2(vy + yaw_rate * lf, vx) - delta)


# 차량 뒷 바퀴 lateral slip 값 계산
# -std::atan2(x.vy-x.r*param_.lr,x.vx);
pred_result['slip_r'] = []
for vx, vy, yaw_rate in zip(vx_list, vy_list, dic['Car.YawRate']):
    if (vx < 1):
        pred_result['slip_r'].append(0)
    else:
        pred_result['slip_r'].append(math.atan2(vy - yaw_rate * lr, vx))

        # pred_result['lon_slip'] =
pred_result['F_fx'] = 0

# param_.Df * std::sin(param_.Cf * std::atan(param_.Bf * alpha_f ));
Df = 9873.3928
Cf = 1.4754
Bf = 15.8359
pred_result['F_fy'] = [-Df * math.sin(Cf * math.atan(Bf*alpha_f))
                       for alpha_f in pred_result['slip_f']]
# param_.Dr * std::sin(param_.Cr * std::atan(param_.Br * alpha_r );
Dr = 418852.4447
C_ry = 0.0115
B_ry = 42.2619
pred_result['F_ry'] = [-Dr * math.sin(C_ry * math.atan(B_ry * alpha_r))
                       for alpha_r in pred_result['slip_r']]


# Define the function to calculate the pred_result based on Cr and Cd
# def calculate_pred_result(Cr, Cd, real_result, vy_list):
#     pred_result = [
#         1 / m * (
#             F_rx - Cr - Cd * vx ** 2 - F_fy * math.sin(delta) + m * vy * r
#         ) for F_rx, F_fy, delta, vy, r in zip(
#             real_result['F_rx'],
#             real_result['F_fy'],
#             real_result['delta'],
#             vy_list,
#             real_result['yaw_rate']
#         )
#     ]
#     return pred_result

# # Initialize the search parameters and error
# min_error = float('inf')
# best_coefficients = (None, None)

# # Perform the full search
# for Cr in np.arange(500, 1000.001, 100):
#     for Cd in np.arange(-100000, -50000.001, 100):
#         pred_result = calculate_pred_result(Cr, Cd, real_result, vy_list)
#         error = np.sum((np.array(real_result['ax']) - np.array(pred_result)) ** 2)

#         if error < min_error:
#             min_error = error
#             best_coefficients = (Cr, Cd)
#             print("Cr : {}, Cd : {}, error : {}".format(Cr, Cd, error))

# # Output the best coefficients
# print(f"Best coefficients: Cr = {best_coefficients[0]}, Cd = {best_coefficients[1]}")
# ################
# Cr_best, Cd_best = best_coefficients
# pred_result_ax = calculate_pred_result(Cr_best, Cd_best, real_result, vy_list)
# pred_result['ax'] = [1 / m * (F_rx - Cr - Cd * vx ** 2 - F_fy * math.sin(delta) + m * vy * r) for F_rx, F_fy, delta, vy, r, m, vx in zip(real_result['F_rx'], real_result['F_fy'], real_result['delta'], vy_list, real_result['yaw_rate'], real_result['m'], real_result['vx'])]


d_delta_list = []
d_delta_list.append(0)
for i in range(1, len(real_result['delta'])):
    d_delta_list.append(
        (real_result['delta'][i] - real_result['delta'][i-1]))

# kinematic
# pred_result['yaw_rate'] = [vx * math.cos((lr*math.tan(steer)/(lf+lr)))/(
#     lf+lr) * math.tan(steer) for vx, steer in zip(vx_list, real_result['delta'])]

# dynamic
# pred_result['yaw_rate'] = [1/Iz * (F_fy * lf * math.cos(steer) - F_ry*lr) for steer, F_fy, F_ry in zip(real_result['delta'],real_result['F_fy'], real_result['F_ry'])]

# 상수
pred_result['F_fz'] = [m * g * lr/(lr+lf) for i in range(len(dic['Car.vx']))]
pred_result['F_rz'] = [m * g * lf/(lr+lf) for i in range(len(dic['Car.vx']))]

pred_result['F_z'] = [(F_rz + F_fz) for F_rz,
                      F_fz in zip(pred_result['F_rz'], pred_result['F_fz'])]


# pred_result['F_fz'] = [2/20*0.5 *(m*az + C *vz + K *0) * lr/(lf+lr) for az, vz in zip(dic['Car.az'],dic['Car.vz'])]
# pred_result['F_rz'] = [1/10*0.5 *(m*az + C *vz + K * 0) for az, vz in zip(dic['Car.az'],dic['Car.vz'])]#[1/2*m * g * lf/(lr+lf) for i in range(len(dic['Car.vx']))]

# param_.Cm1*x.D - param_.Cm2*x.D*x.vx;
# ㅗ
Cm1 = 100
# ㅗㅗ
Cm2 = 5
# 공기저항 상수
Cr_rx = 280
# 공기저항 속도제곱비례 상수
Cd_rx = 0

real_reff = []
rw = 0.321  # wheel radius
for Fz in real_result['F_rz']:
    if (Fz == 0):
        real_reff.append(0)
    else:
        real_reff.append(rw * math.sin(math.acos((rw - Fz/C)/rw)) /
                         math.acos((rw - Fz/C)/rw))

pred_reff = []
for Fz in pred_result['F_rz']:
    Fz = 0.5*Fz
    if (Fz == 0):
        pred_reff.append(0)
    else:
        pred_reff.append(rw * math.sin(math.acos((rw - Fz/C)/rw)) /
                         math.acos((rw - Fz/C)/rw))

pred_result['lon_slip'] = []
for Reff, W_w, vx in zip(pred_reff, real_result['tire_w'], vx_list):
    lon_slip_bias = 0
    pred_result['lon_slip'].append(
        (Reff*W_w - vx) / (vx + lon_slip_bias) + 0.01)
    # if(vx < 2):
    #     lon_slip_bias = 0
    #     pred_result['lon_slip'].append((Reff*W_w- vx) / (vx +lon_slip_bias))
    # else:
    #     lon_slip_bias = 0
    #     pred_result['lon_slip'].append((Reff*W_w- vx) / (vx +lon_slip_bias) + 0.01)

    # if ( vx < 2):
    #     pred_result['lon_slip'].append(0)
    # else:
    #     lon_slip_bias = 0.011
    #     pred_result['lon_slip'].append((Reff*W_w- vx) / vx +lon_slip_bias )


def calc_Fx(lon_slip, throttle, vx):
    # pred_aceel = (Cm1 - Cm2 * vx) * throttle
    return C*lon_slip + (Cm1 - Cm2 * vx) * throttle - Cr_rx - Cd_rx * vx ** 2
    # if lon_slip < 0:
    #     return 2*C*lon_slip + (Cm1 - Cm2 * vx) * throttle - Cr_rx - Cd_rx * vx ** 2
    # else:
    #     return C*lon_slip + (Cm1 - Cm2 * vx) * throttle - Cr_rx - Cd_rx * vx ** 2


pred_result['F_x'] = [
    calc_Fx(*args) for args in zip(pred_result['lon_slip'], gas_list, vx_list)]
# pred_result['F_x'] = [
    # 0 for args in zip(pred_result['lon_slip'], gas_list, vx_list)]


pred_result['yaw_acc'] = [1/Iz*(lf*F_fy*math.cos(delta) - lr*F_ry) for F_fy, F_ry,
                          delta in zip(pred_result['F_fy'], pred_result['F_ry'], real_result['delta'])]
curr_yaw_rate = real_result['yaw_rate'][0]
pred_result['yaw_rate'] = []
for dt, yaw_acc in zip(dic['Time'], pred_result['yaw_acc']):
    if dt == 0:
        pred_result['yaw_rate'].append(curr_yaw_rate)
    else:
        curr_yaw_rate += yaw_acc * dt
        pred_result['yaw_rate'].append(curr_yaw_rate)


# kinematic
# pred_result['ax'] = [2*(F_rx + F_fx)/(m) for F_fx,
        #  F_rx in zip(real_result['F_fx'], real_result['F_rx'])]

# dynamic
#  2.0/param_.m*(tire_forces_rear.F_x + friction_force - tire_forces_front.F_y*std::sin(delta) + param_.m*vy*r);
Cr0 = 500
Cr20 = -100000
friction_force = - Cr0 - Cr20*vx**2
pred_result['ax'] = [1/m * (F_x + friction_force - F_fy * math.sin(delta) + m*vy*r) for F_x, F_fy, delta, vy,
                     r in zip(pred_result['F_x'], pred_result['F_fy'], real_result['delta'], vy_list, pred_result['yaw_rate'])]

# kinematic
# pred_result['ay'] = [ (d_steer*vx + steer * ax) * lr/(lr+lf) for vx, ax, steer, d_steer in zip(dic['Car.vx'],pred_result['ax'],real_result['delta'],d_delta_list)]

# dynamic
pred_result['ay'] = [1/m*(m*vx*yaw_rate) for vx, F_fy, F_ry, delta, yaw_rate in zip(
    dic['Car.vx'], pred_result['F_fy'], pred_result['F_ry'], real_result['delta'], pred_result['yaw_rate'])]


plt.figure(1)
plt.subplot(2, 1, 1)
plt.title('F_fy Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('F_fy[N]')
# Plot the predicted results
plt.plot(dic['Time'], pred_result['F_fy'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['F_fy'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.title('F_fy Comparison')
plt.xlabel('slip_f[rad]')
plt.ylabel('F_fy[N]')
# Plot the predicted results
plt.plot(pred_result['slip_f'], pred_result['F_fy'],
         marker=',', label='Predicted')
# Plot the real results
plt.plot(real_result['slip_f'], real_result['F_fy'], marker=",", label='Real')
# Add a legend to the graph
plt.legend(loc='best')
plt.tight_layout()  # 두 subplot graph간 간격 적절히 조정

# plt.figure(2)
# plt.subplot(2, 1, 1)
# plt.title('front slip angle Comparison')
# plt.xlabel('Time[sec]')
# plt.ylabel('slip angel[rad]')
# # Plot the predicted results
# plt.plot(dic['Time'], pred_result['slip_f'], marker=",", label='Predicted')
# # Plot the real results
# plt.plot(dic['Time'], real_result['slip_f'], marker=",", label='Real')
# # Add a legend to the graph
# plt.legend(loc='best')

# plt.subplot(2, 1, 2)
# plt.title('rear slip angle Comparison')
# plt.xlabel('Time[sec]')
# plt.ylabel('slip angel[rad]')
# # Plot the predicted results
# plt.plot(dic['Time'], pred_result['slip_r'], marker=",", label='Predicted')
# # Plot the real results
# plt.plot(dic['Time'], real_result['slip_r'], marker=",", label='Real')
# # Add a legend to the graph
# plt.legend(loc='best')
# plt.tight_layout()  # 두 subplot graph간 간격 적절히 조정

# plt.figure(3)
# plt.title('front slip angle Comparison')
# plt.xlabel('pred slip angle[rad]')
# plt.ylabel('pred slip angel[rad]')
# # Plot the predicted results
# plt.plot(pred_result['slip_f'], pred_result['slip_f'],
#          marker=",", label='Predicted')
# # Plot the real results
# plt.plot(pred_result['slip_f'], real_result['slip_f'],
#          marker=",", label='Real')
# # Add a legend to the graph
# plt.legend(loc='best')

plt.figure(4)
plt.subplot(2, 1, 1)
plt.title('yaw_rate Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('yaw rate[rad/s]')
# Plot the predicted results
plt.plot(dic['Time'][:1000], pred_result['yaw_rate']
         [:1000], marker=",", label='Predicted')
# Plot the real results
plt.plot(dic['Time'][:1000], real_result['yaw_rate']
         [:1000], marker=",", label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.title('yaw_rate Comparison')
plt.xlabel('wheel_delta[rad]')
plt.ylabel('yaw rate[rad/s]')
# Plot the predicted results
plt.plot(real_result['delta'], pred_result['yaw_rate'],
         marker=",", label='Predicted')
# Plot the real results
plt.plot(real_result['delta'], real_result['yaw_rate'],
         marker=",", label='Real')
# Add a legend to the graph
plt.legend(loc='best')
plt.tight_layout()  # 두 subplot graph간 간격 적절히 조정

plt.figure(5)
plt.subplot(2, 1, 1)
plt.title('F_ry Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('F_ry[N]')
# Plot the predicted results
plt.plot(dic['Time'], pred_result['F_ry'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['F_ry'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.title('F_ry Comparison')
plt.xlabel('slip_f[rad]')
plt.ylabel('F_ry[N]')
# Plot the predicted results
plt.plot(real_result['slip_f'], pred_result['F_ry'], ',', label='Predicted')
# Plot the real results
plt.plot(real_result['slip_f'], real_result['F_ry'], marker=",", label='Real')
# Add a legend to the graph
plt.legend(loc='best')
plt.tight_layout()  # 두 subplot graph간 간격 적절히 조정

plt.figure(6)
plt.subplot(2, 1, 1)
plt.title('F_x Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('F_x[N]')
plt.plot(dic['Time'], pred_result['F_x'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['F_x'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')
# Display the graph

plt.subplot(2, 1, 2)
plt.title('F_x Comparison')
plt.xlabel('lon_slip')

plt.ylabel('F_x[N]')
plt.plot(real_result['lon_slip'], pred_result['F_x'], ',', label='Predicted')
# Plot the real results
plt.plot(real_result['lon_slip'], real_result['F_x'], ',', label='Real')
# Add a legend to the graph
plt.legend(loc='best')
plt.tight_layout()  # 두 subplot graph간 간격 적절히 조정

plt.figure(7)
plt.subplot(2, 1, 1)
plt.title('ax Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('ax[m/s^2]')
plt.plot(dic['Time'], pred_result['ax'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['ax'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.title('ay Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('ay[m/s^2]')
plt.plot(dic['Time'], pred_result['ay'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['ay'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')
plt.tight_layout()  # 두 subplot graph간 간격 적절히 조정


plt.figure(8)
plt.subplot(3, 1, 1)
plt.title('F_fz Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('F_fz[N]')
plt.plot(dic['Time'], pred_result['F_fz'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['F_fz'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.subplot(3, 1, 2)
plt.title('F_rz Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('F_rz[N]')
plt.plot(dic['Time'], pred_result['F_rz'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['F_rz'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.subplot(3, 1, 3)
plt.title('total F_z')
plt.xlabel('Time[sec]')
plt.ylabel('F_z[N]')
plt.plot(dic['Time'], pred_result['F_z'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['F_z'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')


plt.figure(9)
plt.title('reff Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('slip[N]')
plt.plot(dic['Time'], pred_reff, label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_reff, label='Real')
# Add a legend to the graph
plt.legend(loc='best')
# Display the graph

plt.figure(10)
plt.title('lon slip Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('slip[N]')
plt.plot(dic['Time'], pred_result['lon_slip'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['lon_slip'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')
# Display the graph

plt.figure(11)
plt.title('yaw acc Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('yaw acc[rad/s^2]')
plt.plot(dic['Time'], pred_result['yaw_acc'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['yaw_acc'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')
# Display the graph

# plt.figure(11)
# plt.title('yaw acc Comparison')
# plt.xlabel('Time[sec]')
# plt.ylabel('yaw acc[rad/s^2]')
# plt.plot(dic['Time'], pred_result['yaw_acc'], label='Predicted')
# # Plot the real results
# plt.plot(dic['Time'], real_result['yaw_acc'], label='Real')
# # Add a legend to the graph
# plt.legend(loc='best')
# # Display the graph

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(vx_list, real_result['ay'], real_result['ax'])

# # ax.set_xlabel('vx_list')
# # ax.set_ylabel('real_result[\'ay\']')
# # ax.set_zlabel('real_result[\'ax\']')

plt.figure(12)
plt.title('ax vs v')
plt.xlabel('v[m/sec]')
plt.ylabel('ax[rad/s^2]')
plt.plot(vx_list, real_result['ax'])
# Add a legend to the graph

plt.figure(13)
plt.title('ay vs v')
plt.xlabel('v[m/sec]')
plt.ylabel('ay[rad/s^2]')
plt.plot(vx_list, real_result['ay'])
# Add a legend to the graph

plt.figure(14)
plt.title('ax vs ay')
plt.xlabel('ax[m/sec^2]')
plt.ylabel('ay[m/s^2]')
plt.plot(real_result['ax'], real_result['ay'])
# Add a legend to the graph

plt.show()

# eval_keys = ['F_fy', 'F_x', 'slip_f',
#              'slip_r', 'yaw_rate', 'F_ry', 'ax', 'ay', 'F_z', 'yaw_acc']


# for key in eval_keys:
#     error_list = [abs(real - pred)
#                   for real, pred in zip(pred_result[key], real_result[key])]
#     error_avg = sum(error_list) / len(error_list)
#     print("avg", key, " error : ", error_avg)


# ##ggv Digaram drawing

# for i in range(0, 7):
#     vx1 = 5*i
#     vx2 = 5*(i+1)
#     indices = [i for i, value in enumerate(vx_list) if vx1 <= value <= vx2]
#     vx_list_new = [vx_list[i] for i in indices]
#     ax_list_new = [real_result['ax'][i] for i in indices]
#     ay_list_new = [real_result['ay'][i] for i in indices]

#     plt.figure(i)
#     plt.title(str(vx1) + ' ~ ' + str(vx2) + 'm/s')
#     plt.xlabel('ay[m/sec^2]')
#     plt.ylabel('ax[m/s^2]')
#     plt.plot(ay_list_new, ax_list_new, '.')

# vx1 = 0
# vx2 = 50
# indices = [i for i, value in enumerate(vx_list) if vx1 <= value <= vx2]
# vx_list_new = [vx_list[i] for i in indices]
# ax_list_new = [real_result['ax'][i] for i in indices]
# ay_list_new = [real_result['ay'][i] for i in indices]

# plt.figure(7)
# plt.title(str(vx1) + ' ~ ' + str(vx2) + 'm/s')
# plt.xlabel('ay[m/sec^2]')
# plt.ylabel('ax[m/s^2]')
# plt.plot(ay_list_new, ax_list_new, '.')

# Filter the data based on constraints ax > 0 and ay < 0
# ax = np.array(real_result['ax'])
# ay = np.array(real_result['ay'])
# indices = np.where((ax > 0) & (ay < 0) & (np.array(vx_list) > 10) & (np.array(vx_list) < 15))
# filtered_ax = ax[indices]
# filtered_ay = ay[indices]

# # Define the ellipse function
# def ellipse(x, a, b, h, k):
#     return k + (b * np.sqrt(1 - ((x - h) ** 2) / (a ** 2)))
# from scipy.optimize import curve_fit

# # Fit the ellipse to the data
# params, _ = curve_fit(ellipse, filtered_ax, filtered_ay)

# a, b, h, k = params

# # Function to get ax when given vx, ay
# def get_ax(vx, ay):
#     return h + a * np.sqrt(1 - ((ay - k) ** 2) / (b ** 2))

# # Plot the ellipse and data points
# ellipse_x = np.linspace(h - a, h + a, 1000)
# ellipse_y = ellipse(ellipse_x, a, b, h, k)

# plt.figure(8)
# plt.plot(filtered_ay, filtered_ax, 'ro', label='Data points')
# plt.plot(ellipse_y, ellipse_x, label='Fitted ellipse')
# plt.ylabel('ax')
# plt.xlabel('ay')
# plt.xlim(-8,8)
# plt.ylim(-8,8)
# plt.legend()
# plt.show()
