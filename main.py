import csv
import math
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
import numpy as np

filename = "tire_model_data_v2.csv"

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

keys = ['F_fx', 'F_fy', 'F_fz', 'F_rx', 'F_ry', 'F_rz', 'delta' 'yaw', 'yaw_rate', 'lon_slip', 'slip_f', 'slip_r', 'ax']
m = 2065.03 # [kg]
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

real_result['ax'] = dic['Car.ax']
real_result['ay'] = dic['Car.ay']

# 차량 앞 바퀴 lateral slip 값 계산
# -std::atan2(x.vy+x.r*param_.lf,x.vx) + x.delta;
pred_result['slip_f'] = []
for vx, vy, yaw_rate, delta in zip(vx_list, vy_list, dic['Car.YawRate'], real_result['delta']):
    if(vx < 1):
        pred_result['slip_f'].append(0)
    else:
        pred_result['slip_f'].append(math.atan2(vy + yaw_rate * lf, vx+0.001)- delta)


# 차량 뒷 바퀴 lateral slip 값 계산
# -std::atan2(x.vy-x.r*param_.lr,x.vx);
pred_result['slip_r'] = []
for vx, vy, yaw_rate in zip(vx_list, vy_list, dic['Car.YawRate']):
    if(vx < 1):
        pred_result['slip_r'].append(0)
    else:
        pred_result['slip_r'].append(math.atan2(vy - yaw_rate * lr, vx))



                                 # pred_result['lon_slip'] =
pred_result['F_fx'] = 0

# param_.Df * std::sin(param_.Cf * std::atan(param_.Bf * alpha_f ));
Df = 112000.0; Cf = 1; Bf = 1;
pred_result['F_fy'] = [-Df * math.sin(Cf* math.atan(Bf*alpha_f)) for alpha_f in pred_result['slip_f']]

# pred_result['F_fz'] =

# param_.Dr * std::sin(param_.Cr * std::atan(param_.Br * alpha_r ));
Cm1 = 3000; Cm2 = 500; Cr = 0; Cd = 0;
pred_result['F_rx'] = [-(Cm1 - Cm2 * vx) * throttle - Cr - Cd * vx**2 for vx, throttle in zip(vx_list, gas_list)]

# param_.Cm1*x.D - param_.Cm2*x.D*x.vx;
Dr = 130000; Cr = 1; Br = 1;
pred_result['F_ry'] = [-Dr * math.sin(Cr * math.atan(Br* alpha_r)) for alpha_r in pred_result['slip_r']]

#kinematic
pred_result['ax'] = [ 2*(F_rx + F_fx)/(m) for F_fx, F_rx in zip(real_result['F_fx'],real_result['F_rx'])]

d_delta_list = []
d_delta_list.append(0)
for i in range(1, len(real_result['delta'])):
    d_delta_list.append((real_result['delta'][i] - real_result['delta'][i-1])/0.01)

#kinematic
pred_result['yaw_rate'] = [vx * math.cos((lr*math.tan(steer)/(lf+lr)))/(lf+lr) * math.tan(steer) for vx, steer in zip(vx_list, real_result['delta'])]

#dynamic
# pred_result['yaw_rate'] = [2/Iz * (F_fy * lf * math.cos(steer) - F_ry*lr) for steer, F_fy, F_ry in zip(real_result['delta'],real_result['F_fy'], real_result['F_ry'])]

#kinematic
# pred_result['ay'] = [ (d_steer*vx + steer * ax) * lr/(lr+lf) for vx, ax, steer, d_steer in zip(dic['Car.vx'],pred_result['ax'],real_result['delta'],d_delta_list)]

#dynamic
pred_result['ay'] = [ -1/m*(F_ry - F_fy*math.cos(steer)-m*vx*yaw_rate) for vx, F_fy, F_ry, steer, yaw_rate in zip(dic['Car.vx'],pred_result['F_fy'], pred_result['F_ry'],real_result['delta'],pred_result['yaw_rate'])]

# pred_result['F_rz'] =

plt.figure(1)
plt.subplot(2,1,1)
plt.title('F_fy Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('F_fy[N]')
# Plot the predicted results
plt.plot(dic['Time'], pred_result['F_fy'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['F_fy'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.subplot(2,1,2)
plt.title('F_fy Comparison')
plt.xlabel('slip_f[rad]')
plt.ylabel('F_fy[N]')
# Plot the predicted results
plt.plot(real_result['slip_f'], pred_result['F_fy'],'.', label='Predicted')
# Plot the real results
plt.plot(real_result['slip_f'], real_result['F_fy'], marker=",", label='Real')
# Add a legend to the graph
plt.legend(loc='best')
plt.tight_layout() # 두 subplot graph간 간격 적절히 조정

plt.figure(2)
plt.subplot(2,1,1)
plt.title('front slip angle Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('slip angel[rad]')
# Plot the predicted results
plt.plot(dic['Time'], pred_result['slip_f'],marker=",", label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['slip_f'], marker=",", label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.subplot(2,1,2)
plt.title('rear slip angle Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('slip angel[rad]')
# Plot the predicted results
plt.plot(dic['Time'], pred_result['slip_r'],marker=",", label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['slip_r'], marker=",", label='Real')
# Add a legend to the graph
plt.legend(loc='best')
plt.tight_layout() # 두 subplot graph간 간격 적절히 조정

plt.figure(3)
plt.title('front slip angle Comparison')
plt.xlabel('real slip angle[rad]')
plt.ylabel('pred slip angel[rad]')
# Plot the predicted results
plt.plot(real_result['slip_f'], pred_result['slip_f'],marker=",", label='Predicted')
# Plot the real results
plt.plot(real_result['slip_f'], real_result['slip_f'], marker=",", label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.figure(4)
plt.title('yaw_rate Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('yaw rate[rad/s]')
# Plot the predicted results
plt.plot(dic['Time'], pred_result['yaw_rate'],marker=",", label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['yaw_rate'], marker=",", label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.figure(5)
plt.subplot(2,1,1)
plt.title('F_ry Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('F_ry[N]')
# Plot the predicted results
plt.plot(dic['Time'], pred_result['F_ry'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['F_ry'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.subplot(2,1,2)
plt.title('F_ry Comparison')
plt.xlabel('slip_f[rad]')
plt.ylabel('F_ry[N]')
# Plot the predicted results
plt.plot(real_result['slip_f'], pred_result['F_ry'],',', label='Predicted')
# Plot the real results
plt.plot(real_result['slip_f'], real_result['F_ry'], marker=",", label='Real')
# Add a legend to the graph
plt.legend(loc='best')
plt.tight_layout() # 두 subplot graph간 간격 적절히 조정

plt.figure(6)
plt.subplot(2,1,1)
plt.title('F_rx Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('F_rx[N]')
plt.plot(dic['Time'], pred_result['F_rx'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['F_rx'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')
# Display the graph

plt.subplot(2,1,2)
plt.title('F_rx Comparison')
plt.xlabel('lon_slip')

plt.ylabel('F_rx[N]')
plt.plot(real_result['lon_slip'], pred_result['F_rx'],',', label='Predicted')
# Plot the real results
plt.plot(real_result['lon_slip'], real_result['F_rx'],',', label='Real')
# Add a legend to the graph
plt.legend(loc='best')
plt.tight_layout() # 두 subplot graph간 간격 적절히 조정

plt.figure(7)
plt.subplot(2,1,1)
plt.title('ax Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('ax[m/s^2]')
plt.plot(dic['Time'], pred_result['ax'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['ax'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')

plt.subplot(2,1,2)
plt.title('ay Comparison')
plt.xlabel('Time[sec]')
plt.ylabel('ay[m/s^2]')
plt.plot(dic['Time'], pred_result['ay'], label='Predicted')
# Plot the real results
plt.plot(dic['Time'], real_result['ay'], label='Real')
# Add a legend to the graph
plt.legend(loc='best')
plt.tight_layout() # 두 subplot graph간 간격 적절히 조정


plt.show()
eval_keys = ['F_fy', 'F_rx','slip_f','slip_r','yaw_rate','F_ry','ax', 'ay']
for key in eval_keys:
    error_list = [abs(real - pred) for real, pred in zip(pred_result[key], real_result[key])]
    error_avg = sum(error_list) / len(error_list)
    print("avg", key ," error : ", error_avg)