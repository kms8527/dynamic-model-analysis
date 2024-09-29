import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# 전달함수 정의 (H(s) = 1/s: 가속도를 적분하여 속도를 구하는 시스템)
numerator = [2]  # 분자: 1
denominator = [1, 3, 2]  # 분모: s (적분기)
sample_time = 0.001

# TransferFunction 객체 생성
system = signal.TransferFunction(numerator, denominator)

# # 시간 벡터 (0초부터 10초까지 1000개의 점)
# time = np.linspace(0, 10, 1000)
time = np.arange(0, 1000, sample_time)

# 무작위 입력 신호 (가속도 입력, 정규분포에서 난수 생성)
random_acceleration = np.random.randn(len(time))

# 시스템 응답 시뮬레이션 (가속도 입력에 대한 속도 출력)
t_out, velocity_output, _ = signal.lsim(system, U=random_acceleration, T=time)

# 결과 플로팅
plt.figure(figsize=(10, 6))

# 가속도 입력 플롯
plt.subplot(2, 1, 1)
plt.plot(t_out, random_acceleration, label='Random Acceleration Input', color='blue', alpha=0.7)
plt.title('Random Acceleration Input and Velocity Output')
plt.ylabel('Acceleration [m/s^2]')
plt.grid(True)
plt.legend()

# 속도 출력 플롯
plt.subplot(2, 1, 2)
plt.plot(t_out, velocity_output, label='Velocity Output', color='green')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.grid(True)
plt.legend()

# 그래프 표시
plt.tight_layout()
plt.show()

save_data = np.vstack((t_out, velocity_output, random_acceleration)).T

pd.DataFrame(save_data, columns=['time', 'velocity_x', 'acceleration']).to_csv('./simple_simulation_data.csv', index=False)
