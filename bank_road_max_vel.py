import math

# 최대 속력을 구하는 함수
def v_max(g, r, mu, theta_deg):
    theta = math.radians(theta_deg)  # 각도를 라디안으로 변환
    numerator = math.sin(theta) + mu * math.cos(theta)
    denominator = math.cos(theta) - mu * math.sin(theta)
    
    if denominator == 0:
        raise ValueError("분모가 0이 되어 속도를 계산할 수 없습니다.")
    
    return math.sqrt((g * r * numerator) / denominator)

def v_min(g, r, mu, theta_deg):
    theta = math.radians(theta_deg)  # 각도를 라디안으로 변환
    numerator = math.sin(theta) - mu * math.cos(theta)
    denominator = math.cos(theta) + mu * math.sin(theta)
    
    if denominator == 0 or numerator < 0:
        # raise ValueError("분모가 0이 되어 속도를 계산할 수 없습니다.")
        return 0
    
    return math.sqrt((g * r * numerator) / denominator)

# 주어진 거리 s를 기반으로 도착 시간을 구하는 함수
def time_to_reach(v, s):
    if v == 0:
        raise ValueError("속도가 0일 때 시간은 무한대입니다.")
    
    return s / v

# 예시로 사용될 값들
g = 9.8  # 중력 가속도 m/s^2
r = 103.5  # 곡률 반경, 예시로 100m
mu = 0.9  # 마찰계수
theta_deg = 30  # 각도 45도

# 거리 s는 200m로 예시
s = r * math.pi

# 최대 속력 계산
v_max_speed = v_max(g, r, mu, theta_deg)
v_min_speed = v_min(g, r, mu, theta_deg)

# 도착 시간 계산
arrival_time = time_to_reach(v_max_speed, s)

v_max_speed, arrival_time

print(f"최대 속력: {v_max_speed:.2f} m/s")
print(f"최대 속력: {v_max_speed * 3.6:.2f} kph")
print(f"최소 속력: {v_min_speed:.2f} m/s")
print(f"도착 시간: {arrival_time:.2f} s")


