import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 샘플 데이터 개수 설정
data_size = 200

# 날짜 및 시간 생성
start_date = datetime(2025, 3, 1)
dates = [(start_date + timedelta(days=np.random.randint(0, 10))).strftime('%Y-%m-%d') for _ in range(data_size)]
times = [f'{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}' for _ in range(data_size)]

# 구역 리스트
divisions = ['사무실 1층', '사무실 2층', '서버실', '회의실']
zones = [np.random.choice(divisions) for _ in range(data_size)]

# 온도 및 습도 데이터 생성
temps = [round(np.random.uniform(15, 30), 1) for _ in range(data_size)]
humids = [np.random.randint(30, 80) for _ in range(data_size)]

# 에너지 사용량 데이터 생성
power_usages = [round(np.random.uniform(50, 500), 1) for _ in range(data_size)]
gas_usages = [round(np.random.uniform(0, 10), 1) for _ in range(data_size)]
co2_levels = [np.random.randint(400, 1000) for _ in range(data_size)]

# 상태 설정 함수
def determine_status(temp, humid):
    if temp < 18 or temp > 26:
        return '경고-온도'
    elif humid < 40 or humid > 60:
        return '경고-습도'
    return '정상'

statuses = [determine_status(temp, humid) for temp, humid in zip(temps, humids)]

# 데이터프레임 생성
df = pd.DataFrame({
    '날짜': dates,
    '시간': times,
    '구역': zones,
    '온도(°C)': temps,
    '습도(%)': humids,
    '전력 사용량(kWh)': power_usages,
    '가스 사용량(m³)': gas_usages,
    'CO₂ 농도(ppm)': co2_levels,
    '상태': statuses
})

# 데이터 확인
print(df.head())

# CSV로 저장
df.to_csv('bems_sample_200.csv', index=False, encoding='utf-8-sig')
