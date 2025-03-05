import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 샘플 데이터 생성 함수
def generate_bems_data(days=30, buildings=3, sensors_per_building=5):
    start_date = datetime.now() - timedelta(days=days)
    data = []
    building_baselines = {
        f'건물_{i+1}': {
            '온도': np.random.uniform(21, 24),
            '습도': np.random.uniform(40, 60),
            '전력': np.random.uniform(30, 50),
            'CO2': np.random.uniform(400, 600),
            '조명': np.random.choice([True, False])
        } for i in range(buildings)
    }
    warning_thresholds = {
        '온도': {'낮음': 18, '높음': 26},
        '습도': {'낮음': 30, '높음': 70},
        '전력': {'높음': 60},
        'CO2': {'높음': 1000},
        '조명': {'꺼짐': 0}
    }
    for building, baselines in building_baselines.items():
        for sensor in range(sensors_per_building):
            for day in range(days):
                for hour in range(24):
                    timestamp = start_date + timedelta(days=day, hours=hour)
                    temperature = np.random.normal(baselines['온도'], 2)
                    humidity = np.random.normal(baselines['습도'], 5)
                    power = np.random.normal(baselines['전력'], 10)
                    co2 = np.random.normal(baselines['CO2'], 50)
                    light = np.random.choice([True, False], p=[0.7, 0.3])
                    data.append([timestamp, building, f'센서_{sensor+1}', temperature, humidity, power, co2, light])
    df = pd.DataFrame(data, columns=['날짜시간', '건물', '센서', '온도(°C)', '습도(%)', '전력(kW)', 'CO2(ppm)', '조명(On/Off)'])
    # Define states based on warning thresholds
    df['온도_상태'] = pd.cut(df['온도(°C)'], bins=[-float('inf'), warning_thresholds['온도']['낮음'], warning_thresholds['온도']['높음'], float('inf')], labels=['낮음', '정상', '높음'])
    df['습도_상태'] = pd.cut(df['습도(%)'], bins=[-float('inf'), warning_thresholds['습도']['낮음'], warning_thresholds['습도']['높음'], float('inf')], labels=['낮음', '정상', '높음'])
    df['전력_상태'] = pd.cut(df['전력(kW)'], bins=[-float('inf'), warning_thresholds['전력']['높음'], float('inf')], labels=['정상', '높음'])
    df['CO2_상태'] = pd.cut(df['CO2(ppm)'], bins=[-float('inf'), warning_thresholds['CO2']['높음'], float('inf')], labels=['정상', '높음'])
    df['조명_상태'] = df['조명(On/Off)'].apply(lambda x: '꺼짐' if not x else '켜짐')
    return df, warning_thresholds

# 샘플 데이터 생성
df, warning_thresholds = generate_bems_data(days=30, buildings=3, sensors_per_building=5)

# 데이터 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # 2x2 서브플롯 생성

# 1. 건물별 평균 온도 분포 시각화
sns.boxplot(x='건물', y='온도(°C)', data=df, ax=axes[0, 0])
axes[0, 0].axhline(y=warning_thresholds['온도']['낮음'], color='b', linestyle='--', label=f'경고 하한: {warning_thresholds["온도"]["낮음"]}°C')
axes[0, 0].axhline(y=warning_thresholds['온도']['높음'], color='r', linestyle='--', label=f'경고 상한: {warning_thresholds["온도"]["높음"]}°C')
axes[0, 0].set_title('건물별 온도 분포')
axes[0, 0].legend()

# 2. 시간대별 전력 사용량 추이
df['시간'] = df['날짜시간'].dt.hour
hourly_power = df.groupby(['건물', '시간'])['전력(kW)'].mean().unstack()
hourly_power.plot(marker='o', ax=axes[0, 1])
axes[0, 1].axhline(y=warning_thresholds['전력']['높음'], color='r', linestyle='--', label=f'경고 임계값: {warning_thresholds["전력"]["높음"]}kW')
axes[0, 1].set_title('시간대별 평균 전력 사용량')
axes[0, 1].set_xlabel('시간')
axes[0, 1].set_ylabel('전력 사용량 (kW)')
axes[0, 1].set_xticks(range(24))
axes[0, 1].grid(True)
axes[0, 1].legend()

# 3. CO2 농도 분포
sns.histplot(df['CO2(ppm)'], bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].axvline(x=warning_thresholds['CO2']['높음'], color='r', linestyle='--', label=f'경고 상한: {warning_thresholds["CO2"]["높음"]}ppm')
axes[1, 0].set_title('CO2 농도 분포')
axes[1, 0].set_xlabel('CO2 농도 (ppm)')
axes[1, 0].set_ylabel('빈도')
axes[1, 0].legend()

# 4. 상태별 데이터 개수
status_counts = pd.DataFrame({
    '온도': df['온도_상태'].value_counts(),
    '습도': df['습도_상태'].value_counts(),
    '전력': df['전력_상태'].value_counts(),
    'CO2': df['CO2_상태'].value_counts(),
    '조명': df['조명_상태'].value_counts()
})
status_counts.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('센서 상태별 데이터 개수')
axes[1, 1].set_xlabel('상태')
axes[1, 1].set_ylabel('개수')
axes[1, 1].grid(axis='y')

plt.tight_layout()
plt.savefig('BEMS_데이터_시각화.png')
plt.show()