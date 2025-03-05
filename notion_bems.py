import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# 한글 폰트 설정 - 맑은 고딕 사용
import matplotlib.font_manager as fm

# 한글 폰트 경로 (윈도우 기준)
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕 폰트 경로
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

# 샘플 데이터 생성 함수
def generate_bems_data(days=30, buildings=3, sensors_per_building=5):
    # 시작 날짜 설정
    start_date = datetime.now() - timedelta(days=days)
    
    # 데이터 프레임을 위한 빈 리스트
    data = []
    
    # 건물별 기준값 설정 (각 건물마다 다른 기준값)
    building_baselines = {
        f'건물_{i+1}': {
            '온도': np.random.uniform(21, 24),  # 섭씨
            '습도': np.random.uniform(40, 60),  # 퍼센트
            '전력': np.random.uniform(30, 50),  # kW
            'CO2': np.random.uniform(400, 600),  # ppm
            '조명': np.random.uniform(300, 500)  # lux
        } for i in range(buildings)
    }
    
    # 경고 임계값 설정 (모든 건물에 공통적으로 적용)
    warning_thresholds = {
        '온도': {'낮음': 18.0, '높음': 28.0},
        '습도': {'낮음': 30.0, '높음': 70.0},
        '전력': {'낮음': None, '높음': 75.0},
        'CO2': {'낮음': None, '높음': 1000.0},
        '조명': {'낮음': 200.0, '높음': 800.0}
    }
    
    # 데이터 생성
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        
        # 하루 24시간 데이터
        for hour in range(24):
            current_datetime = current_date + timedelta(hours=hour)
            
            # 각 건물별 데이터
            for building_id in range(buildings):
                building_name = f'건물_{building_id+1}'
                baseline = building_baselines[building_name]
                
                # 각 센서별 데이터
                for sensor_id in range(sensors_per_building):
                    sensor_name = f'센서_{sensor_id+1}'
                    
                    # 시간에 따른 변동 패턴 추가 (주간/야간)
                    time_factor = 1.0 if 8 <= hour <= 18 else 0.7
                    
                    # 각 측정값 생성
                    temp = baseline['온도'] + np.random.normal(0, 2) * time_factor
                    humidity = baseline['습도'] + np.random.normal(0, 5) * time_factor
                    power = baseline['전력'] + np.random.normal(0, 8) * time_factor
                    co2 = baseline['CO2'] + np.random.normal(0, 50) * time_factor
                    light = baseline['조명'] + np.random.normal(0, 30) * time_factor
                    
                    # 주말 패턴 (전력 사용량 감소)
                    if current_datetime.weekday() >= 5:  # 토, 일요일
                        power *= 0.6
                        co2 *= 0.8
                        
                    # 비정상적인 값 가끔 추가 (이상치)
                    if np.random.random() < 0.02:  # 2% 확률로 이상치 발생
                        anomaly_factor = np.random.choice([0.5, 1.5])  # 정상값의 50% 또는 150%
                        anomaly_type = np.random.choice(['온도', '습도', '전력', 'CO2', '조명'])
                        
                        if anomaly_type == '온도':
                            temp *= anomaly_factor
                        elif anomaly_type == '습도':
                            humidity *= anomaly_factor
                        elif anomaly_type == '전력':
                            power *= anomaly_factor
                        elif anomaly_type == 'CO2':
                            co2 *= anomaly_factor
                        else:
                            light *= anomaly_factor
                    
                    # 상태 판정 (정상/경고/위험)
                    temp_status = '정상'
                    if temp < warning_thresholds['온도']['낮음']:
                        temp_status = '경고 (낮음)'
                    elif temp > warning_thresholds['온도']['높음']:
                        temp_status = '경고 (높음)'
                        
                    humidity_status = '정상'
                    if humidity < warning_thresholds['습도']['낮음']:
                        humidity_status = '경고 (낮음)'
                    elif humidity > warning_thresholds['습도']['높음']:
                        humidity_status = '경고 (높음)'
                        
                    power_status = '정상'
                    if warning_thresholds['전력']['높음'] and power > warning_thresholds['전력']['높음']:
                        power_status = '경고 (높음)'
                        
                    co2_status = '정상'
                    if warning_thresholds['CO2']['높음'] and co2 > warning_thresholds['CO2']['높음']:
                        co2_status = '경고 (높음)'
                        
                    light_status = '정상'
                    if light < warning_thresholds['조명']['낮음']:
                        light_status = '경고 (낮음)'
                    elif light > warning_thresholds['조명']['높음']:
                        light_status = '경고 (높음)'
                    
                    # 데이터 추가
                    data.append({
                        '날짜시간': current_datetime,
                        '건물': building_name,
                        '센서': sensor_name,
                        '온도(°C)': round(temp, 2),
                        '온도_상태': temp_status,
                        '습도(%)': round(humidity, 2),
                        '습도_상태': humidity_status,
                        '전력(kW)': round(power, 2),
                        '전력_상태': power_status,
                        'CO2(ppm)': round(co2, 2),
                        'CO2_상태': co2_status,
                        '조명(lux)': round(light, 2),
                        '조명_상태': light_status
                    })
    
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    
    return df, warning_thresholds

# 샘플 데이터 생성
df, warning_thresholds = generate_bems_data(days=30, buildings=3, sensors_per_building=5)

# 엑셀 파일로 저장
excel_filename = 'BEMS_샘플데이터.xlsx'
df.to_excel(excel_filename, index=False)
print(f"데이터가 '{excel_filename}' 파일로 저장되었습니다.")

# 데이터 시각화

# 1. 건물별 평균 온도 분포 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(x='건물', y='온도(°C)', data=df)
plt.axhline(y=warning_thresholds['온도']['낮음'], color='b', linestyle='--', label=f'경고 하한: {warning_thresholds["온도"]["낮음"]}°C')
plt.axhline(y=warning_thresholds['온도']['높음'], color='r', linestyle='--', label=f'경고 상한: {warning_thresholds["온도"]["높음"]}°C')
plt.title('건물별 온도 분포')
plt.legend()
plt.tight_layout()
plt.savefig('건물별_온도_분포.png')

# 2. 시간대별 전력 사용량 추이
df['시간'] = df['날짜시간'].dt.hour
hourly_power = df.groupby(['건물', '시간'])['전력(kW)'].mean().unstack()

plt.figure(figsize=(12, 6))
hourly_power.plot(marker='o')
plt.axhline(y=warning_thresholds['전력']['높음'], color='r', linestyle='--', label=f'경고 임계값: {warning_thresholds["전력"]["높음"]}kW')
plt.title('시간대별 평균 전력 사용량')
plt.xlabel('시간')
plt.ylabel('전력 사용량 (kW)')
plt.xticks(range(24))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('시간대별_전력_사용량.png')

# 3. CO2 농도 분포
plt.figure(figsize=(10, 6))
sns.histplot(df['CO2(ppm)'], bins=30, kde=True)
plt.axvline(x=warning_thresholds['CO2']['높음'], color='r', linestyle='--', label=f'경고 상한: {warning_thresholds["CO2"]["높음"]}ppm')
plt.title('CO2 농도 분포')
plt.xlabel('CO2 농도 (ppm)')
plt.ylabel('빈도')
plt.legend()
plt.tight_layout()
plt.savefig('CO2_농도_분포.png')

# 4. 상태별 데이터 개수
status_counts = pd.DataFrame({
    '온도': df['온도_상태'].value_counts(),
    '습도': df['습도_상태'].value_counts(),
    '전력': df['전력_상태'].value_counts(),
    'CO2': df['CO2_상태'].value_counts(),
    '조명': df['조명_상태'].value_counts()
})

plt.figure(figsize=(12, 8))
status_counts.plot(kind='bar')
plt.title('센서 상태별 데이터 개수')
plt.xlabel('상태')
plt.ylabel('개수')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('센서_상태별_개수.png')

print("데이터 시각화가 완료되었습니다.")