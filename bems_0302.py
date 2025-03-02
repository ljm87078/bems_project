import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BEMSData:
    def __init__(self, num_samples=200):
        self.num_samples = num_samples
        self.sensor_data = None
        self.generate_sample_data()
    
    def generate_sample_data(self):
        np.random.seed(42)
        start_date = datetime(2025, 3, 1)
        zone_list = ['사무실 1층', '사무실 2층', '서버실', '회의실']
        
        data = []
        for _ in range(self.num_samples):
            date = start_date + timedelta(days=np.random.randint(0, 10))
            time = f"{np.random.randint(0, 24):02d}:{np.random.choice(['00', '30'])}"
            zone = np.random.choice(zone_list)
            temp = np.round(np.random.uniform(16, 30), 1)
            humid = np.round(np.random.uniform(30, 70), 1)
            power_usage = np.round(np.random.uniform(50, 400), 1)
            gas_usage = np.round(np.random.uniform(2, 10), 1)
            co2_level = np.random.randint(400, 1000)
            
            data.append([date.strftime('%Y-%m-%d'), time, zone, temp, humid, power_usage, gas_usage, co2_level])
        
        self.sensor_data = pd.DataFrame(data, columns=['날짜', '시간', '구역', '온도(°C)', '습도(%)', '전력 사용량(kWh)', '가스 사용량(m³)', 'CO₂ 농도(ppm)'])
    
    def visualize_data(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='구역', y='온도(°C)', data=self.sensor_data)
        plt.title('구역별 온도 분포')
        plt.savefig('temperature_distribution.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='구역', y='습도(%)', data=self.sensor_data)
        plt.title('구역별 습도 분포')
        plt.savefig('humidity_distribution.png')
        plt.close()

if __name__ == "__main__":
    bems = BEMSData()
    print(bems.sensor_data.head())
    bems.visualize_data()
    print("시각화 자료가 저장되었습니다.")
