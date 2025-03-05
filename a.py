import pandas as pd
import numpy as np

# 가상 IoT 센서 데이터 생성 (전력 사용량, 온도, 습도)
np.random.seed(42)
data = {
"timestamp": pd.date_range(start="2025-01-01", periods=100, freq="H"), # 100시간 데이터
"power_usage": np.random.normal(loc=100, scale=20, size=100).clip(50, 200), # 전력 사용량 (50~200W)
"temperature": np.random.normal(loc=22, scale=2, size=100).clip(18, 26), # 실내 온도 (18~26도)
"humidity": np.random.normal(loc=50, scale=5, size=100).clip(40, 60), # 습도 (40~60%)
}

df = pd.DataFrame(data)
print(df.head()) #