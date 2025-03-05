import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 장애 데이터 생성 함수
def generate_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'cpu_usage': np.random.randint(10, 100, n_samples),  # CPU 사용량 (%)
        'memory_usage': np.random.randint(10, 100, n_samples),  # 메모리 사용량 (%)
        'disk_usage': np.random.randint(10, 100, n_samples),  # 디스크 사용량 (%)
        'network_traffic': np.random.randint(1, 1000, n_samples),  # 네트워크 트래픽 (MB)
        'error_logs': np.random.randint(0, 50, n_samples),  # 로그에서 감지된 오류 개수
        'downtime': np.random.randint(0, 2, n_samples)  # 장애 발생 여부 (0: 정상, 1: 장애)
    }
    return pd.DataFrame(data)

# 머신러닝 모델 훈련 함수
def train_model(df):
    X = df[['cpu_usage', 'memory_usage', 'disk_usage', 'network_traffic', 'error_logs']]
    y = df['downtime']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    
    return model

# 장애 예측 함수
def predict_failure(model, cpu, memory, disk, network, errors):
    input_data = np.array([[cpu, memory, disk, network, errors]])
    prediction = model.predict(input_data)
    return "장애 발생 가능성이 높습니다!" if prediction[0] == 1 else "정상적으로 운영될 가능성이 높습니다."

# 데이터 생성 및 모델 훈련
data = generate_data()
model = train_model(data)

# 새로운 입력값 테스트
sample_prediction = predict_failure(model, cpu=85, memory=90, disk=80, network=700, errors=20)
print(f'예측 결과: {sample_prediction}')
