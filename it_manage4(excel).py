# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 한글 폰트 설정 (Windows 환경에 맞게 경로 수정)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 엑셀 파일 생성 및 데이터 저장
np.random.seed(42)
n_samples = 1000
dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
data = {
    '날짜': dates,
    'CPU 사용률 (%)': np.random.uniform(10, 100, n_samples),
    '메모리 사용률 (%)': np.random.uniform(20, 95, n_samples),
    '네트워크 트래픽 (Gbps)': np.random.exponential(1, n_samples),
    '서버 응답 시간 (ms)': np.random.chisquare(5, n_samples) * 10,
    '스토리지 I/O (MB/s)': np.random.normal(50, 15, n_samples),
    '장애 발생': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
}
df = pd.DataFrame(data)

# 이상치 추가 (장애 상황 시뮬레이션)
for i in range(n_samples):
    if df.loc[i, '장애 발생'] == 1:
        df.loc[i, 'CPU 사용률 (%)'] = np.random.uniform(80, 100)
        df.loc[i, '메모리 사용률 (%)'] = np.random.uniform(90, 100)
        df.loc[i, '네트워크 트래픽 (Gbps)'] = np.random.exponential(5)
        df.loc[i, '서버 응답 시간 (ms)'] = np.random.chisquare(10) * 20
        df.loc[i, '스토리지 I/O (MB/s)'] = np.random.normal(100, 30)

df.to_excel('IT_장애_데이터.xlsx', index=False)  # 엑셀 파일로 저장

# 엑셀 파일 불러오기
try:
    df = pd.read_excel('IT_장애_데이터.xlsx')
except FileNotFoundError:
    print("오류: 'IT_장애_데이터.xlsx' 파일을 찾을 수 없습니다. 현재 작업 디렉토리에 파일이 있는지 확인해주세요.")
    exit()

# 데이터 전처리
X = df.drop(['날짜', '장애 발생'], axis=1)
y = df['장애 발생']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습 (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# 시각화 (예시: CPU 사용률 vs. 메모리 사용률)
plt.figure(figsize=(10, 6))
plt.scatter(X_test['CPU 사용률 (%)'], X_test['메모리 사용률 (%)'], c=y_pred, cmap='RdYlBu', alpha=0.7)
plt.title('IT 장애 예측 결과 (CPU, 메모리)')
plt.xlabel('CPU 사용률 (%)')
plt.ylabel('메모리 사용률 (%)')
plt.colorbar(label='장애 발생 (0: 정상, 1: 장애)')
plt.grid(True)
plt.show()

# 추가 시각화 (예시: 시간 경과에 따른 CPU 사용률 변화)
plt.figure(figsize=(12, 6))
plt.plot(df['날짜'], df['CPU 사용률 (%)'])
plt.title('시간별 CPU 사용률 변화')
plt.xlabel('시간')
plt.ylabel('CPU 사용률 (%)')
plt.grid(True)
plt.show()

# 추가 시각화 (예시: 각 지표별 장애 발생 빈도)
indicators = ['CPU 사용률 (%)', '메모리 사용률 (%)', '네트워크 트래픽 (Gbps)', '서버 응답 시간 (ms)', '스토리지 I/O (MB/s)']
plt.figure(figsize=(15, 10))
for i, indicator in enumerate(indicators):
    plt.subplot(2, 3, i+1)
    plt.hist(df[df['장애 발생'] == 0][indicator], bins=20, alpha=0.5, label='정상')
    plt.hist(df[df['장애 발생'] == 1][indicator], bins=20, alpha=0.5, label='장애 발생')
    plt.title(f'{indicator}별 장애 발생 빈도')
    plt.legend()
plt.tight_layout()
plt.show()

# 데이터 프레임의 모든 데이터 출력
print(df)