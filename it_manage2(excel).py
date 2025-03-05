import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 데이터 생성 (엑셀 파일)
np.random.seed(42)
n_samples = 1000
data = {
    '온도': np.random.normal(25, 3, n_samples),
    '습도': np.random.normal(60, 10, n_samples),
    '전압': np.random.normal(220, 5, n_samples),
    '장애 발생': np.random.choice([0, 1], n_samples)  # 0: 정상, 1: 장애 발생
}
df = pd.DataFrame(data)
df.to_excel('장애_데이터.xlsx', index=False)  # 엑셀 파일로 저장

# 2. 데이터 불러오기 및 전처리
data = pd.read_excel('장애_데이터.xlsx')
X = data[['온도', '습도', '전압']]
y = data['장애 발생']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 학습 (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# 5. 예측 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_test['온도'], X_test['습도'], c=y_pred, cmap='RdYlBu', alpha=0.7)
plt.title('장애 예측 결과 (온도, 습도)')
plt.xlabel('온도')
plt.ylabel('습도')
plt.colorbar(label='장애 발생 (0: 정상, 1: 장애)')
plt.show()