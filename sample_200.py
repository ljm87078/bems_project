import pandas as pd
import numpy as np

# 200개의 샘플 데이터 생성
np.random.seed(42)  # 재현성을 위한 시드 값 설정
dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
products = np.random.choice(["A", "B", "C", "D", "E"], size=200)
sales = np.random.randint(50, 500, size=200)
quantity = np.random.randint(5, 50, size=200)

data = {
    "Date": dates,
    "Product": products,
    "Sales": sales,
    "Quantity": quantity,
}

df = pd.DataFrame(data)

# 엑셀 파일로 저장
file_path = "sample_data_200.xlsx"
df.to_excel(file_path, index=False, engine="openpyxl")

print(f"200개 데이터가 포함된 샘플이 {file_path}로 저장되었습니다!")