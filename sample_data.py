import pandas as pd

# 샘플 데이터 생성
data = {
    "Date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
    "Product": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
    "Sales": [100, 150, 200, 130, 180, 220, 140, 170, 210, 160],
    "Quantity": [10, 15, 20, 13, 18, 22, 14, 17, 21, 16],
}

df = pd.DataFrame(data)

# 엑셀 파일로 저장
file_path = "sample_data.xlsx"
df.to_excel(file_path, index=False, engine="openpyxl")

print(f"샘플 데이터가 {file_path}로 저장되었습니다!")