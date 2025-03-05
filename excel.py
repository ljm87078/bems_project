import pandas as pd
import matplotlib.pyplot as plt
# import sys

# Excel 파일 로드 (샘플 데이터 사용)
def load_excel(file_path, sheet_name=0):
    """엑셀 파일을 로드하는 함수"""
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    return df

# 데이터 기본 통계 요약
def summarize_data(df):
    """데이터프레임의 기본 통계 요약을 출력"""
    print("데이터 개요:")
    print(df.info())
    print("\n기본 통계 요약:")
    print(df.describe())

# 데이터 시각화
def visualize_data(df, column):
    """특정 컬럼의 분포를 시각화"""
    if column in df.columns:
        plt.figure(figsize=(8, 5))
        df[column].hist(bins=20, edgecolor='black')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'{column} Distribution')
        plt.show()
    else:
        print(f"컬럼 '{column}' 이(가) 데이터에 없습니다.")

# 실행 예제
if __name__ == "__main__":
    file_path = "D:/project2/bems_project/sample_data_200.xlsx"  # 엑셀 파일 경로
    df = load_excel(file_path)
    summarize_data(df)
    
    # 특정 컬럼 시각화 (예: 'Sales')
    visualize_data(df, 'Sales')