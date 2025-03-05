import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.drawing.image import Image
import seaborn as sns

# 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("muted")
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# BEMS 데이터 생성 함수
def generate_bems_data(num_samples=200):
    np.random.seed(42)  # 재현성을 위한 시드 값 설정
    timestamps = pd.date_range(start="2024-01-01", periods=num_samples, freq="H")
    temperature = np.random.uniform(18, 30, size=num_samples)
    humidity = np.random.uniform(30, 70, size=num_samples)
    power_usage = np.random.uniform(100, 1000, size=num_samples)
    co2_levels = np.random.uniform(300, 600, size=num_samples)
    occupancy = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])

    data = {
        "Timestamp": timestamps,
        "Temperature": temperature,
        "Humidity": humidity,
        "Power Usage (kWh)": power_usage,
        "CO2 Levels (ppm)": co2_levels,
        "Occupancy": occupancy,
    }
    df = pd.DataFrame(data)
    df["Date"] = df["Timestamp"].dt.date  # 일별 그룹화를 위한 날짜 컬럼 추가
    return df

# Excel 파일 저장 함수
def save_to_excel(df, file_path="bems_data.xlsx"):
    df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"BEMS 데이터가 {file_path} 파일로 저장되었습니다!")

# 모든 시각화를 가로 막대로 표시하고 적정 기준선을 추가하여 엑셀에 저장
def visualize_all_data_to_excel(df, file_path="bems_data.xlsx"):
    df_daily = df.groupby("Date").mean().reset_index()  # 일별 평균값 계산
    columns_to_visualize = [
        ("Temperature", 25),
        ("Humidity", 50),
        ("Power Usage (kWh)", 500),
        ("CO2 Levels (ppm)", 400),
        ("Occupancy", 0.5)
    ]
    num_columns = len(columns_to_visualize)
    fig, axes = plt.subplots(nrows=(num_columns + 1) // 2, ncols=2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (column, threshold) in enumerate(columns_to_visualize):
        sns.barplot(y=df_daily[column], x=df_daily["Date"].astype(str), ax=axes[i], color=np.random.choice(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
        axes[i].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        axes[i].legend()
        axes[i].set_title(f'{column} Levels (Daily Avg)', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Date', fontsize=12)
        axes[i].set_ylabel(column, fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # 빈 차트 제거
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig("bems_visualization.png", dpi=300)
    print("시각화 이미지가 생성되었습니다!")
    
    # 엑셀에 이미지 추가
    wb = openpyxl.load_workbook(file_path)
    ws = wb.create_sheet(title="Visualizations")
    img = Image("bems_visualization.png")
    ws.add_image(img, "A1")
    wb.save(file_path)
    print(f"시각화 이미지가 {file_path}에 저장되었습니다!")

# 실행 예제
if __name__ == "__main__":
    df = generate_bems_data()
    save_to_excel(df)
    
    # 모든 컬럼 시각화를 하나의 시트에 저장
    visualize_all_data_to_excel(df)

