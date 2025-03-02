import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 모델 정의
class BEMSData:
    def __init__(self):
        # 기준치 설정
        self.temp_standards = {
            '일반': {'적정': (18, 26), '경고': [(16, 18), (26, 28)], '위험': [(0, 16), (28, 100)]},
            '서버실': {'적정': (18, 24), '경고': [(16, 18), (24, 27)], '위험': [(0, 16), (27, 100)]}
        }
        self.humid_standards = {
            '일반': {'적정': (40, 60), '경고': [(30, 40), (60, 70)], '위험': [(0, 30), (70, 100)]},
            '서버실': {'적정': (45, 55), '경고': [(35, 45), (55, 65)], '위험': [(0, 35), (65, 100)]}
        }
        
        # 데이터프레임 초기화
        self.sensor_data = None
        self.energy_summary = None
        self.equipment_status = None
        self.alarm_history = None
        
        # 샘플 데이터 생성
        self.generate_sample_data()
    
    def generate_sample_data(self):
        # 센서 데이터 생성
        dates = []
        times = []
        zones = []
        temps = []
        humids = []
        power_usages = []
        gas_usages = []
        co2_levels = []
        statuses = []
        
        start_date = datetime(2025, 3, 1)
        time_slots = ['08:00', '12:00', '16:00']
        zone_list = ['사무실 1층', '사무실 2층', '서버실', '회의실']
        
        for day in range(3):
            current_date = start_date + timedelta(days=day)
            date_str = current_date.strftime('%Y-%m-%d')
            
            for time_slot in time_slots:
                for zone in zone_list:
                    dates.append(date_str)
                    times.append(time_slot)
                    zones.append(zone)
                    
                    # 시간대별 온도, 습도 변화 패턴 반영
                    base_temp = 22 if time_slot == '08:00' else 24 if time_slot == '12:00' else 25.5
                    base_humid = 45 if time_slot == '08:00' else 50 if time_slot == '12:00' else 58
                    
                    # 구역별 특성 반영
                    if zone == '서버실':
                        temp = base_temp - 0.5 + np.random.uniform(-1, 1)
                        humid = base_humid - 3 + np.random.uniform(-2, 2)
                    else:
                        temp = base_temp + np.random.uniform(-1.5, 1.5)
                        humid = base_humid + np.random.uniform(-5, 5)
                    
                    temps.append(round(temp, 1))
                    humids.append(round(humid))
                    
                    # 에너지 사용량
                    if zone == '서버실':
                        power = 350 + np.random.uniform(-20, 40)
                        gas = 0
                    else:
                        power = 110 + np.random.uniform(-30, 60)
                        gas = 4.5 + np.random.uniform(-1.5, 2)
                    
                    power_usages.append(round(power))
                    gas_usages.append(round(gas, 1))
                    
                    # CO2 농도
                    co2 = 650 + np.random.uniform(-50, 150)
                    co2_levels.append(round(co2))
                    
                    # 상태 확인
                    status = self.check_status(temp, humid, zone)
                    statuses.append(status)
        
        # 데이터프레임 생성
        self.sensor_data = pd.DataFrame({
            '날짜': dates,
            '시간': times,
            '구역': zones,
            '온도(°C)': temps,
            '습도(%)': humids,
            '전력 사용량(kWh)': power_usages,
            '가스 사용량(m³)': gas_usages,
            'CO₂ 농도(ppm)': co2_levels,
            '상태': statuses
        })
        
        # 에너지 사용량 요약 데이터 생성
        self.generate_energy_summary()
        
        # 장비 상태 데이터 생성
        self.generate_equipment_status()
        
        # 알람 기록 생성
        self.generate_alarm_history()
    
    def check_status(self, temp, humid, zone):
        zone_type = '서버실' if zone == '서버실' else '일반'
        
        # 온도 상태 확인
        temp_status = "정상"
        for i, (low, high) in enumerate(self.temp_standards[zone_type]['경고']):
            if low <= temp <= high:
                temp_status = "경고-온도"
                break
        for i, (low, high) in enumerate(self.temp_standards[zone_type]['위험']):
            if low <= temp <= high:
                temp_status = "위험-온도"
                break
        
        # 습도 상태 확인
        humid_status = "정상"
        for i, (low, high) in enumerate(self.humid_standards[zone_type]['경고']):
            if low <= humid <= high:
                humid_status = "경고-습도"
                break
        for i, (low, high) in enumerate(self.humid_standards[zone_type]['위험']):
            if low <= humid <= high:
                humid_status = "위험-습도"
                break
        
        # 최종 상태 결정
        if temp_status == "정상" and humid_status == "정상":
            return "정상"
        elif temp_status.startswith("위험") or humid_status.startswith("위험"):
            if temp_status.startswith("위험") and humid_status.startswith("위험"):
                return "위험-온도/습도"
            elif temp_status.startswith("위험"):
                return temp_status
            else:
                return humid_status
        else:
            if temp_status != "정상" and humid_status != "정상":
                return "경고-온도/습도"
            elif temp_status != "정상":
                return temp_status
            else:
                return humid_status
    
    def generate_energy_summary(self):
        # 날짜별, 구역별 에너지 사용량 요약
        summary = self.sensor_data.groupby(['날짜', '구역']).agg({
            '온도(°C)': 'mean',
            '습도(%)': 'mean',
            '전력 사용량(kWh)': 'sum',
            '가스 사용량(m³)': 'sum',
            'CO₂ 농도(ppm)': 'mean'
        }).reset_index()
        
        # 컬럼명 변경
        summary.columns = ['날짜', '구역', '평균 온도(°C)', '평균 습도(%)', 
                          '총 전력 사용량(kWh)', '총 가스 사용량(m³)', '평균 CO₂ 농도(ppm)']
        
        # 소수점 반올림
        summary['평균 온도(°C)'] = summary['평균 온도(°C)'].round(1)
        summary['평균 습도(%)'] = summary['평균 습도(%)'].round()
        summary['총 전력 사용량(kWh)'] = summary['총 전력 사용량(kWh)'].round()
        summary['총 가스 사용량(m³)'] = summary['총 가스 사용량(m³)'].round(1)
        summary['평균 CO₂ 농도(ppm)'] = summary['평균 CO₂ 농도(ppm)'].round()
        
        self.energy_summary = summary
    
    def generate_equipment_status(self):
        # 장비 상태 데이터 생성
        equipment_data = {
            '장비 ID': ['HVAC-001', 'HVAC-002', 'HVAC-003', 'VENT-001', 'VENT-002', 'LIGHT-001', 'ELEV-001'],
            '장비명': ['냉난방기 1호', '냉난방기 2호', '서버실 항온항습기', '환기장치 1호', '환기장치 2호', '조명제어시스템', '엘리베이터 1호'],
            '설치 위치': ['사무실 1층', '사무실 2층', '서버실', '사무실 1층', '사무실 2층', '전체', '중앙 홀'],
            '가동 상태': ['가동 중'] * 7,
            '온도(°C)': [23.5, 24.1, 21.8, np.nan, np.nan, np.nan, np.nan],
            '습도(%)': [48, 50, 49, np.nan, np.nan, np.nan, np.nan],
            '전력 소비(kW)': [8.2, 8.5, 12.3, 3.2, 3.4, 5.6, 4.2],
            '마지막 점검일': ['2025-02-15', '2025-02-15', '2025-02-10', '2025-02-20', '2025-02-20', '2025-02-25', '2025-01-30'],
            '다음 점검 예정일': ['2025-05-15', '2025-05-15', '2025-04-10', '2025-05-20', '2025-05-20', '2025-05-25', '2025-04-30'],
            '이상 상태': ['없음'] * 7
        }
        
        self.equipment_status = pd.DataFrame(equipment_data)
    
    def generate_alarm_history(self):
        # 경고 상태인 데이터 찾기
        warnings = self.sensor_data[self.sensor_data['상태'] != '정상'].copy()
        
        if len(warnings) == 0:
            # 경고가 없을 경우 빈 데이터프레임 생성
            self.alarm_history = pd.DataFrame(columns=[
                '알람 ID', '날짜', '시간', '구역', '알람 유형', '측정값', '기준치', '조치 사항', '조치 완료', '담당자'
            ])
            return
        
        # 알람 이력 생성
        alarm_ids = [f'A{i:04d}' for i in range(1, len(warnings) + 1)]
        alarm_times = []
        alarm_types = []
        measurements = []
        standards = []
        actions = []
        completions = ['완료'] * len(warnings)
        managers = []
        
        for idx, row in warnings.iterrows():
            # 알람 시간 (센서 데이터 시간보다 5분 뒤)
            hour, minute = map(int, row['시간'].split(':'))
            minute += 5
            if minute >= 60:
                hour += 1
                minute -= 60
            alarm_times.append(f'{hour:02d}:{minute:02d}')
            
            # 알람 유형
            if '온도/습도' in row['상태']:
                if '위험' in row['상태']:
                    alarm_types.append('온도/습도 위험')
                else:
                    alarm_types.append('온도/습도 경고')
                measurements.append(f"{row['온도(°C)']}°C / {row['습도(%)']}%")
                zone_type = '서버실' if row['구역'] == '서버실' else '일반'
                temp_range = f"{self.temp_standards[zone_type]['적정'][0]}~{self.temp_standards[zone_type]['적정'][1]}°C"
                humid_range = f"{self.humid_standards[zone_type]['적정'][0]}~{self.humid_standards[zone_type]['적정'][1]}%"
                standards.append(f"{temp_range} / {humid_range}")
                actions.append('냉방 및 제습기 가동 강화')
            elif '온도' in row['상태']:
                if '위험' in row['상태']:
                    alarm_types.append('온도 위험')
                else:
                    alarm_types.append('온도 상승 경고')
                measurements.append(f"{row['온도(°C)']}°C")
                zone_type = '서버실' if row['구역'] == '서버실' else '일반'
                temp_range = f"{self.temp_standards[zone_type]['적정'][0]}~{self.temp_standards[zone_type]['적정'][1]}°C"
                standards.append(temp_range)
                if row['구역'] == '서버실':
                    actions.append('항온항습기 설정 조정')
                else:
                    actions.append('냉방 가동 강화')
            else:
                if '위험' in row['상태']:
                    alarm_types.append('습도 위험')
                else:
                    alarm_types.append('습도 상승 경고')
                measurements.append(f"{row['습도(%)']}%")
                zone_type = '서버실' if row['구역'] == '서버실' else '일반'
                humid_range = f"{self.humid_standards[zone_type]['적정'][0]}~{self.humid_standards[zone_type]['적정'][1]}%"
                standards.append(humid_range)
                actions.append('제습기 가동')
            
            # 담당자 할당
            if row['구역'] == '서버실':
                managers.append('이서버')
            elif '1층' in row['구역']:
                managers.append('김기술')
            else:
                managers.append('박시설')
        
        # 알람 이력 데이터프레임 생성
        self.alarm_history = pd.DataFrame({
            '알람 ID': alarm_ids,
            '날짜': warnings['날짜'].values,
            '시간': alarm_times,
            '구역': warnings['구역'].values,
            '알람 유형': alarm_types,
            '측정값': measurements,
            '기준치': standards,
            '조치 사항': actions,
            '조치 완료': completions,
            '담당자': managers
        })
    
    def export_to_excel(self, filename='BEMS_데이터.xlsx'):
        """모든 데이터를 Excel 파일로 내보내기"""
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            # 시트 1: 센서 데이터
            self.sensor_data.to_excel(writer, sheet_name='데이터 수집', index=False)
            
            # 시트 2: 에너지 요약
            self.energy_summary.to_excel(writer, sheet_name='에너지 사용량 요약', index=False)
            
            # 시트 3: 장비 상태
            self.equipment_status.to_excel(writer, sheet_name='장비 상태 모니터링', index=False)
            
            # 시트 4: 알람 이력
            self.alarm_history.to_excel(writer, sheet_name='알람 이력', index=False)
            
            # 시트 5: 시각화
            workbook = writer.book
            worksheet = workbook.add_worksheet('시각화')
            worksheet.write('A1', '시각화 시트입니다. Python에서 생성된 차트를 이 시트에 포함시킬 수 있습니다.')
            
            # 각 시트의 형식 설정
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                # 열 너비 조정
                for i, col in enumerate(self.sensor_data.columns if sheet_name == '데이터 수집' else 
                                       self.energy_summary.columns if sheet_name == '에너지 사용량 요약' else
                                       self.equipment_status.columns if sheet_name == '장비 상태 모니터링' else
                                       self.alarm_history.columns if sheet_name == '알람 이력' else ['설명']):
                    max_width = max(
                        len(str(col)),
                        self.sensor_data[col].astype(str).map(len).max() if sheet_name == '데이터 수집' and col in self.sensor_data.columns else
                        self.energy_summary[col].astype(str).map(len).max() if sheet_name == '에너지 사용량 요약' and col in self.energy_summary.columns else
                        self.equipment_status[col].astype(str).map(len).max() if sheet_name == '장비 상태 모니터링' and col in self.equipment_status.columns else
                        self.alarm_history[col].astype(str).map(len).max() if sheet_name == '알람 이력' and col in self.alarm_history.columns else 10
                    )
                    worksheet.set_column(i, i, max_width + 2)
    
    def visualize_data(self):
        """데이터 시각화"""
        # 폰트 설정 (한글 깨짐 방지)
        plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 경우
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        
        # 1. 시간대별 온도 변화 그래프
        plt.figure(figsize=(12, 6))
        for zone in self.sensor_data['구역'].unique():
            zone_data = self.sensor_data[self.sensor_data['구역'] == zone]
            dates_times = zone_data['날짜'] + ' ' + zone_data['시간']
            plt.plot(dates_times, zone_data['온도(°C)'], marker='o', label=zone)
        plt.title('시간대별 구역 온도 변화', fontsize=15)
        plt.xlabel('날짜 및 시간')
        plt.ylabel('온도(°C)')
        plt.axhline(y=26, color='r', linestyle='--', alpha=0.7, label='일반구역 상한')
        plt.axhline(y=24, color='orange', linestyle='--', alpha=0.7, label='서버실 상한')
        plt.axhline(y=18, color='b', linestyle='--', alpha=0.7, label='하한')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('온도변화_그래프.png')
        plt.close()
        
        # 2. 시간대별 습도 변화 그래프
        plt.figure(figsize=(12, 6))
        for zone in self.sensor_data['구역'].unique():
            zone_data = self.sensor_data[self.sensor_data['구역'] == zone]
            dates_times = zone_data['날짜'] + ' ' + zone_data['시간']
            plt.plot(dates_times, zone_data['습도(%)'], marker='o', label=zone)
        plt.title('시간대별 구역 습도 변화', fontsize=15)
        plt.xlabel('날짜 및 시간')
        plt.ylabel('습도(%)')
        plt.axhline(y=60, color='r', linestyle='--', alpha=0.7, label='일반구역 상한')
        plt.axhline(y=55, color='orange', linestyle='--', alpha=0.7, label='서버실 상한')
        plt.axhline(y=40, color='b', linestyle='--', alpha=0.7, label='일반구역 하한')
        plt.axhline(y=45, color='green', linestyle='--', alpha=0.7, label='서버실 하한')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('습도변화_그래프.png')
        plt.close()
        
        # 3. 온도-습도 산점도
        plt.figure(figsize=(10, 8))
        colors = {'사무실 1층': 'blue', '사무실 2층': 'green', '서버실': 'red', '회의실': 'purple'}
        for zone, color in colors.items():
            zone_data = self.sensor_data[self.sensor_data['구역'] == zone]
            plt.scatter(zone_data['온도(°C)'], zone_data['습도(%)'], 
                       c=color, label=zone, alpha=0.7, s=80)
            
        # 일반구역 적정 범위
        plt.axvspan(18, 26, alpha=0.1, color='blue', label='일반구역 적정 온도')
        plt.axhspan(40, 60, alpha=0.1, color='green', label='일반구역 적정 습도')
        
        # 서버실 적정 범위 (점선)
        plt.axvspan(18, 24, alpha=0.1, color='red', label='서버실 적정 온도')
        plt.axhspan(45, 55, alpha=0.1, color='purple', label='서버실 적정 습도')
        
        plt.title('구역별 온도-습도 분포도', fontsize=15)
        plt.xlabel('온도(°C)')
        plt.ylabel('습도(%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('온습도_분포도.png')
        plt.close()
        
        # 4. 에너지 사용량 막대 그래프
        energy_by_zone = self.energy_summary.groupby('구역')['총 전력 사용량(kWh)'].sum().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='구역', y='총 전력 사용량(kWh)', data=energy_by_zone)
        plt.title('구역별 총 전력 사용량', fontsize=15)
        plt.xlabel('구역')
        plt.ylabel('총 전력 사용량(kWh)')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('전력사용량_비교.png')
        plt.close()
        
        return ['온도변화_그래프.png', '습도변화_그래프.png', '온습도_분포도.png', '전력사용량_비교.png']

# 메인 코드 실행 예시
if __name__ == "__main__":
    # BEMS 데이터 객체 생성
    bems = BEMSData()
    
    # 데이터 내용 확인
    print("=== 센서 데이터 샘플 ===")
    print(bems.sensor_data.head())
    print("\n=== 에너지 사용량 요약 ===")
    print(bems.energy_summary.head())
    print("\n=== 장비 상태 모니터링 ===")
    print(bems.equipment_status.head())
    print("\n=== 알람 이력 ===")
    print(bems.alarm_history.head())
    
    # 데이터 시각화
    image_files = bems.visualize_data()
    print(f"\n시각화 파일 생성 완료: {image_files}")
    
    # Excel 파일로 내보내기
    bems.export_to_excel()
    print("\nExcel 파일 생성 완료: BEMS_데이터.xlsx")
