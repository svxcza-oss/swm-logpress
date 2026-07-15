import pandas as pd
from drain3 import TemplateMiner
#여기에서 cvs파일 불러서 돌리면 됨.

# 1. B트랙이 줬다고 가정하는 가상의 로그 데이터 (나중엔 pd.read_csv('파일.csv')로 바꿈)
log_data = [
    "User Macui failed to login from IP 192.168.0.1",
    "User Admin failed to login from IP 10.0.0.55",
    "Database connection lost at port 5432",
    "User Guest failed to login from IP 172.16.0.3",
    "Database connection lost at port 3306"
]
df = pd.DataFrame({'log_message': log_data})

print("=== 🚀 드레인3 작업 시작 ===")

# 2. 빈칸 뚫기 전문가(Miner) 소환
miner = TemplateMiner()

# 3. 로그를 한 줄씩 전문가에게 던져주고 결과(템플릿) 받아오기
templates = []
for log in df['log_message']:
    result = miner.add_log_message(log)
    templates.append(result['template_mined']) # 빈칸 뚫린 문장만 쏙 빼서 저장

# 4. 원래 엑셀(데이터프레임) 옆칸에 결과 붙여넣기
df['template'] = templates

print("\n=== 📊 판다스 통계(희귀도) 결과 ===")
# 5. 어떤 템플릿이 몇 번 나왔는지 개수 세기!
print(df['template'].value_counts())