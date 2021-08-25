import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# pip install pandas
# 공공데이터포털 -> '교육부 학생건강검사 결과분석' rawdata 서울 2015 다운로드 -> weight_height 로 파일명 변경

# 문제상황 : 여학생의 몸무게를 입력하면, 키를 예측하고 싶다.

# 데이터 준비
pd.set_option("display.width", 300)
pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",30)
df = pd.read_csv("weight_height.csv", encoding="euc-kr")

df = df[['학교명', '학년', '성별', '키', '몸무게']]
df.dropna(inplace=True)
df['grade'] = list(map(lambda x:0 if x[-4:] == "초등학교" else (6 if x[-3:] == "중학교" else 9), df["학교명"])) + df["학년"]
df.drop(["학교명", "학년"], axis="columns", inplace=True)
df.columns = ["gender", "height", "weight"]

# 데이터 분할


# train / test set 분리


# 모델 준비


# 학습


# 예측 및 평가
