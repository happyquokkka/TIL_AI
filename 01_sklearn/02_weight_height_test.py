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

# 학년 표기 변경 : 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 로 변경
df['grade'] = list(map(lambda x:0 if x[-4:] == "초등학교" else (6 if x[-3:] == "중학교" else 9), df["학교명"])) + df["학년"]
# 얘는 왜 list로 변환하는거지? df["학년"] list와 더하려고?
df.drop(["학교명", "학년"], axis="columns", inplace=True)

# 컬럼명 변경
df.columns = ["gender", "height", "weight", "grade"]

# df['gender'] 의 값을, 남 -> 0, 여 -> 1로 변환
df['gender'] = df['gender'].map(lambda x:0 if x=="남" else 1)
# print(df)

# 여자인 df만 분리
is_girl = df["gender"] == 1
girl_df = df[is_girl]
# print(girl_df)


# 데이터 분할
X = girl_df['weight']
y = girl_df['height']

# train / test set 분리
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)
# 2차원 데이터로 가공
train_X = train_X.values.reshape(-1, 1)
test_X = test_X.values.reshape(-1, 1)

# 모델 준비 (선형회귀분석모델)
linear = LinearRegression()

# 학습
linear.fit(train_X, train_y)

# 예측 및 평가
predict = linear.predict(test_X)

# 그래프 그리기
plt.plot(test_X, test_y, "b.")
plt.plot(test_X, predict, "r.")

plt.xlim(10, 140)
plt.ylim(100, 220)
plt.grid()
# plt.show()

# 몸무게가 50kg인 사람의 키는?
pred_grd = linear.predict([[60]])
print("여학생 키 예측 :", pred_grd)