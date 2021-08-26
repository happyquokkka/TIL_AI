import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def celsius_to_fahrenheit(x) :
    return x * 1.8 + 32

data_C = np.array(range(0, 100))
data_F = celsius_to_fahrenheit(data_C)

# 데이터 준비
X = data_C
y = data_F

# 데이터 분할
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.3, random_state=1)
train_X = train_X.reshape(-1, 1)
test_X = test_X.reshape(-1, 1)

# 모델 준비
linear = LinearRegression()

# 학습
linear.fit(train_X, train_y)

# 예측 및 평가
predict = linear.predict(test_X)
pred_f = linear.predict([[40]])
print("40 to fahrenheit : ", pred_f)

# 정확도 검사
# score()은 얼마나 잘 학습되어있는지 평가해주는 함수기 때문에 test_X 와 test_y 를 넣어준다
accuracy = linear.score(test_X.reshape(-1, 1), test_y)
print("accuracy : ", accuracy)