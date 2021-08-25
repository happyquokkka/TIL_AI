from sklearn.linear_model import LinearRegression
# 선형회귀모델
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
X = X.reshape(-1, 1) # 2차원 행렬로 만들어야 machine이 학습할 수 있다!
#print(X)

linear = LinearRegression() # linear 이라는 객체 생성
linear.fit(X, y) # fit은 모델을 얘한테 학습시키는거다~
# linear.fit(X.reshape(-1,1), y) 로 쓸 수 있음

test_X = np.array([6, 7, 8, 9, 10])
predict = linear.predict(test_X.reshape(-1, 1))
print(predict)

plt.plot(test_X, predict)
plt.show()
# 문제와 답을 알려주니 컴퓨터가 스스로 y = Wx + b 의 식에서 W 와 b 를 스스로 파악하여
# 그래프를 그린 것
