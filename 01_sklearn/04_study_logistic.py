from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# 1. 데이터 준비

# 문제: 공부, 과외 [ , ] -> 합격 or 불합격?
# [1, 0] / [0] (fail)
# [8, 1] / [1] pass
# a 시간 공부하고, b 시간 과외받은 학생은 합격? 불합격?

# 문제상황에 어떤 모델을 적용할지가 가장 어려운 부분임! 

X = [
    [1, 0],
    [2, 0],
    [5, 1],
    [2, 3],
    [3, 3],
    [8, 1],
    [10, 0]
]

y = [
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
]

# 2. 데이터 분할
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)

# 3. 모델 준비
logistic = LogisticRegression()

# 4. 학습
logistic.fit(train_X, np.ravel(train_y))
# np.ravel() 은 뭐하는 함수인지 찾아보기! <숙제>

# 5. 예측 및 평가
pred = logistic.predict(test_X)
print(test_X)
print(pred)
pred_pass = logistic.predict([[4,0]])
print("{} 시간 공부, {} 시간 과외 : {}".format(4, 0, 'pass' if pred_pass == 1 else 'fail'))