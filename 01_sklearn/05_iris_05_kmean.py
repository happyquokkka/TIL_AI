from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 준비
iris = load_iris()
X = iris.data
y = iris.target

# 2. 데이터 분할
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)

# 3. 모델 준비
mean = KMeans(n_clusters=3)
# cluster 개수를 모를 때도 있다!

# 4. 학습
mean.fit(train_X)

# 5. 예측 및 평가
pred = mean.predict(test_X)
# print(pred)
# predic = mean.predict(np.array([5.0, 3.1, 0.2, 1.0], dtype=np.float64).reshape(1, -1))
# print(predic)

df = pd.DataFrame(test_X)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df["category"] = pd.DataFrame(pred)
# print(df)

centers = pd.DataFrame(mean.cluster_centers_) # centroids
centers.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
center_X = centers["sepal_length"]
center_y = centers["sepal_width"]

plt.scatter(df["sepal_length"], df["sepal_width"], c=df["category"])
plt.scatter(center_X, center_y, s=100, c='r', marker="*")
plt.show()

# pred와 test_y의 값을 비교하면 평가를 할 수는 있으나 굳이 할 필요는 없다
