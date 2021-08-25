import matplotlib.pyplot as plt

# pip install matplotlib

def f(x):
    y = [i*2 for i in x]
    return y
# 내가한짓..
# x = x*2
# return x

X = [1, 2, 3, 4, 5]
y = f(X)
print(y)
# [2, 4, 6, 8, 10]

plt.plot(X, y)
plt.show()