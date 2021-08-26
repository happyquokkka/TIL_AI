import numpy as np

def celsius_to_fahrenheit(x) :
    return x * 1.8 + 32

data_C = np.array(range(0, 100))
data_F = celsius_to_fahrenheit(data_C)
print(data_F)

inp = int(input("섭씨 온도를 입력하세요 : "))
print("화씨 온도로 ", celsius_to_fahrenheit(inp), "입니다.")