import tensorflow.compat.v1 as tf

# 1. 데이터 준비
# 2. 데이터 분할 ㅡ 공간 할당
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 3. 준비
# 가설 설정
# W 와 b 는 변수!
# H = W * X + b
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
H = W * X + b

# loss function (cost function) : 가설과 실제값의 차이를 제곱한 값
loss = tf.reduce_mean(tf.square(H-y))

# optimizer
# 경사 하강법 (Gradient Descent) : 가설과 실제값의 차이를 제곱해서 평균으로 가져와 loss function이 최소가 되도록!
# learning_rate : 얼만큼씩 내려갈건지를 () 안에 써 준다
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# Session
# 변수 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 4. 학습
# 학습 횟수 : epochs
epochs = 10000
for step in range(epochs) :
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={X: [1, 2, 3, 4], y: [3, 5, 7, 9]})
    if step % 500 == 0:
        print("W : {} \t b : {} \t loss : {}".format(W_val, b_val, loss_val))

# 5. 예측 및 평가
print(sess.run(H, feed_dict={X: [13, 14, 15, 16]}))