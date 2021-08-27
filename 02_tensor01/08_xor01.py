import tensorflow.compat.v1 as tf

# 1. 데이터 준비
# 2. 데이터 분할
train_X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

train_y = [
    [0],
    [1],
    [1],
    [0]
]

X = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 3. 준비
# 가설
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

logit = tf.matmul(X, W) + b
# activation function : 이진분류이므로 sigmoid function 사용
H = tf.sigmoid(logit)

# loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y))

# optimizer
learning_rate = 0.1
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Session 객체 형성 및 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 4. 학습
epochs = 30000
for step in range(epochs) :
    _, loss_val = sess.run([train, loss], feed_dict={X: train_X, y: train_y})
    if step % 300 == 0:
        print("loss : {}".format(loss_val))

# 5. 예측 및 평가
predict = tf.cast(H > 0.5, dtype=tf.float32)
# predict 의 실제값과 얼마나 같은지
equal = tf.cast(tf.equal(predict, y), dtype=tf.float32)
accuracy = tf.reduce_mean(equal)

print("정확도 : ", sess.run(accuracy, feed_dict={X: train_X, y: train_y}))
# print("예측 : ", sess.run(H, feed_dict={X: [[0,1]]}))
# 예측의 의미가 없어서 안 함 ㅠ