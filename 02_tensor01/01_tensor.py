# import tensorflow as tf
import tensorflow.compat.v1 as tf

# print(tf.__version__)

'''
Tensorflow
- tensor    : 데이터 저장 객체
- variable  : Weight, bias
- Operation : H = W * X + b (수식이자 노드) => 그래프(operation들이 연결되면) : 
- Session   : 실행환경(학습)
'''

# 상수노드
node = tf.constant(100)

# Session : 그래프 실행 (runner)
sess = tf.Session()

# 노드(노드가 연결되면 그래프) 실행
print(sess.run(node))