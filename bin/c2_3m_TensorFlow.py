import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)


#--Tf with constant coefficients
#Tf overwrite the basic operators
#The 2 formula are the same
cost = tf.add(tf.add(w ** 2, tf.multiply(-10.0, w)), 25)
cost = w ** 2 - 10 * w + 25

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))
#The same as:
# with tf.Session() as session:
#   session.run(init)
#   print(session.run(w))

session.run(train)
print(session.run(w))

for i in range(1000):
    session.run(train)
print(session.run(w))


#--Tf with data coefficients
coefficients = np.array([[1.0], [-10.0], [25.0]])
x = tf.placeholder(tf.float32, [3, 1])
cost = x[0][0] * w ** 2 + x[1][0] * w + x[2][0]

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))

session.run(train, feed_dict=[x:coefficient])
print(session.run(w))

for i in range(1000):
    session.run(train, feed_dict=[x:coefficient])
print(session.run(w))
