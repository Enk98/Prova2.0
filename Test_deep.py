import tensorflow as tf

x = tf.range(15)
y = tf.range(10)
print(x)
X = tf.reshape(x, (3, 5))
Y = tf.reshape(y, (2, -1))
print(X)
print(Y)
print(tf.ones((3, 4)))
print(tf.ones((3, 4, 5)))
print(tf.concat([X, Y], axis=0))
#print(tf.concat([X,Y], axis=1)) non si puo fare perche non sono comparabili come axis
