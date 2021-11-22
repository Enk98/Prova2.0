import tensorflow as tf
import pandas as pd
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
#operazioni sui tensori
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

#operazioni sui dati
data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
#Separare prima le caselle con testo in 0 1 e poi fare la media su le rimanenti che non possiedono un valore definito
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

inputs = inputs.fillna(inputs.mean())
print(inputs)
print(outputs)

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
print(X)
print(y)
