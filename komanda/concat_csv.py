import os
import csv

hmb1 = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\data\\data\\hmb1\\interpolated.csv"
hmb2 = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\data\\data\\hmb2\\interpolated.csv"
hmb4 = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\data\\data\\hmb4\\interpolated.csv"
hmb5 = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\data\\data\\hmb5\\interpolated.csv"
hmb6 = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\data\\data\\hmb6\\interpolated.csv"
hmb = [hmb1, hmb2, hmb4, hmb5, hmb6]
path = 'interpolated_concat.csv'

tf = open(path, 'a')
#print(h[1:])
tf.write(open(hmb1).read())
tf.write(open(hmb2).read())
tf.write(open(hmb4).read())
tf.write(open(hmb5).read())
tf.write(open(hmb6).read())
tf.close()

for i in hmb:
    with open(i, 'r', newline='') as f:
        print(str(len(list(csv.reader(f,delimiter = ",")))) + " data points in " + i)

with open(path, 'r') as f:
    reader = csv.reader(f,delimiter = ",")
    data = list(reader)
    n = len(data)
    print(str(n) + " data points in interpolated_concat.csv")