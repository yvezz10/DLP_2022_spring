import pandas as pd 
import csv
import numpy as np
import os
import random

df_nm = pd.read_excel('dataset/Normal.xlsx')#0
df_lo = pd.read_excel('dataset/Lung_Opacity.xlsx') #1 
df_vp = pd.read_excel('dataset/Viral_Pneumonia.xlsx') #2
df_cv = pd.read_excel('dataset/COVID.xlsx') #3

train_img = []
train_label = []
test_img = []
test_label = []

def getData(df, train_test_ratio = 0.8):
    for im in df[0]['FILE NAME']:
        if random.random()<train_test_ratio:
            train_img.append(im)
            train_label.append(df[1])
        else:
            test_img.append(im)
            test_label.append(df[1])

dataframes = [[df_nm, 0], [df_lo, 1], [df_vp, 2], [df_cv, 3]]

for data in dataframes:
    getData(data)

train_img = np.reshape(train_img, (len(train_img), 1))
train_label = np.reshape(train_label, (len(train_label), 1))
test_img = np.reshape(test_img, (len(test_img), 1))
test_label = np.reshape(test_label, (len(test_label), 1))

with open('train_image.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['train_image'])
    writer.writerows(train_img)

with open('train_label.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['train_label'])
    writer.writerows(train_label)

with open('test_image.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['test_image'])
    writer.writerows(test_img)

with open('test_label.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['test_label'])
    writer.writerows(test_label)