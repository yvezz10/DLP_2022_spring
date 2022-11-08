import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_resnet18_wi = pd.read_csv('./test/resnet18_w_pretrained.csv')
df_resnet18_wo = pd.read_csv('./test/resnet18_wo_pretrained.csv')
df_resnet50_wi = pd.read_csv('./test/resnet50_w_pretrained.csv')
df_resnet50_wo = pd.read_csv('./test/resnet50_wo_pretrained.csv')

plt.figure(figsize=(8,10))
plt.subplot(2,1,1)
plt.plot(df_resnet18_wi['epoch'], df_resnet18_wi['train_acc'], label="train with pretrained")
plt.plot(df_resnet18_wi['epoch'], df_resnet18_wi['test_acc'], label="test with pretrained")
plt.plot(df_resnet18_wo['epoch'], df_resnet18_wo['train_acc'], label="train w/o pretrained")
plt.plot(df_resnet18_wo['epoch'], df_resnet18_wo['test_acc'], label="test w/o pretrained")
plt.xticks(np.arange(1,21,1))
plt.title("Resnet 18")
plt.xlabel("epoch")
plt.ylabel("accuracy(%)")
plt.legend()

plt.subplot(2,1,2)
plt.plot(df_resnet50_wi['epoch'], df_resnet50_wi['train_acc'], label="train with pretrained")
plt.plot(df_resnet50_wi['epoch'], df_resnet50_wi['test_acc'], label="test with pretrained")
plt.plot(df_resnet50_wo['epoch'], df_resnet50_wo['train_acc'], label="train w/o pretrained")
plt.plot(df_resnet50_wo['epoch'], df_resnet50_wo['test_acc'], label="test w/o pretrained")
plt.xticks(np.arange(1,21,1))
plt.title("Resnet 50")
plt.xlabel("epoch")
plt.ylabel("accuracy(%)")
plt.legend()

print("max of resnet18, with pretrained: {}, w/o pretrained: {}".format(df_resnet18_wi['test_acc'].max(), df_resnet18_wo['test_acc'].max()))
print("max of resnet50, with pretrained: {}, w/o pretrained: {}".format(df_resnet50_wi['test_acc'].max(), df_resnet50_wo['test_acc'].max()))
plt.savefig('./test/compare.png')
plt.show()