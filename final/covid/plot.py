from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_ac = pd.read_csv('loss/autoencoder_covid_loss.csv')
df_as = pd.read_csv('loss/autoencoder_stl10_loss.csv')
df_kc = pd.read_csv('loss/kmeans_covid_loss.csv')
df_ks = pd.read_csv('loss/kmeans_stl10_loss.csv')
df_p = pd.read_csv('loss/pretrain_loss.csv')
df_s = pd.read_csv('loss/scratch_loss.csv')
df_sc = pd.read_csv('loss/simCLR_covid_loss.csv')
df_ss = pd.read_csv('loss/simCLR_stl10_loss.csv')
df_mc = pd.read_csv('loss/kmeans_covid_mapping_loss.csv')
df_ms = pd.read_csv('loss/kmeans_stl10_mapping_loss.csv')
df_cs = pd.read_csv('loss/classification_stl10_loss.csv')

"""
plt.figure(figsize=(10, 10))
plt.plot(df_ac['epoch'], df_ac['loss'], 'b', label = 'autoencoder_covid')
plt.plot(df_as['epoch'], df_as['loss'], 'b--', label = 'autoencoder_stl10')
plt.plot(df_kc['epoch'], df_kc['loss'],  'y', label = 'kmeans_covid')
plt.plot(df_ks['epoch'], df_ks['loss'], 'y--', label = 'kmeans_stl10')
#plt.plot(df_mc['epoch'], df_mc['loss'], label = 'kmeans_covid mapping')
#plt.plot(df_ms['epoch'], df_ms['loss'], '--', label = 'kmeans_stl10 mapping')
plt.plot(df_p['epoch'], df_p['loss'], 'r', label = 'pretrained')
plt.plot(df_s['epoch'], df_s['loss'], 'k', label = 'scratch')
plt.plot(df_sc['epoch'], df_sc['loss'], 'c', label = 'simCLR_covid')
plt.plot(df_ss['epoch'], df_ss['loss'], 'c--', label = 'simCLR_stl10')
#plt.plot(df_cs['epoch'], df_cs['loss'], 'm', label = 'classification_stl10')

plt.title('loss-epoch',fontsize=30)
plt.xlabel('epoch',fontsize=20)
plt.ylabel('loss',fontsize=20)
plt.legend()
plt.tight_layout()
plt.grid()"""
"""
plt.figure(figsize=(10, 10))
plt.plot(df_ac['epoch'], df_ac['train accuracy'], label = 'autoencoder_covid')
plt.plot(df_as['epoch'], df_as['train accuracy'], '--', label = 'autoencoder_stl10')
plt.plot(df_kc['epoch'], df_kc['train accuracy'], label = 'kmeans_covid')
plt.plot(df_ks['epoch'], df_ks['train accuracy'], '--', label = 'kmeans_stl10')
#plt.plot(df_mc['epoch'], df_mc['train accuracy'], label = 'kmeans_covid mapping')
#plt.plot(df_ms['epoch'], df_ms['train accuracy'], '--', label = 'kmeans_stl10 mapping')
plt.plot(df_p['epoch'], df_p['train accuracy'], label = 'pretrained')
plt.plot(df_s['epoch'], df_s['train accuracy'], '--', label = 'scratch')
plt.plot(df_sc['epoch'], df_sc['train accuracy'], label = 'simCLR_covid')
plt.plot(df_ss['epoch'], df_ss['train accuracy'], '--', label = 'simCLR_stl10')
plt.plot(df_cs['epoch'], df_cs['train accuracy'], label = 'classification_stl10')

plt.title('train accuracy-epoch')
plt.xlabel('epoch')
plt.ylabel('train accuracy')
plt.legend()
plt.tight_layout()
plt.grid()"""
"""
plt.figure(figsize=(10, 10))
plt.plot(df_ac['epoch'], df_ac['test accuracy'], label = 'autoencoder_covid')
#plt.plot(df_as['epoch'], df_as['test accuracy'], '--', label = 'autoencoder_stl10')
plt.plot(df_kc['epoch'], df_kc['test accuracy'], label = 'kmeans_covid')
#plt.plot(df_ks['epoch'], df_ks['test accuracy'], '--', label = 'kmeans_stl10')
#plt.plot(df_mc['epoch'], df_mc['test accuracy'], label = 'kmeans_covid mapping')
#plt.plot(df_ms['epoch'], df_ms['test accuracy'], '--', label = 'kmeans_stl10 mapping')
plt.plot(df_p['epoch'], df_p['test accuracy'], label = 'pretrained')
plt.plot(df_s['epoch'], df_s['test accuracy'], '--', label = 'scratch')
plt.plot(df_sc['epoch'], df_sc['test accuracy'], label = 'simCLR_covid')
#plt.plot(df_ss['epoch'], df_ss['test accuracy'], '--', label = 'simCLR_stl10')
#plt.plot(df_cs['epoch'], df_cs['test accuracy'], label = 'classification_stl10')

plt.title('test accuracy covid-epoch',fontsize=30)
plt.xlabel('epoch',fontsize=20)
plt.ylabel('test accuracy',fontsize=20)
plt.legend()
plt.tight_layout()
plt.grid()

plt.figure(figsize=(10, 10))
#plt.plot(df_ac['epoch'], df_ac['test accuracy'], label = 'autoencoder_covid')
plt.plot(df_as['epoch'], df_as['test accuracy'], label = 'autoencoder_stl10')
#plt.plot(df_kc['epoch'], df_kc['test accuracy'], label = 'kmeans_covid')
plt.plot(df_ks['epoch'], df_ks['test accuracy'],  label = 'kmeans_stl10')
#plt.plot(df_mc['epoch'], df_mc['test accuracy'], label = 'kmeans_covid mapping')
#plt.plot(df_ms['epoch'], df_ms['test accuracy'], '--', label = 'kmeans_stl10 mapping')
plt.plot(df_p['epoch'], df_p['test accuracy'], label = 'pretrained')
plt.plot(df_s['epoch'], df_s['test accuracy'], label = 'scratch')
#plt.plot(df_sc['epoch'], df_sc['test accuracy'], label = 'simCLR_covid')
plt.plot(df_ss['epoch'], df_ss['test accuracy'], label = 'simCLR_stl10')
#plt.plot(df_cs['epoch'], df_cs['test accuracy'], label = 'classification_stl10')

plt.title('test accuracy stl 10-epoch',fontsize=30)
plt.xlabel('epoch',fontsize=20)
plt.ylabel('test accuracy',fontsize=20)
plt.legend()
plt.tight_layout()
plt.grid()"""
"""
plt.figure(figsize=(10, 10))
x = ['autoencoder', 'kmeans', 'simCLR', 'classification', 'scratch', 'pretrained']
y = [df_ac['test accuracy'].max(), df_kc['test accuracy'].max(), df_sc['test accuracy'].max(), 
    df_cs['test accuracy'].max(), df_s['test accuracy'].max(), df_p['test accuracy'].max()]
plt.bar(x, y)
plt.title('max test accuracy-pretrain covid')
plt.xlabel('epoch')
plt.ylabel('test accuracy')
#plt.legend()
plt.tight_layout()


plt.figure(figsize=(10, 10))
x = ['autoencoder', 'kmeans', 'simCLR', 'classification', 'scratch', 'pretrained']
y = [df_as['test accuracy'].max(), df_ks['test accuracy'].max(), df_ss['test accuracy'].max(), 
    df_cs['test accuracy'].max(), df_s['test accuracy'].max(), df_p['test accuracy'].max()]
plt.bar(x, y)
plt.title('max test accuracy-pretrain stl10')
plt.xlabel('epoch')
plt.ylabel('test accuracy')
#plt.legend()
plt.tight_layout()"""

plt.figure(figsize=(10, 10))
x = ['autoencoder', 'kmeans', 'simCLR']
y = [df_ac['test accuracy'].mean(), df_kc['test accuracy'].mean(), df_sc['test accuracy'].mean()]
yerr = [df_ac['test accuracy'].std(), df_kc['test accuracy'].std(), df_sc['test accuracy'].std()]
plt.bar(x, y, yerr=yerr)
plt.title('test accuracy-pretrain on covid', fontsize = 30)
#plt.xlabel('method', fontsize = 20)
plt.ylabel('test accuracy', fontsize = 20)
#plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 10))
x = ['autoencoder', 'kmeans', 'simCLR']
y = [df_as['test accuracy'].mean(), df_ks['test accuracy'].mean(), df_ss['test accuracy'].mean()]
yerr = [df_as['test accuracy'].std(), df_ks['test accuracy'].std(), df_ss['test accuracy'].std()]
plt.bar(x, y, yerr = yerr)
plt.title('test accuracy-pretrain on stl10', fontsize = 30)
#plt.xlabel('epoch', fontsize = 20)
plt.ylabel('test accuracy', fontsize = 20)
#plt.legend()
plt.tight_layout()

plt.show()