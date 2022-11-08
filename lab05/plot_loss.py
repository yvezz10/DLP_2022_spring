import pandas as pd
import matplotlib.pyplot as plt
import os 

df_loss = pd.read_csv('./loss_record/loss.csv')
df_psnr = pd.read_csv('./loss_record/psnr.csv')

epoch_loss = df_loss['epoch']
total_loss = df_loss['loss']
mse_loss = df_loss['mse']
kld_loss = df_loss['kld']
tfr = df_loss['tfr']
beta = df_loss['beta']

epoch_psnr = df_psnr['epoch']
psnr = df_psnr['ave_psnr']

img_data_root = './plot/mono'
if not os.path.isdir(img_data_root):
    os.makedirs(img_data_root)

plt.figure(figsize=(9, 6))
size = 10
plt.scatter(epoch_loss, total_loss, label = 'total_loss', s = size+1)
plt.scatter(epoch_loss,mse_loss, label = 'mse_loss', s = size)
plt.scatter(epoch_loss, kld_loss, label = 'kld_loss', s = size)
plt.title('loss-epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(os.path.join(img_data_root,'loss.png'))

plt.figure(figsize=(9,6))
plt.plot(epoch_loss, tfr, label = 'tfr')
plt.plot(epoch_loss, beta, label = 'beta' )
plt.title('ratio-epoch')
plt.xlabel('epoch')
plt.ylabel('ratio')
plt.legend()
plt.savefig(os.path.join(img_data_root,'ratio.png'))

plt.figure(figsize=(9,6))
plt.plot(epoch_psnr, psnr, label = 'ave_psnr')
plt.title('psnr-epoch')
plt.xlabel('epoch')
plt.ylabel('ratio')
plt.legend()
plt.savefig(os.path.join(img_data_root,'psnr.png'))

