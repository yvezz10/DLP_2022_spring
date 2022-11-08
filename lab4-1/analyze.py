import pandas as pd
import matplotlib.pyplot as plt

df_cnn_elu = pd.read_csv("accuracy_CNNnet_ELU")
df_cnn_rel = pd.read_csv("accuracy_CNNnet_ReLU")
df_cnn_lea = pd.read_csv("accuracy_CNNnet_LeakyReLU")

df_eeg_elu = pd.read_csv("accuracy_EEGnet_ELU")
df_eeg_rel = pd.read_csv("accuracy_EEGnet_ReLU")
df_eeg_lea = pd.read_csv("accuracy_EEGnet_LeakyReLU")

plt.figure(figsize=(8,10))
plt.subplot(2,1,1)
plt.plot(df_eeg_elu['epoch'], df_eeg_elu['train_acc'], label="ELU_train")
plt.plot(df_eeg_elu['epoch'], df_eeg_elu['test_acc'], label="ELU_test")
plt.plot(df_eeg_rel['epoch'], df_eeg_rel['train_acc'], label="ReLU_train")
plt.plot(df_eeg_rel['epoch'], df_eeg_rel['test_acc'], label="ReLU_test")
plt.plot(df_eeg_lea['epoch'], df_eeg_lea['train_acc'], label="LeakyReLU_train")
plt.plot(df_eeg_lea['epoch'], df_eeg_lea['test_acc'], label="LeakyReLU_test")
plt.title("Activation function comparison(EEGNet)")
plt.xlabel("epoch")
plt.ylabel("accuracy(%)")
plt.legend()

plt.subplot(2,1,2)
plt.plot(df_cnn_elu['epoch'], df_cnn_elu['train_acc'], label="ELU_train")
plt.plot(df_cnn_elu['epoch'], df_cnn_elu['test_acc'], label="ELU_test")
plt.plot(df_cnn_rel['epoch'], df_cnn_rel['train_acc'], label="ReLU_train")
plt.plot(df_cnn_rel['epoch'], df_cnn_rel['test_acc'], label="ReLU_test")
plt.plot(df_cnn_lea['epoch'], df_cnn_lea['train_acc'], label="LeakyReLU_train")
plt.plot(df_cnn_lea['epoch'], df_cnn_lea['test_acc'], label="LeakyReLU_test")
plt.title("Activation function comparison(DeepConvNet)")
plt.xlabel("epoch")
plt.ylabel("accuracy(%)")
plt.legend()



print("max of CNNnet, ELU: {}, ReLU: {}, LeakyReLU: {}".format(df_cnn_elu['test_acc'].max(), df_cnn_rel['test_acc'].max(), df_cnn_lea['test_acc'].max()))
print("max of EGGnet, ELU: {}, ReLU: {}, LeakyReLU: {}".format(df_eeg_elu['test_acc'].max(), df_eeg_rel['test_acc'].max(), df_eeg_lea['test_acc'].max()))
plt.show()