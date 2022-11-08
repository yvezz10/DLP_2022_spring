from torch.utils.data import TensorDataset
import dataloader
from trainer import Trainer

def trainModel(ds_train, ds_test):
    net = Trainer('EEGnet','LeakyReLU',ds_train, ds_test)
    epochs = 500

    for epoch in range(epochs):
        net.train()
        net.test()
        if ((epoch+1)%100 == 0):
            net.showInfo(epoch)

        net.writeData(epoch)

    #net.saveWeight()

def showBestAcc(ds_train, ds_test):
    net = Trainer('EEGnet','LeakyReLU',ds_train, ds_test,recreate=False)
    net.loadWeight('weight.pt')
    net.test()
    net.showInfo()

def main():
    X_train, y_train, X_test, y_test = dataloader.read_bci_data()
    ds_train = TensorDataset(X_train, y_train)
    ds_test = TensorDataset(X_test, y_test)
    #trainModel(ds_train, ds_test)
    showBestAcc(ds_train, ds_test)

if __name__ == "__main__":
    main()