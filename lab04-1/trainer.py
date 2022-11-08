import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import csv
import copy

from model import EEGNet, DeepConvNet

class Trainer:
    def __init__(self, model, activate, train_data, test_data, recreate = True):

        DeepNet = {
            'EEGnet' : EEGNet(activate=activate),
            'CNNnet' : DeepConvNet(activate=activate)
        }

        self.lr_rate = 1e-2
        self.batch_size = 128
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DeepNet[model].to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr_rate)
        self.loader_train = DataLoader(train_data, batch_size = self.batch_size, shuffle = True)
        self.loader_test = DataLoader(test_data, batch_size = self.batch_size, shuffle = True)
        self.max_accuracy = 0

        self.model_name = model
        self.activate_name = activate        
        print(self.model)
        print("using device:{}".format(self.device))
        print("settings:\nlr rate: {},  batch size: {}".format(self.lr_rate, self.batch_size))
        print("network: {}, activation function: {}".format(self.model_name, self.activate_name))
        if recreate:
            with open('accuracy_'+self.model_name+"_"+self.activate_name, 'w') as f:
                writer = csv.writer(f)
                header = ['epoch', 'train_acc', 'test_acc']
                writer.writerow(header)

    def train(self):
        
        self.model.train()
        correct = 0

        for data, targets in self.loader_train:
            
            data = data.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.loss_fn(outputs, targets.long())

            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data.view_as(predicted)).sum()

        data_num = len(self.loader_test.dataset)
        self.train_acc = 100.*correct/ data_num

    def test(self):

        self.model.eval()
        correct = 0

        with torch.no_grad():
            for data, targets in self.loader_test:

                data = data.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(data)              

                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets.data.view_as(predicted)).sum()
 
        data_num = len(self.loader_test.dataset)
        self.test_acc = 100.*correct/data_num

        if self.test_acc > self.max_accuracy:
            self.best_model = copy.deepcopy(self.model.state_dict())
            self.max_accuracy = self.test_acc

        if eval("{:.2f}".format(self.test_acc)) >=85:
            self.lr_rate = 1e-5

        elif eval("{:.2f}".format(self.test_acc)) >=83:
            self.lr_rate = 1e-4

        elif eval("{:.2f}".format(self.test_acc)) >=78:
            self.lr_rate = 1e-3

    def showInfo(self, epoch=0):
        try:      
            current = "epoch: {:>4d} training finished, train accuracy: {:.2f}%, test accuracy {:.2f}%".format(epoch+1, self.train_acc, self.test_acc)
        except:
            current = "best test accuracy {:.2f}%".format(self.test_acc)

        print(current)

    def writeData(self, epoch):
        train_acc = eval("{:.2f}".format(self.train_acc))
        test_acc = eval("{:.2f}".format(self.test_acc))
        log = [(epoch+1), train_acc,test_acc]

        with open('accuracy_'+self.model_name+"_"+self.activate_name, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(log)

    def saveWeight(self):
        FILE = 'weight.pt'
        torch.save(self.best_model, FILE)

    def loadWeight(self, file):
        self.model.load_state_dict(torch.load(file))
        


