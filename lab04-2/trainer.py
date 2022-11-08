import torchvision.models as models
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dataloader import RetinopathyLoader

from sklearn.metrics import classification_report, confusion_matrix
from confusion import plot_confusion_matrix

import csv
import copy
import os

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def model_init(model:str, use_pre_weight: bool, featrure_extract = False):

    Resnet = {
        'resnet18': models.resnet18(pretrained=use_pre_weight),
        'resnet50': models.resnet50(pretrained=use_pre_weight)
    }

    model = Resnet[model]
    set_parameter_requires_grad(model=model, feature_extracting=featrure_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)

    return model


class Trainer:
    def __init__(self, model_name: str, use_pre_weight: bool, epochs=1, data_weight = False, recreate = True, 
                data_path = '.', momentum = 0.0, weight_decay = 0.0):
        """
        Args:
            model: 'resnet18', 'resnet50'
            use_pre_weight: bool
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_pre_weight = use_pre_weight
        self.title = "_w_pretrained" if self.use_pre_weight else "_wo_pretrained"
        self.title = self.model_name + self.title 
        self.model = model_init(self.model_name, self.use_pre_weight).to(self.device)
        self.max_acc = 0

        self.lr_rate = 1e-3
        self.epoch = epochs
        self.batch_size = 8 
        self.weight_decay = weight_decay
        self.momentum = momentum 

        trainset = RetinopathyLoader('./data','train')
        self.data_weight = trainset.getWeight() if data_weight else None
        testset = RetinopathyLoader('./data', 'test')
        self.loader_train = DataLoader(dataset=trainset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.loader_test = DataLoader(dataset=testset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.loss_fn = nn.CrossEntropyLoss(weight=self.data_weight).to(self.device)
        #self.optimizer = optim.Adam(self.model.parameters(), lr= self.lr_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr_rate, momentum=self.momentum, weight_decay=self.weight_decay)
    
        #print(self.model)

        self.data_path = data_path
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)

        
        txt_path = os.path.join(self.data_path, self.title+'_log.txt')
        with open(txt_path,'w') as f:
            f.write("using device: {}\n".format(self.device))
            f.write("settings:\nlr rate: {}\nbatch size: {}\n".format(self.lr_rate, self.batch_size))
            f.write("loss function: {}\noptimizer: {}\n".format(self.loss_fn, self.optimizer))
            f.write("network: {}, {}\n".format(self.model_name, "with pretrained weight" if use_pre_weight else "w/o pretrained weight"))
            f.write("data weight: {}\n".format("True" if data_weight else "False"))
            f.write("total epochs: {}".format(self.epoch))

        with open(txt_path, 'r') as f:
            print(f.read())

        if recreate:
            self.csv_path = os.path.join(self.data_path, self.title +'.csv')
            with open(self.csv_path, 'w') as f:
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

        data_num = len(self.loader_train.dataset)
        self.train_acc = 100.*correct/ data_num

    def test(self):

        self.model.eval()
        correct = 0
        self.y_true = []
        self.y_pred = []
        with torch.no_grad():
            for data, targets in self.loader_test:

                data = data.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(data)

                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets.data.view_as(predicted)).sum()

                y_true = targets.cpu().detach().numpy()
                y_pred = predicted.cpu().detach().numpy()
                
                for i in range(len(y_true)):
                    self.y_true.append(y_true[i])
                    self.y_pred.append(y_pred[i])

 
        data_num = len(self.loader_test.dataset)
        self.test_acc = 100.*correct/data_num
        if self.test_acc > self.max_acc:
            self.max_acc = self.test_acc
            self.best_model = copy.deepcopy(self.model.state_dict())

    def showInfo(self, epoch=0):
        try:      
            current = "epoch: {:>2d}/ {} training finished, train accuracy: {:.2f}%, test accuracy {:.2f}%".format(epoch+1,self.epoch, self.train_acc, self.test_acc)
        except:
            current = "test accuracy {:.2f}%".format(self.test_acc)

        print(current)

    def writeData(self, epoch):
        train_acc = eval("{:.2f}".format(self.train_acc))
        test_acc = eval("{:.2f}".format(self.test_acc))
        log = [(epoch+1), train_acc,test_acc]

        with open(self.csv_path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(log)


    def saveWeight(self):
        if self.best_model is not None:
            FILE = os.path.join(self.data_path, self.title+'weight.pt')
            torch.save(self.best_model, FILE)
            print("{} model weight is saved".format(self.title))

    def loadWeight(self, file):
        self.model.load_state_dict(torch.load(file))


    def showConMat(self):
        cMat = confusion_matrix(self.y_true, self.y_pred)
        classes = [0,1,2,3,4]

        plot_confusion_matrix(cMat,classes=classes, normalize=True, title=self.title, data_path=self.data_path)
        #print(classification_report(self.y_true, self.y_pred, labels=classes))