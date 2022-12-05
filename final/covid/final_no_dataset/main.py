import torch
import torchvision.models as models
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable

from dataset import covid_dataset
from args import parse_args
from ModelAE import resnet_encoder, Identity
import csv

from sklearn.metrics import classification_report, confusion_matrix
from confusion import plot_confusion_matrix

def model_init(model:str, args, use_pre_weight = False):

    Resnet = {
        'resnet18': models.resnet18(pretrained=use_pre_weight),
        'resnet50': models.resnet50(pretrained=use_pre_weight),
    }
    """
    Calling model by torchvision default or from modelAE.py may need changed.
    Usually loading weight from autoencdor need to call model from modelAE.py.
    """
    #model = Resnet[model]
    #model.fc = Identity()

    model = resnet_encoder(100)
    for para in model.parameters():
        para.requires_grad = True
    model.load_state_dict(torch.load('encoder.pt'))
    out = nn.Sequential(model,
                        nn.Linear(2048,256),
                        nn.ReLU(),
                        nn.Linear(256, 4))

    return out

def train(dataloader, testloader, model, args, device):
    model.to(device)

    data_weight = covid_dataset('train').getWeight()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0., 0.9))
    loss_fn = nn.CrossEntropyLoss(weight=data_weight).to(device)

    for e in range(args.epoch):
        model.train()

        tbar = tqdm(dataloader)
        correct = 0
        train_loss = 0
        total = 0
        for i, (img, label) in enumerate(tbar):
            img, label = Variable(img).to(device), Variable(label).to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label.long())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predict = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predict == label).sum().item()

        test_acc = eval(testloader, model, device)
        
        with open('loss.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([e, train_loss/total, correct/total, test_acc])

        print('epoch:{:3d}, loss:{:.5f}, train accuracy: {:2.2f}% test accuracy {:2.2f}%'.format(e, train_loss/total, 100*correct/total, test_acc))
        if (e+1)%50 == 0:
            torch.save(model.state_dict(), 'result.pt')
    
    _ = eval(testloader, model, device, cMat = True)
def eval(test_loader, model, device, cMat = False):
    model.eval()

    total = 0
    correct = 0

    y_true = []
    y_pred = []


    for i, (img, label) in enumerate(test_loader):
        img, label = Variable(img).to(device), Variable(label).to(device)
        with torch.no_grad():
            output = model(img)
            _, predict = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predict == label).sum().item()

            true = label.cpu().detach().numpy()
            pred = predict.cpu().detach().numpy()
                
            for i in range(len(true)):
                y_true.append(true[i])
                y_pred.append(pred[i])

    def showConMat():
        cMat = confusion_matrix(y_true, y_pred)
        classes = [0,1,2,3]

        plot_confusion_matrix(cMat,classes=classes, normalize=False)
        #print(classification_report(self.y_true, self.y_pred, labels=classes))

    if cMat:
        print("test accuracy: {:.2f}".format(100*correct/total))
        showConMat()

    else:
        return 100*correct/total

def main():
    args = parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model_init('resnet50', args)
    with open('loss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'train accuracy', 'test accuracy'])

    train_data = covid_dataset('train')
    test_data = covid_dataset('test')

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers= args.num_workers,
                              pin_memory=True)

    test_loader = DataLoader(test_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=True,
                              num_workers= args.num_workers,
                              pin_memory=True)

    train(train_loader, test_loader, model, args, device)

if __name__ =="__main__":
    main()