import numpy as np

class dataGen:
    def __init__(self):
        pass

    def generate_linear(self, n = 100):
        pts = np.random.uniform(0, 1, (n,2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0], pt[1]])
            #distance = (pt[0]-pt[1])/1.414
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape(n,1)
    
    def generate_XOR_easy(self):
        inputs = []
        labels = []
        for i in range(11):
            inputs.append([0.1*i, 0.1*i])
            labels.append(0)

            if 0.1*i == 0.5:
                continue

            inputs.append([0.1*i, 1-0.1*i])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape(21,1)

class deepNetFun:
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def relu(self, x):
        x2 = np.copy(x)
        x2[x2<0] = 0
        return x2

    def leaky_relu(self,x):
        x2 = np.copy(x)
        x2[x2<0] = 0.01*x2[x2<0]
        return x2

    def elu(self, x, alpha=1.0):
        x2 = np.copy(x)
        return x2 if x2 >= 0 else alpha*(np.exp(x2) - 1)

    def noAct(self, x):
        return x

    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def de_sigmoid(self, x):
        return np.multiply(x, 1.0-x)

    def de_relu(self, x):
        x2 = np.copy(x)
        x2[x2>=0] = 1
        x2[x2<0] = 0
        return x2

    def de_leaky_relu(self,x):
        x2 = np.copy(x)
        x2[x2>=0] = 1
        x2[x2<0] = 0.01
        return x2
        
    def de_noAct(self, x):
        return 1
    
    def de_tanh(self,x):
        return (1-deepNetFun.tanh(self, x)**2)
            
    def lossFun(self, y, y_pred):
        total = (y-y_pred)**2
        loss = np.mean(total)
        return loss

    def layerForward(self, preHidden, weight, bias):
        y_current = np.matmul(np.transpose(weight),preHidden)+ bias
        current_hidden = deepNetFun.sigmoid(self,y_current)
        return y_current, current_hidden

class deepLearning:

    def __init__(self, input, h1, h2, output):
        self.size_input = input
        self.size_h1 = h1
        self.size_h2 = h2
        self.size_output = output

        np.random.seed(2)
        self.weight1 = np.random.rand(input, h1)
        self.weight2 = np.random.rand(h1,h2)
        self.weight3 = np.random.rand(h2,output)

    def forwardPropagation(self, inputlayer):

        self.data_size = len(inputlayer)

        self.bias1 = np.random.rand(self.size_h1, self.data_size)
        self.bias2 = np.random.rand(self.size_h2, self.data_size)
        self.bias3 = np.random.rand(self.size_output, self.data_size)

        self.input_layer = np.transpose(inputlayer)
        self.preAct1, self.hidden_layer1 = deepNetFun.layerForward(self, self.input_layer, self.weight1, self.bias1)
        self.preAct2, self.hidden_layer2 = deepNetFun.layerForward(self, self.hidden_layer1, self.weight2, self.bias2)
        self.preAct3, self.output_layer = deepNetFun.layerForward(self, self.hidden_layer2, self.weight3, self.bias3)
        self.output = self.output_layer.reshape(len(inputlayer), self.size_output)
        
    def lossEvaluation(self, y):
        self.y = self.output
        self.y_hat = y #ground truth
        self.loss = deepNetFun.lossFun(self,self.y_hat,self.y)
        return self.loss, self.y

    def backwardPropagarion(self):

        grad_y2out = 2*(self.y-self.y_hat)
        grad_3 = grad_y2out * deepNetFun.de_sigmoid(self, self.y)
        self.grad_W3 = np.matmul(self.hidden_layer2,grad_3)
        self.grad_B3 = np.transpose(grad_3)

        grad_out2h2 = np.matmul(self.weight3,np.transpose(grad_3))
        grad_2 = deepNetFun.de_sigmoid(self, self.hidden_layer2)*grad_out2h2
        self.grad_W2 = np.transpose(np.matmul(grad_2, np.transpose(self.hidden_layer1)))
        self.grad_B2 = grad_2

        grad_h22h1 = np.matmul(self.weight2,grad_2)
        grad_1 = deepNetFun.de_sigmoid(self, self.hidden_layer1)*grad_h22h1
        self.grad_W1 = np.transpose(np.matmul(grad_1, np.transpose(self.input_layer)))
        self.grad_B1 = grad_1

    def updateGradient(self, learningRate):
        self.weight1 -= learningRate*self.grad_W1
        self.weight2 -= learningRate*self.grad_W2
        self.weight3 -= learningRate*self.grad_W3

        self.bias1 -= learningRate*self.grad_B1
        self.bias2 -= learningRate*self.grad_B2
        self.bias3 -= learningRate*self.grad_B3

    def evaluatAns(self, result, grTrue):
        pred = np.copy(result)
        pred[pred>=0.5] = 1
        pred[pred<0.5] = 0
        correct = 0
        for i in range(len(pred)):
            if grTrue[i] == pred[i]:
                correct +=1

        accuracy = correct*100/len(pred)

        return pred, accuracy


    def train(self, X, y, epoch = 15000):
        lossHistory = []
        accHistory = []
        for i in range(epoch):
            deepLearning.forwardPropagation(self, X)
            loss, ans = deepLearning.lossEvaluation(self, y)
            y_pred, acc = deepLearning.evaluatAns(self, ans, y)
            deepLearning.backwardPropagarion(self)
            deepLearning.updateGradient(self, learningRate=0.1)

            if i%250 == 0:
                lossHistory.append([i, loss])
                accHistory.append([i, acc])

            if i%500 == 0:
                print("epoch:{:>10d}, loss:{:.8f}, accuracy:{:.2f}%".format(i,loss, acc))

        return y_pred, np.array(lossHistory).reshape(len(lossHistory),2), np.array(accHistory).reshape(len(accHistory),2)

