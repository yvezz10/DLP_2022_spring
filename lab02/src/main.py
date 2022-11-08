import matplotlib.pyplot as plt
from helperfun import dataGen, deepLearning

def show_result(x, y, pred_y):
    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0.:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')


def show_history(loss, accuracy):
    plt.figure(figsize=(8,8))

    plt.subplot(2,1,1)
    plt.title('Loss - Epoch', fontsize = 18)
    plt.plot(loss[:,0], loss[:,1])

    plt.subplot(2,1,2)
    plt.title('Accuracy - Epoch', fontsize = 18)
    plt.plot(accuracy[:,0], accuracy[:,1], 'c')


data = dataGen()

def main():
    X, y = data.generate_linear()

    two_layer_network = deepLearning(2,7,7,1)
    y_pred, loss, acc = two_layer_network.train(X,y)

    show_result(X, y, y_pred)
    show_history(loss, acc)

    plt.show()
    
if __name__ == '__main__':
    main()