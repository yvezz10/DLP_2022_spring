from trainer import Trainer

def main():
    """
    models = ['resnet50']
    pre_weights = [True]
    epochs = 20

    for model in models:
        for pre_weight in pre_weights:
            net = Trainer(model_name = model, use_pre_weight=pre_weight,epochs = epochs, data_path='./test2', momentum=0.9, data_weight=True)

            for epoch in range(epochs):
                net.train()
                net.test()
                net.showInfo(epoch)
                net.writeData(epoch)

            net.saveWeight()
            net.showConMat()"""
    
    net = Trainer(model_name='resnet50', use_pre_weight=True, recreate=False)
    net.loadWeight('./test/resnet50_w_pretrainedweight.pt')
    net.test()
    net.showInfo(0)
    net.showConMat()
        
if __name__ == "__main__":
    main()
