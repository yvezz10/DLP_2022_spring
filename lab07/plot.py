import matplotlib.pyplot as plt
import pandas as pd
import time

def main():
    df = pd.read_csv('log/log.csv')

    plt.figure(figsize=(15, 6))

    plt.subplot(2,3,1)
    plt.plot(df['epoch'], df['generator loss'])
    plt.title("generator loss")
    plt.xlabel("epoch")
    plt.grid()

    plt.subplot(2,3,2)
    plt.plot(df['epoch'], df['discriminator loss'])
    plt.title("discriminaotor loss")
    plt.xlabel("epoch")
    plt.grid()

    plt.subplot(2,3,3)
    plt.plot(df['epoch'], df['train score'])
    plt.title("train socre")
    plt.xlabel("epoch")
    plt.grid()

    plt.subplot(2,3,4)
    plt.plot(df['epoch'], df['test score'])
    plt.title("test score")
    plt.xlabel("epoch")
    plt.grid()
    
    plt.subplot(2,3,5)
    #plt.plot(df['epoch'], df['new test score'])
    plt.title("new test score")
    plt.xlabel("epoch")
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()