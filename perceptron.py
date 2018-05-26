# coding = utf-8
# author: biyoner
import numpy as np
import matplotlib.pyplot as plt
import argparse

class simple_perceptron(object):
    def __init__(self, x, y, lr=1):
        self.x = x
        self.y = y
        self.lr = lr
        self.b = 0
        self.w = np.zeros([1,x.shape[1]],int)
    def train(self):
        "k is the iteration times"
        k = 0
        while(1):
            flag = False
            for i in range(self.x.shape[0]):
                k += 1
                if self.y[i]*(np.dot(self.x[i],self.w.T)+self.b)<=0:
                    self.w = self.w + self.lr*self.y[i]*self.x[i]
                    self.b = self.b + self.lr*self.y[i]
                    flag = True
            if not flag:
                break
    def plot(self):
        plt.figure()
        for i in range(len(self.x)):
            if self.y[i] == 1:
                plt.plot(self.x[i][0], self.x[i][1], 'ro')
            else:
                plt.plot(self.x[i][0],self.x[i][1],'bo')
        x_linear = np.linspace(0,10)
        y_linear=[]
        for i in range(len(x_linear)):
            a = (-self.b - (self.w[0][0]) * (x_linear[i])) / (self.w[0][1])
            y_linear.append((-self.b - (self.w[0][0]) * (x_linear[i])) / (self.w[0][1]))
        print y_linear
        plt.plot(x_linear, y_linear)
        plt.show()
        # plt.savefig("picture.png")

class dual_perceptron(object):
    def __init__(self, x, y, lr=1):
        self.x = x
        self.y = y
        self.lr = lr
        self.b = 0
        self.alpha = np.zeros(x.shape[0],int)
    def gramMatrix(self,x):
        gram = x.dot(x.T)
        return gram
    def train(self):
        k = 0
        gramMatrix = self.gramMatrix(self.x)
        while (1):
            flag = False
            for i in range(self.x.shape[0]):
                k += 1
                if self.y[i]*(np.sum(self.alpha*self.y*gramMatrix[i])+self.b)<= 0:
                    self.alpha[i] = self.alpha[i] + self.lr
                    self.b = self.b+self.lr * self.y[i]
                    flag = True
            if not flag:
                break

    def plot(self):
        plt.figure()
        for i in range(len(self.x)):
            if self.y[i] == 1:
                plt.plot(self.x[i][0], self.x[i][1], 'ro')
            else:
                plt.plot(self.x[i][0],self.x[i][1],'bo')
        x_linear = np.linspace(0,10)
        y_linear=[]
        w = (self.alpha*self.y.T).dot(self.x)
        for i in range(len(x_linear)):
            y_linear.append((-self.b - (w[0]) * (x_linear[i])) / (w[1]))
        plt.plot(x_linear, y_linear)
        plt.show()


def main():
    x = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])

    parser = argparse.ArgumentParser(description='Choose different types of perceptrons')
    parser.add_argument('--choice', type=str, default=None)
    args = parser.parse_args()

    if args.choice == 'dual':
        p = dual_perceptron(x, y)
        p.train()
        p.plot()
    elif args.choice == 'simple':
        p = simple_perceptron(x, y)
        p.train()
        p.plot()
    else:
        print "Please choose correctly."



if __name__ == '__main__':

    main()





