import numpy as np
import pickle
from keras.datasets import mnist
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class NN:
    def __init__(self, train=False):
        if train:
            self.W0 = np.random.rand(240, 784) - 0.5
            self.B0 = np.random.rand(240, 1) - 0.5

            self.W1 = np.random.rand(80, 240) - 0.5
            self.B1 = np.random.rand(80, 1) - 0.5

            self.W2 = np.random.rand(10, 80) - 0.5
            self.B2 = np.random.rand(10, 1) - 0.5
        else:
            with open("W0.pickle", 'rb') as f:
                self.W0 = pickle.load(f)
            with open("W1.pickle", 'rb') as f:
                self.W1 = pickle.load(f)
            with open("W2.pickle", 'rb') as f:
                self.W2 = pickle.load(f)
            with open("B0.pickle", 'rb') as f:
                self.B0 = pickle.load(f)
            with open("B1.pickle", 'rb') as f:
                self.B1 = pickle.load(f)
            with open("B2.pickle", 'rb') as f:
                self.B2 = pickle.load(f)

        self.alpha = 0.1


    def forward(self, p):
        self.p = p

        self.in_0 = self.W0.dot(self.p.T) + self.B0 # 240 x m
        self.out_0 = ReLU(self.in_0) # 240 x m

        self.in_1 = self.W1.dot(self.out_0) + self.B1 # 80 x m
        self.out_1 = ReLU(self.in_1) # 80 x m

        self.in_2 = self.W2.dot(self.out_1) + self.B2 # 10 x m
        self.out_2 = softmax(self.in_2) # 10 x m

        return np.argmax(self.out_2, axis=0)

    def backward(self, act):
        self.diff = self.out_2 - onehot(act)
        self.i0 = self.diff.dot(self.out_1.T) / act.size
        self.b0 = np.sum(self.diff) / act.size

        self.relu0 = self.W2.T.dot(self.diff) * (self.in_1 > 0)
        self.i1 = self.relu0.dot(self.out_0.T) / act.size
        self.b1 = np.sum(self.relu0) / act.size

        self.relu1 = self.W1.T.dot(self.relu0) * (self.in_0 > 0)
        self.i2 = self.relu1.dot(self.p) / act.size
        self.b2 = np.sum(self.relu1) / act.size

        self.W0 -= self.alpha * self.i2
        self.B0 -= self.alpha * self.b2
        self.W1 -= self.alpha * self.i1
        self.B1 -= self.alpha * self.b1
        self.W2 -= self.alpha * self.i0
        self.B2 -= self.alpha * self.b0

    def classify(self, p):
        return self.forward(p)

    def dump(self):
        with open("W0.pickle", 'wb') as f:
            pickle.dump(self.W0, f)
        with open("W1.pickle", 'wb') as f:
            pickle.dump(self.W1, f)
        with open("W2.pickle", 'wb') as f:
            pickle.dump(self.W2, f)
        with open("B0.pickle", 'wb') as f:
            pickle.dump(self.B0, f)
        with open("B1.pickle", 'wb') as f:
            pickle.dump(self.B1, f)
        with open("B2.pickle", 'wb') as f:
            pickle.dump(self.B2, f)

def ReLU(a):
    return np.maximum(a, 0)

def softmax(a):
    e = np.exp(a)
    return e / sum(e)

def onehot(x):
    a = np.zeros((x.size, 10))
    a[np.arange(x.size), x] = 1
    return a.T

class Flip:
    def __init__(self, f, a, img):
        self.p = img.reshape(img.shape[0], img.shape[1]*img.shape[2]) / 256
        self.images = img
        self.i = 0
        self.a = a
        self.f = f
        a.imshow(self.images[self.i]) # Display the first image
        a.set_title(f"Prediction: {n.classify(self.p[self.i].reshape((1, 784)))}")
        f.canvas.draw()
        f.canvas.flush_events()

    def cb_left(self, event):
        self.i = (self.i - 1) % 100
        self.a.imshow(self.images[self.i]) # Display the first image
        self.a.set_title(f"Prediction: {n.classify(self.p[self.i].reshape((1, 784)))}")
        f.canvas.draw()
        f.canvas.flush_events()

    def cb_right(self, event):
        self.i = (self.i + 1) % 100
        self.a.imshow(self.images[self.i]) # Display the first image
        self.a.set_title(f"Prediction: {n.classify(self.p[self.i].reshape((1, 784)))}")
        f.canvas.draw()
        f.canvas.flush_events()

if __name__ == "__main__":
    (xt, yt), (xv, yv) = mnist.load_data()
    p = argparse.ArgumentParser()
    p.add_argument("-t", help="Train Mode: whether to 'train' the model (doesn't write the weights for safety).")
    p.add_argument("-e", help="Epochs: number of epochs.")
    o = p.parse_args()
    if o.t is None:
        p.error("-t must be used.")
        exit(0)

    
    if o.t == 'True':
        if o.e is None:
            p.error("-e must be used.")
            exit(0)

        n = NN(train=True)
        p = xt.reshape(xt.shape[0], xt.shape[1]*xt.shape[2]) / 256

        for k in range(int(o.e)):
            e = n.forward(p)
            n.backward(yt)
            print(f"epoch: {k}/{int(o.e)} acc = {(np.sum(e == yt) / yt.size)*100}%", end='\r')

        print()

        p = xv.reshape(xv.shape[0], xv.shape[1]*xv.shape[2]) / 256
        e = n.forward(p)
        print(f"acc = {(np.sum(e == yv) / yv.size)*100}%", end='\r')

        # n.dump()

    else:
        n = NN()

        # p = xv.reshape(xv.shape[0], xv.shape[1]*xv.shape[2]) / 256
        # e = n.forward(p)
        # print(f"acc = {(np.sum(e == yv) / yv.size)*100}%", end='\r')

        f, a = plt.subplots()
        flip = Flip(f, a, xv)
        right_a = plt.axes([0.7, 0.05, 0.2, 0.075])
        left_a = plt.axes([0.1, 0.05, 0.2, 0.075])
        right = Button(right_a, "next")
        left = Button(left_a, "prev")
        right.on_clicked(flip.cb_right)
        left.on_clicked(flip.cb_left)
        plt.show()
