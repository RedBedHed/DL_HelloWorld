import numpy as np
import pickle
from keras.datasets import mnist
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class Loader:
    def __init__(self, data, lbl, sz, b_sz):
        self.div = (sz // b_sz)
        self.batch_sz = sz // self.div
        self.data = data
        self.lbl = lbl
        self.i = 0

    def next(self):
        if self.i >= self.div:
            perm = np.random.permutation(len(self.data))
            self.data = self.data[perm]
            self.lbl = self.lbl[perm]
            self.i = 0
        idx = self.i * self.batch_sz
        self.i += 1
        return self.data[idx: idx + self.batch_sz], self.lbl[idx: idx + self.batch_sz]

class NN:
    def __init__(self, train=False):
        if train:
            self.W0 = np.random.rand(240, 784) - 0.5
            self.B0 = np.random.rand(240, 1) - 0.5

            self.W1 = np.random.rand(80, 240) - 0.5
            self.B1 = np.random.rand(80, 1) - 0.5

            self.W2 = np.random.rand(10, 80) - 0.5
            self.B2 = np.random.rand(10, 1) - 0.5

            self.VW0 = np.ones((240, 784))
            self.VB0 = np.ones((240, 1))

            self.VW1 = np.ones((80, 240))
            self.VB1 = np.ones((80, 1))

            self.VW2 = np.ones((10, 80))
            self.VB2 = np.ones((10, 1))

        else:
            with open("pickles/W0.pickle", 'rb') as f:
                self.W0 = pickle.load(f)
            with open("pickles/W1.pickle", 'rb') as f:
                self.W1 = pickle.load(f)
            with open("pickles/W2.pickle", 'rb') as f:
                self.W2 = pickle.load(f)
            with open("pickles/B0.pickle", 'rb') as f:
                self.B0 = pickle.load(f)
            with open("pickles/B1.pickle", 'rb') as f:
                self.B1 = pickle.load(f)
            with open("pickles/B2.pickle", 'rb') as f:
                self.B2 = pickle.load(f)

        self.alpha = 0.001
        self.ff = 0.95
        self.e = 0.000001

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
        # Calculate dl/dw dl/db
        dz2 = self.out_2 - onehot(act)
        dw2 = dz2.dot(self.out_1.T) / act.size
        db2 = np.sum(dz2) / act.size

        dz1 = self.W2.T.dot(dz2) * (self.in_1 > 0)
        dw1 = dz1.dot(self.out_0.T) / act.size
        db1 = np.sum(dz1) / act.size

        dz0 = self.W1.T.dot(dz1) * (self.in_0 > 0)
        dw0 = dz0.dot(self.p) / act.size
        db0 = np.sum(dz0) / act.size

        # RMSProp optimization.
        self.VW0 = self.ff * (self.VW0) + (1 - self.ff) * ((dw0) ** 2)
        self.VB0 = self.ff * (self.VB0) + (1 - self.ff) * ((db0) ** 2)

        self.VW1 = self.ff * (self.VW1) + (1 - self.ff) * ((dw1) ** 2)
        self.VB1 = self.ff * (self.VB1) + (1 - self.ff) * ((db1) ** 2)

        self.VW2 = self.ff * (self.VW2) + (1 - self.ff) * ((dw2) ** 2)
        self.VB2 = self.ff * (self.VB2) + (1 - self.ff) * ((db2) ** 2)

        self.W0 -= ( self.alpha / (np.sqrt( self.VW0 ) + self.e) ) * dw0
        self.B0 -= ( self.alpha / (np.sqrt( self.VB0 ) + self.e) ) * db0
        self.W1 -= ( self.alpha / (np.sqrt( self.VW1 ) + self.e) ) * dw1
        self.B1 -= ( self.alpha / (np.sqrt( self.VB1 ) + self.e) ) * db1
        self.W2 -= ( self.alpha / (np.sqrt( self.VW2 ) + self.e) ) * dw2
        self.B2 -= ( self.alpha / (np.sqrt( self.VB2 ) + self.e) ) * db2

    def classify(self, p):
        return self.forward(p)

    def dump(self):
        with open("pickles/W0.pickle", 'wb') as f:
            pickle.dump(self.W0, f)
        with open("pickles/W1.pickle", 'wb') as f:
            pickle.dump(self.W1, f)
        with open("pickles/W2.pickle", 'wb') as f:
            pickle.dump(self.W2, f)
        with open("pickles/B0.pickle", 'wb') as f:
            pickle.dump(self.B0, f)
        with open("pickles/B1.pickle", 'wb') as f:
            pickle.dump(self.B1, f)
        with open("pickles/B2.pickle", 'wb') as f:
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
        a.imshow(self.images[self.i])
        a.set_title(f"Prediction: {n.classify(self.p[self.i].reshape((1, 784)))}")
        f.canvas.draw()
        f.canvas.flush_events()

    def cb_left(self, event):
        self.i = (self.i - 1) % 100
        self.a.imshow(self.images[self.i])
        self.a.set_title(f"Prediction: {n.classify(self.p[self.i].reshape((1, 784)))}")
        f.canvas.draw()
        f.canvas.flush_events()

    def cb_right(self, event):
        self.i = (self.i + 1) % 100
        self.a.imshow(self.images[self.i])
        self.a.set_title(f"Prediction: {n.classify(self.p[self.i].reshape((1, 784)))}")
        f.canvas.draw()
        f.canvas.flush_events()

if __name__ == "__main__":
    (xt, yt), (xv, yv) = mnist.load_data()
    p = argparse.ArgumentParser()
    p.add_argument("-t", help="Train Mode: whether to 'train' the model (doesn't write the weights for safety).")
    p.add_argument("-e", help="Epochs: number of epochs.")
    p.add_argument("-b", help="Batch Size: size of a mini batch.")
    o = p.parse_args()

    if o.t is None:
        p.error("-t must be used.")
        exit(0)
    
    if o.t == 'True':
        if o.e is None:
            p.error("-e must be used.")
            exit(0)
        
        if o.b is None:
            p.error("-b must be used.")
            exit(0)

        n = NN(train=True)
        p1 = xt.reshape(xt.shape[0], xt.shape[1]*xt.shape[2]) / 256
        p2 = xv.reshape(xv.shape[0], xv.shape[1]*xv.shape[2]) / 256
        l = Loader(p1, yt, p1.shape[0], int(o.b))
        test_acc_mx    = -1
        train_acc_curr = -1
        tests_acc_curr = -1
        for k in range(int(o.e)):
            ln, lbl = l.next()
            e1 = n.forward(ln)
            n.backward(lbl)
            e2 = n.forward(p2)
            train_acc_curr = (np.sum(e1 == lbl) / int(o.b)) * 100
            tests_acc_curr = (np.sum(e2 ==  yv) /  yv.size) * 100
            print(f"epoch: {k:6}/{int(o.e)} train acc = {round(train_acc_curr, 3):.3f}% test acc = {round(tests_acc_curr, 3):.3f}%", end='\r')
            if tests_acc_curr > 96.0 and tests_acc_curr < test_acc_mx - 1.0:
                break
            if tests_acc_curr > test_acc_mx:
                test_acc_mx = tests_acc_curr

        print()

        # n.dump()

    else:
        n = NN()

        f, a = plt.subplots()
        flip = Flip(f, a, xv)
        right_a = plt.axes([0.7, 0.05, 0.2, 0.075])
        left_a = plt.axes([0.1, 0.05, 0.2, 0.075])
        right = Button(right_a, "next")
        left = Button(left_a, "prev")
        right.on_clicked(flip.cb_right)
        left.on_clicked(flip.cb_left)
        plt.show()
