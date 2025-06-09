import tkinter as tk
import tkinter.font as tf
import numpy as np
import cv2
import nn

MAX = 255
WIN = 728
INC = 26
FONT_SZ = 50
MNIST_SZ = 28
MNIST_FLAT = 784
KERNEL_SZ = 3

class Sketch:
    def __init__(self):
        self.moore = [(1, 1), (-1, -1), (1, -1), (-1, 1), (1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        self.back = np.asarray([[0] * MNIST_SZ] * MNIST_SZ, dtype=np.uint8)
        self.nn = nn.NN()
        self.x, self.y = 0, 0

        w = tk.Tk()
        w.title('sketch')
        w.geometry("850x870")
        w.resizable(width=False, height=False)
        self.button = tk.Button(w, text="Clear", command=self.onclick)
        self.button.pack(pady=10)
        self.canvas = tk.Canvas(w, width=WIN, height=WIN, bg="white")
        self.canvas.pack()

        self.canvas.bind("<Button>", self.draw)
        self.canvas.bind("<B1-Motion>", self.draw)

        f = tf.Font(family="Arial", size=FONT_SZ)
        self.label = tk.Label(w, text="Draw a number!", font=f)
        self.label.pack(pady=10)
        w.mainloop()

    def onclick(self):
        self.back = np.asarray([[0] * MNIST_SZ] * MNIST_SZ, dtype=np.uint8)
        self.label.config(text="Draw a number!")
        for cx in range(0, WIN, INC):
            for cy in range(0, WIN, INC):
                rgb = "#%02x%02x%02x" % (MAX, MAX, MAX)
                self.canvas.create_rectangle(cx, cy, cx + 26, cy + 26, fill=rgb, outline=rgb)

    def draw(self, event):
        if event.x == self.x and event.y == self.y:
            return
        
        self.x, self.y = event.x, event.y
        xx = self.x - self.x % INC
        yy = self.y - self.y % INC
        if yy >= WIN or xx >= WIN or yy < 0 or xx < 0:
            return

        self.back[yy//INC][xx//INC] = MAX

        for cx, cy in self.moore:
            cx *= INC
            cy *= INC
            cx += xx
            cy += yy
            if cy >= WIN or cx >= WIN or cy < 0 or cx < 0:
                continue
            rgb = MAX-self.blur()[cy//INC][cx//INC]
            rgb = "#%02x%02x%02x" % (rgb, rgb, rgb)
            self.canvas.create_rectangle(cx, cy, cx + INC, cy + INC, fill=rgb, outline=rgb)
        xv = self.blur()
        p = xv.reshape(xv.shape[0]*xv.shape[1]) / (MAX + 1)
        self.label.config(text=f"{self.nn.classify(p.reshape((1, MNIST_FLAT)))[0]}")

    def blur(self):
        blur = np.zeros((KERNEL_SZ, KERNEL_SZ))
        blur[(KERNEL_SZ - 1)//2, :] = np.ones(KERNEL_SZ)
        blur[:, (KERNEL_SZ - 1)//2] = np.ones(KERNEL_SZ)
        return cv2.filter2D(src=self.back, ddepth=-1, kernel=blur/KERNEL_SZ)
    
if __name__ == "__main__":
    s = Sketch()