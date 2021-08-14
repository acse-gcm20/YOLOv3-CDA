import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from pytorchyolo.test import evaluate_model_file
from sklearn.metrics import auc
from PIL import Image

# Utility function to write lists to a text file
def list_to_file(list_array, filename):
    dims = len(np.shape(list_array))
    
    if dims == 1:
        rows = len(list_array)
        cols = 1
    else:
        rows = len(list_array[0])
        cols = len(list_array)

    with open(filename, 'w') as f:
        for row in range(rows):
            vals = [list_array[col][row] for col in range(cols)]
            f.write(("{} "*cols+'\n').format(*vals))

# Statistics utility functions
def plot_stats(stats_file):
    training_losses = []
    validation_losses = []
    precision_vals = []
    recall_vals = []
    mAP_vals = []

    with open(stats_file) as f:
        for line in f:
            tl, vl, precision, recall = line.rstrip("\n").split()[0:4]
            training_losses.append(float(tl))
            validation_losses.append(float(vl))
            precision_vals.append(float(precision))
            recall_vals.append(float(recall))

            if len(line.rstrip("\n").split()) == 5:
                mAP_vals.append(float(line.rstrip("\n").split()[4]))

    epochs = len(training_losses)

    fig, axs = plt.subplots(2, 2, figsize=(16, 16))

    axs[0,0].plot(range(1, epochs+1), training_losses, label="Training Loss")
    axs[0,0].plot(range(1, epochs+1), validation_losses, label="Validation Loss")
    axs[0,0].set_xlim(1, epochs)
    axs[0,0].set_xlabel("Epoch")
    axs[0,0].set_ylabel("Loss")
    axs[0,0].legend()
    axs[0,0].grid(True)

    axs[0,1].plot(range(1, epochs+1), mAP_vals)
    axs[0,1].set_xlim(1, epochs)
    axs[0,1].set_ylim(-0.1, 1.1)
    axs[0,1].set_xlabel("Epoch")
    axs[0,1].set_ylabel("mAP")
    axs[0,1].grid(True)

    axs[1,0].scatter(range(1, epochs+1), recall_vals)
    axs[1,0].set_xlim(1, epochs)
    axs[1,0].set_ylim(-0.1, 1.1)
    axs[1,0].set_xlabel("Epoch")
    axs[1,0].set_ylabel("Recall")
    axs[1,0].grid(True)

    axs[1,1].scatter(range(1, epochs+1), precision_vals)
    axs[1,1].set_xlim(1, epochs)
    axs[1,1].set_ylim(-0.1, 1.1)
    axs[1,1].set_xlabel("Epoch")
    axs[1,1].set_ylabel("Precision")
    axs[1,1].grid(True)

# Class to calculate, save and plot Precision-Recall curves
class PRcurve:
    def __init__(self, weights_path, start=0.05, stop=0.5, n=10):
        self.confs = np.linspace(start, stop, num=n)
        self.precisions = []
        self.recalls = []

        for i, conf in enumerate(self.confs):
            print("\n%i/%i: Confidence Threshold = %.3f" % (i+1, n, conf))
            metrics = evaluate_model_file('yolov3.cfg', weights_path, 'data/valid.txt', ['crater'], batch_size=32, n_cpu=2, conf_thres=conf, verbose=False)
            self.precisions.append(float(np.mean(metrics[0])))
            self.recalls.append(float(np.mean(metrics[1])))
            print("Precision = %.3f, Recall = %.3f" % (np.mean(metrics[0]), np.mean(metrics[1])))

        list_to_file([self.precisions, self.recalls], "PRstats.txt")
        self.auc_score = auc(self.recalls, self.precisions)

    def save_stats(self, filename):
        list_to_file([self.precisions, self.recalls], filename)

    def plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.plot(self.recalls, self.precisions)
        ax.set_title("Validation Set Precision-Recall Curve, AUC = %.2f" % self.auc_score)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True)

def get_rects(fname, label_dir):
    with open(f'{label_dir}{fname.rstrip(".png")}.txt', 'r') as f:
        labels = f.readlines()
    
    rects = []

    if len(labels[0]) == 1:
        state = labels[0]
        w = float(labels[3])*593
        h = float(labels[4])*593
        x = float(labels[1])*593 - w/2
        y = float(labels[2])*593 - h/2
        rects.append([state, x, y, w, h])
    else:
        for label in labels:
            label = label.rstrip('\n').split(' ')
            state = label[0]
            w = float(label[3])*593
            h = float(label[4])*593
            x = float(label[1])*593 - w/2
            y = float(label[2])*593 - h/2
            coords = (state, x, y, w, h)
            rects.append(coords)   

    return rects

def plot_image_list(image_list_path, image_dir, label_dir, label=True):
    imgs = np.loadtxt(image_list_path, dtype=str)

    fig = plt.figure(figsize=(16,16))

    for i in range(16):
        r = random.randint(0, 139)
        fname = imgs[r]
        im = Image.open(image_dir + fname)
        print(im)

        # Create figure and axes
        ax = fig.add_subplot(4,4,i+1)

        # Display the image
        ax.imshow(im)
        ax.set_title(fname)

        # Create a Rectangle patch
        rects = get_rects(fname, label_dir)
        for coords in rects:
            rect = patches.Rectangle((coords[1], coords[2]), coords[3], coords[4], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if label:
                plt.text(coords[1], coords[2]-10, coords[0])

    plt.show()